import argparse
import datetime
import json
import logging
import os
import traceback
from functools import reduce

import numpy as np
import pandas as pd

import comp_utils
from constants import DATE_RANGES
from data_formats.base import BaseData
from data_formats.energy import EnergyData
from data_formats.energyCDB import EnergyDataCDB
from data_formats.weather import WeatherData
from logger_utils import setup_logger
from math_utils import pinball_loss_and_unc
from model.bdt_quantile_regressor import BDTQuantileRegressorModel
from model.naive_bidding import BiddingModel
from model.stochastic_bidding import StochasticBiddingModel
from model.ssp import SSPModel
from model.optimal_bid import OptimalBid
from plotting_utils import QuantilePrediction, plot_pred_vs_actual
from utils import is_on_terminal, send_message_to_teams, shared_competition_directory
from wind.turbines import WindFarm

try:
    from cdb_pycomm_lib import cdbreader
    from cdb_pycomm_lib import cdbwriter
    import cdb_pycomm_lib.cdbutils as utils
except Exception:
    pass


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Prep all input data, run models and obtain predictions, and submit to competition API!"
    )

    parser.add_argument(
        "--date",
        required=False,
        type=str,
        default=None,
        help="string of csv integers representing todays date, e.g. '2024,1,10'",
    )

    parser.add_argument(
        "--submit", required=False, action="store_true", help="whether to actually submit to API"
    )

    parser.add_argument(
        "--evaluate",
        required=False,
        action="store_true",
        help="calculate mean pinball loss and revenue",
    )

    parser.add_argument(
        "--log_level",
        required=False,
        type=str,
        default="INFO",
        help="verbosity level for logging: 'DEBUG' = high, 'INFO' = low",
    )

    return parser.parse_args()


USECDBFCST = True  # True: use output from EC ensemble 00 forecast from MATLAB / Python models (on CDB curves)
SCALE_WIND_BY_AVAILABILITY = True
MAX_HORNSEA_POWER = 1218.0
REQUIRED_INPUTS = {
    # STATIC INPUTS
    "WIND_SITES": "Input/wind/area_file.geojson",
    "TURBINES": "Input/wind/turbines_file.geojson",
    "TURBINE_CONFIG": "Input/wind/floris/turbines_hornsea_1.yaml",
    "WAKE_MODEL": "Input/wind/floris/default_wake_model_gch.yaml",
    "WAKE_CACHE": "Input/wind/floris/wake_cache_hornsea1_50by50.json",
    "WIND_MODEL": "Wind/wind_model_v1.6/",
    "SOLAR_MODEL": "Solar/solar_model_v2.3/",
    "POWER_MODEL": "Combined/combined_model_v1.6/",
    "SS_PRICE_MODEL": "Input/bidding/best_model_RF.joblib",
    "ENERGY_FILE": "Input/energy/Energy_Data_20240124.csv",
    "AVAILABILITY_FILE": "Input/energy/availability_Hornsea1_latest.parquet",
    "SOLAR_MASK": "Input/solar/solar_mask.parquet",
    # DAILY INPUTS
    "DWD_ICON_EU_WIND": "Input/weather/ICON_Rebase/DATE_DASHED/dwd_icon_eu_hornsea1_DATE_FLAT_0000.nc",
    "DWD_ICON_EU_SOLAR": "Input/weather/ICON_Rebase/DATE_DASHED/dwd_icon_eu_solar_DATE_FLAT_0000.nc",
    "GFS_WIND": "Input/weather/GFS_Rebase/DATE_DASHED/ncep_gfs_hornsea_1DATE_FLAT_0000.nc",
    "GFS_SOLAR": "Input/weather/GFS_Rebase/DATE_DASHED/ncep_gfs_solar_DATE_FLAT_0000.nc",
    # "ENERGY_DATA_CDB": "Input/bidding/EnergyData_CDB_DATE_FLAT.csv",
    "ECO_WIND": "Input/weather/ECoEU3h/DATE_DASHEDT00-00-00/ECoEU3h-Wind_100m_U-Wind_100m_V-Pressure-TotalPrecipitation-DATE_FLAT-DATE_FLAT.parquet",
    "ECO_SOLAR": "Input/weather/ECoEU3h/DATE_DASHEDT00-00-00/ECoEU3h-Temperature-SolarRadiation-TotalCloudCover-DATE_FLAT-DATE_FLAT.parquet",
}


def parse_path(path: str, date: datetime.datetime) -> str:
    """
    Format a templated filepath according to today's date and prepend shared competition directory.
    """
    path_fmt = path.replace("DATE_DASHED", date.strftime("%Y-%m-%d")).replace(
        "DATE_FLAT", date.strftime("%Y%m%d")
    )
    return shared_competition_directory() + path_fmt


def prep(date: datetime.datetime) -> list[str]:
    """
    Fetch all required input data, pre-process (if applicable) and save.
    """
    to_make = []
    for input_name, input_path in REQUIRED_INPUTS.items():
        full_path = parse_path(input_path, date)
        if not os.path.exists(full_path):
            to_make.append(input_name)

    missing = []
    for input_name in to_make:
        if input_name == "DWD_ICON_EU_WIND":
            # call to cdb_pycomm_lib
            missing.append(input_name)
            continue
        elif input_name == "DWD_ICON_EU_SOLAR":
            # call to cdb_pycomm_lib
            missing.append(input_name)
            continue
        else:
            missing.append(input_name)

    return missing


def predict(date: datetime.datetime, USECDBFCST: bool) -> pd.DataFrame:
    """
    Run all model predictions.
    """
    if not USECDBFCST:
        # 1. Wind data
        # 1.1 Load DWD_ICON_EU forecast
        wind_data_icon = WeatherData(
            variable_names=["Wind_100m_ABS", "Wind_100m_DIR", "Temperature", "RelativeHumidity"],
            source_filepath=parse_path(REQUIRED_INPUTS["DWD_ICON_EU_WIND"], date),
            aggregate_lat_lon="summarize",
            is_forecast=True,
            keep_nearest=True,
            interpolate_to_30min_res=True,
        )
        # wind_data_icon.data.columns = [x + "_icon" if x not in ["fD", "vD"] else x for x in wind_data_icon.data.columns]

        # 1.2 Load GFS forecast
        wind_data_gfs = WeatherData(
            variable_names=["Wind_100m_ABS", "Wind_100m_DIR", "Temperature", "RelativeHumidity"],
            source_filepath=parse_path(REQUIRED_INPUTS["GFS_WIND"], date),
            aggregate_lat_lon="summarize",
            is_forecast=True,
            keep_nearest=True,
            interpolate_to_30min_res=True,
        )
        wind_data_gfs.data.columns = [
            x + "_gfs" if x not in ["fD", "vD"] else x for x in wind_data_gfs.data.columns
        ]

        wind_data_eco = WeatherData(
            variable_names=["Wind_100m_ABS", "Wind_100m_DIR"],
            source_filepath=parse_path(REQUIRED_INPUTS["ECO_WIND"], date),
            aggregate_lat_lon="summarize",
            is_forecast=True,
            keep_nearest=True,
            interpolate_to_30min_res=True,
        )

        wind_data_eco.data.columns = [
            x + "_ec" if x not in ["fD", "vD"] else x for x in wind_data_eco.data.columns
        ]

        # 1.2 Add vanilla power pred
        farm = WindFarm(  # ! SAM: last value date fix ***
            name="Hornsea 1",
            wind_sites_file=parse_path(REQUIRED_INPUTS["WIND_SITES"], date),
            turbines_file=parse_path(REQUIRED_INPUTS["TURBINES"], date),
            turbine_config_file=parse_path(REQUIRED_INPUTS["TURBINE_CONFIG"], date),
            wake_model_file=parse_path(REQUIRED_INPUTS["WAKE_MODEL"], date),
        )
        wind_data_eco.add_vanilla_power_curve_prediction(
            turbine_set=farm, wind_speed_name="Wind_100m_ABS_ec"
        )

        # 1.3 Add wake-adj power pred
        farm.create_cache(  # will not rerun assuming `file_path` already exists
            wind_speeds=np.linspace(0.1, 35.0, 50),
            wind_directions=np.linspace(0.0, 360.0, 50),
            file_path=parse_path(REQUIRED_INPUTS["WAKE_CACHE"], date),
        )
        wind_data_eco.add_wakeadj_power_curve_prediction(
            wind_farm=farm,
            wind_speed_name="Wind_100m_ABS_ec",
            wind_direction_name="Wind_100m_DIR_ec",
        )

        # 1.4 Get wind model predictions
        wind_data = wind_data_eco.data.merge(
            wind_data_icon.data.reset_index(), left_on="vD", right_on="vD", how="inner"
        )
        wind_data = wind_data.merge(
            wind_data_gfs.data.reset_index(), left_on="vD", right_on="vD", how="inner"
        )

        # 2. Solar data
        solar_data_icon = WeatherData(
            variable_names=["TotalCloudCover", "SolarRadiation", "Temperature"],
            source_filepath=parse_path(REQUIRED_INPUTS["DWD_ICON_EU_SOLAR"], date),
            aggregate_lat_lon="summarize",
            is_forecast=True,
            keep_nearest=True,
            interpolate_to_30min_res=True,
            prep_solar=True,
        )

        solar_data_icon.add_hour_and_harmonics(
            anchor_date=pd.to_datetime("2020-01-01 00:00:00").tz_localize("UTC")
        )

        solar_data_gfs = WeatherData(
            variable_names=["TotalCloudCover", "SolarRadiation", "Temperature"],
            source_filepath=parse_path(REQUIRED_INPUTS["GFS_SOLAR"], date),
            aggregate_lat_lon="summarize",
            is_forecast=True,
            keep_nearest=True,
            interpolate_to_30min_res=True,
            prep_solar=True,
        )

        solar_data_ec = WeatherData(
            variable_names=["TotalCloudCover", "SolarRadiation", "Temperature"],
            source_filepath=parse_path(REQUIRED_INPUTS["ECO_SOLAR"], date),
            aggregate_lat_lon="summarize",
            is_forecast=True,
            keep_nearest=True,
            interpolate_to_30min_res=True,
            accumulated_variables=["SolarRadiation"],
            prep_solar=True,
        )

        solar_mask = pd.read_parquet(parse_path(REQUIRED_INPUTS["SOLAR_MASK"], date)).reset_index(
            drop=True
        )

        # 3. Merge wind and solar
        solar_data_icon.data.columns = [
            x + "_pes10" if x not in ["fD", "vD"] else x for x in solar_data_icon.data.columns
        ]
        solar_data_gfs.data.columns = [
            x + "_pes10_gfs" if x not in ["fD", "vD"] else x for x in solar_data_gfs.data.columns
        ]

        solar_data_ec.data.columns = [
            x + "_pes10_ec" if x not in ["fD", "vD"] else x for x in solar_data_ec.data.columns
        ]

        solar_data = solar_data_icon.data.merge(
            solar_data_gfs.data.reset_index(), left_on="vD", right_on="vD", how="inner"
        )
        solar_data = solar_data.merge(
            solar_data_ec.data.reset_index(), left_on="vD", right_on="vD", how="inner"
        )

        merged_data = wind_data.merge(solar_data.reset_index(), left_on="vD", right_on="vD").merge(
            solar_mask, left_on="vD", right_on="vD", how="left"
        )

        # 4. Load latest models in 'prod' and run forecasts
        power_model = BDTQuantileRegressorModel(
            input_dir=parse_path(REQUIRED_INPUTS["POWER_MODEL"], date),
            min_value=0.0,
            max_value=1800.0,
        )
        power_model.load()

        wind_model = BDTQuantileRegressorModel(
            input_dir=parse_path(REQUIRED_INPUTS["WIND_MODEL"], date),
            min_value=0.0,
            max_value=1218.0,
        )
        wind_model.load()

        solar_model = BDTQuantileRegressorModel(
            input_dir=parse_path(REQUIRED_INPUTS["SOLAR_MODEL"], date),
            min_value=0.0,
            max_value=1.0,
        )
        solar_model.load()

        ## Wind forecast
        wind_prediction = wind_model.predict(df=merged_data)

        # Scale wind prediction by availability
        availability_data = BaseData(
            variable_names=["availability_HornseaG1_HornseaG2_HornseaG3"],
            source_filepath=parse_path(REQUIRED_INPUTS["AVAILABILITY_FILE"], date),
        )
        availability_data.data = availability_data.data.reset_index()[
            ["vD", "availability_HornseaG1_HornseaG2_HornseaG3"]
        ]
        # Adjustment for q50 bias
        availability_data.data["availability_HornseaG1_HornseaG2_HornseaG3"] *= 1.046
        merged_data = merged_data.merge(
            availability_data.data, left_on="vD", right_on="vD", how="left"
        )
        merged_data.ffill(inplace=True)
        merged_data["wind_power_sf"] = (
            merged_data["availability_HornseaG1_HornseaG2_HornseaG3"] / MAX_HORNSEA_POWER
        )

        wind_prediction = wind_prediction * merged_data["wind_power_sf"].to_numpy().reshape(-1, 1)

        ## Solar forecast
        solar_prediction = solar_model.predict(df=merged_data)

        # Retrieve latest capacity data
        solar_capacity_data = cdbreader.get_curves(
            156484286, date - datetime.timedelta(days=1), date + datetime.timedelta(days=1)
        )
        # Pick latest value
        latest_solar_capacity = solar_capacity_data.sort_values("VD")["V"][0]
        # Backtransform solar_prediction, and clean with respects to SOLAR_MASK
        solar_prediction = (
            solar_prediction
            * latest_solar_capacity
            * merged_data["Solar_Mask"].values.reshape(merged_data["Solar_Mask"].size, 1)
        )

        ## Sum wind and solar predictions together!
        quantiles_prediction = wind_prediction + solar_prediction

        # quantiles_prediction = power_model.predict(df=merged_data)
        quantiles_prediction = pd.DataFrame(
            quantiles_prediction,
            index=merged_data.index,
            columns=[f"q{quantile}" for quantile in range(10, 100, 10)],
        )
        quantiles_prediction = merged_data[["fD_x_x", "vD"]].merge(
            quantiles_prediction, left_index=True, right_index=True
        )
        quantiles_prediction.drop(columns=["fD_x_x"], inplace=True)

        predictions = quantiles_prediction

        # 4.4 Create a df with wind and solar quantiles for the bidding section
        wind_solar_prediction = (
            pd.DataFrame(
                wind_prediction,
                index=merged.index,
                columns=[f"wind_q{quantile}" for quantile in range(10, 100, 10)],
            )
            .merge(
                pd.DataFrame(
                    solar_prediction,
                    index=merged.index,
                    columns=[f"solar_q{quantile}" for quantile in range(10, 100, 10)],
                ),
                left_index=True,
                right_index=True,
            )
            .merge(merged["vD"], left_index=True, right_index=True)
        )

    else:
        # 1-3. Quantiles from CDB curves updated by WindSolarQuantilesGBR model
        qt = {}  # here all quantiles fetched from CDB curves will be stored
        fD = date
        # fD = datetime.datetime(date.year, date.month, date.day - 1, 18)

        
        # quantiles based on weather paths
        quantiles_cdb_curves = {
            "q10": 157171539,
            "q20": 157171538,
            "q30": 157171547,
            "q40": 157171541,
            "q50": 157171546,
            "q60": 157171542,
            "q70": 157171543,
            "q80": 157171540,
            "q90": 157171544,
        }

        """
        # sum of wind and solar quantiles
        quantiles_cdb_curves = {
            "q10": 157530374,
            "q20": 157530372,
            "q30": 157530379,
            "q40": 157530377,
            "q50": 157530373,
            "q60": 157530378,
            "q70": 157530380,
            "q80": 157530375,
            "q90": 157530381,
        }
        """
        data_from_cdb = cdbreader.get_curves(
            [quantiles_cdb_curves[k] for k in quantiles_cdb_curves.keys()], fd_from=fD, fd_to=fD
        )

        for k in quantiles_cdb_curves.keys():
            qt[k] = (
                data_from_cdb[data_from_cdb["ID"].isin([quantiles_cdb_curves[k]])]
                .rename(columns={"V": k})
                .drop(["ID", "FD"], axis="columns")
            )

        quantiles_prediction = reduce(
            lambda left, right: pd.merge(left, right, on=["VD"], how="outer"),
            [qt[k] for k in qt.keys()],
        )

        dates_to_keep = pd.DataFrame(
            {"VD": comp_utils.day_ahead_market_times(today_date=date).tz_convert("UTC")}
        )
        quantiles_prediction["VD"] = quantiles_prediction["VD"].dt.tz_localize(
            "UTC"
        )  # to be converted to "Europe/London" in submit()
        mask = quantiles_prediction["VD"].isin(dates_to_keep["VD"])
        quantiles_prediction = quantiles_prediction[mask]
        quantiles_prediction = (
            quantiles_prediction.rename(columns={"VD": "vD"}).set_index("vD").reset_index()
        )
        predictions = quantiles_prediction
        predictions["market_bid"] = predictions["q50"]
        # real_submit = True

        """

        # wind and solar quantiles
        wind_solar_cdb_curves = {
            "wind_q10": 158750785,
            "wind_q20": 158750786,
            "wind_q30": 158750787,
            "wind_q40": 158750788,
            "wind_q50": 158750798,
            "wind_q60": 158750781,
            "wind_q70": 158750793,
            "wind_q80": 158750783,
            "wind_q90": 158750789,
            "solar_q10": 158750791,
            "solar_q20": 158750795,
            "solar_q30": 158750794,
            "solar_q40": 158750782,
            "solar_q50": 158750796,
            "solar_q60": 158750792,
            "solar_q70": 158750784,
            "solar_q80": 158750797,
            "solar_q90": 158750790,
        }

        wind_solar_data_from_cdb = cdbreader.get_curves(
            [wind_solar_cdb_curves[k] for k in wind_solar_cdb_curves.keys()], fd_from=fD, fd_to=fD
        )

        winsolqt = {}
        for k in wind_solar_cdb_curves.keys():
            winsolqt[k] = (
                wind_solar_data_from_cdb[
                    wind_solar_data_from_cdb["ID"].isin([wind_solar_cdb_curves[k]])
                ]
                .rename(columns={"V": k})
                .drop(["ID", "FD"], axis="columns")
            )

        wind_solar_prediction = reduce(
            lambda left, right: pd.merge(left, right, on=["VD"], how="inner"),
            [winsolqt[k] for k in winsolqt.keys()],
        )

        wind_solar_prediction["VD"] = wind_solar_prediction["VD"].dt.tz_localize("UTC")
        wind_solar_prediction = wind_solar_prediction.rename(columns={"VD": "vD"})

    # 4. DA_Price and SS_Price data
    energy_data = EnergyDataCDB(runDate=date)

    # This should be skipped when running back in time, and just read the saved file instead
    try:
        energy_data.read_data_cdb()
        energy_data.save_data()
    except Exception as exc:
        pass

    df_energy_data = energy_data.prepare_input_data(pd.read_csv(energy_data.path_save))
    ssp_model = SSPModel(model_name="q", runDate=date)  # SSPModel quantiles
    price_data, quantiles = ssp_model.forecast_stochastic(df_energy_data)

    ssp_model_det = SSPModel(runDate=date)  # RF deterministic
    price_data_det = ssp_model_det.forecast_deterministic(df_energy_data)

    price_data.to_csv(
        ssp_model.result_path
    )  # Save the complete forecast (more than needed for the submission)

    merged_data = pd.merge(predictions, price_data, how="inner", on="vD")
    merged_data_det = pd.merge(predictions, price_data_det, how="inner", on="vD")

    if not USECDBFCST:
        meta = ";Pwr.GB;MW;F;PC;PRO;Energy;ECOp;v1.1;UTC;min.30;H.6;"
        columns = {}
        for qu in range(10, 100, 10):
            columns[f"q{qu}"] = meta + f";Percentile.{qu};"
        predictions_to_cdb = predictions.copy()
        predictions_to_cdb = predictions_to_cdb.rename(columns=columns).set_index("vD")
        write_model_output_to_cdb(
            date, predictions_to_cdb, "HEFTcom24.WindSolarQuantilesGBR.Forecast30min", True
        )

    # 5. Run bidding model predictions

    ## 5.A Bidding using SSP
    SSP_scenarios = {}
    for i in quantiles.keys():
        SSP_scenarios.update({i: (quantiles[i], merged_data["SSP_forecast_{:s}".format(i)])})
    stoch_bid = StochasticBiddingModel()
    bidsq50gen_SSPstochastic = stoch_bid.solve_optimization(
        DA_PriceGBR=merged_data["V_spotPr_EC0030min"],
        ACT=merged_data["q50"],
        SSP_scenarios=SSP_scenarios,
    )
    bidsq40gen_SSPstochastic = stoch_bid.solve_optimization(
        DA_PriceGBR=merged_data["V_spotPr_EC0030min"],
        ACT=merged_data["q40"],
        SSP_scenarios=SSP_scenarios,
    )
    bidsq60gen_SSPstochastic = stoch_bid.solve_optimization(
        DA_PriceGBR=merged_data["V_spotPr_EC0030min"],
        ACT=merged_data["q60"],
        SSP_scenarios=SSP_scenarios,
    )

    SSP_deterministic = {"deterministic": (1, merged_data_det["SSP_forecast"])}
    bidsq50gen_SSPdeterminitsicRF = stoch_bid.solve_optimization(
        DA_PriceGBR=merged_data["V_spotPr_EC0030min"],
        ACT=merged_data["q50"],
        SSP_scenarios=SSP_deterministic,
    )

    ## 5.B Bidding using optimal model bid
    bid_optimal = OptimalBid(model_name="RF", runDate=date)
    df_energy_data_opt = df_energy_data.copy(deep=True)
    df_energy_data_opt = pd.merge(df_energy_data_opt, merged_data["vD"], how="inner", on="vD")
    bidsOptimalRF = bid_optimal.predict(df_energy_data_opt)

    ## 6.A Merge with predictions, making sure we use the correct timestamps
    predictions = predictions.merge(
        df_energy_data_opt[["vD"]], left_on="vD", right_on="vD", how="inner"
    )

    ## 6.B Bidding using generation directly
    bidsq50 = predictions["q50"].to_numpy()
    bidsq40 = predictions["q40"].to_numpy()
    bidsq60 = predictions["q60"].to_numpy()

    ## 6.C Define the bids that we want to submit
    predictions["market_bid"] = bidsq40

    # 7. Create a df to write to CDB after
    Outdf = merged_data.copy(deep=True).drop(
        [
            "q10",
            "q20",
            "q30",
            "q40",
            "q50",
            "q60",
            "q70",
            "q80",
            "q90",
            "V_spotPr_EC0030min",
            "V_wind_EC0030min",
            "V_solar_EC0030min",
            "V_demand_EC0030min",
            "V_netImport_EC0030min",
            "res_load",
        ],
        axis=1,
    )
    meta = ";Pwr.GB;SSP;Price;F;PC;UTC;min.30;"

    for old_col in Outdf.columns:
        if old_col.startswith("SSP_forecast_q"):
            # Extract the numeric part (e.g., '10', '20', '50') from the column name
            percentile = old_col.split("_")[2][1:]
            new_col = f"{meta};Percentile.q0.{percentile};"  # ;Pwr.GB;SSP;Price;F;PC;UTC;min.30;
            Outdf.rename(columns={old_col: new_col}, inplace=True)

    Outdf[meta + "RF;"] = merged_data_det["SSP_forecast"].values.tolist()

    Outdf[
        (
            meta + "bids;q50gen_SSPstochastic;CDB?" + str(USECDBFCST)
            if USECDBFCST
            else meta + "bids;q50gen_SSPstochastic;CDB?" + str(USECDBFCST) + "_v1.1"
        )
    ] = bidsq50gen_SSPstochastic.tolist()
    Outdf[
        (
            meta + "bids;q40gen_SSPstochastic;CDB?" + str(USECDBFCST)
            if USECDBFCST
            else meta + "bids;q40gen_SSPstochastic;CDB?" + str(USECDBFCST) + "_v1.1"
        )
    ] = bidsq40gen_SSPstochastic.tolist()
    Outdf[
        (
            meta + "bids;q60gen_SSPstochastic;CDB?" + str(USECDBFCST)
            if USECDBFCST
            else meta + "bids;q60gen_SSPstochastic;CDB?" + str(USECDBFCST) + "_v1.1"
        )
    ] = bidsq60gen_SSPstochastic.tolist()
    Outdf[
        (
            meta + "bids;q50gen_SSPdeterminitsicRF;CDB?" + str(USECDBFCST)
            if USECDBFCST
            else meta + "bids;q50gen_SSPdeterminitsicRF;CDB?" + str(USECDBFCST) + "_v1.1"
        )
    ] = bidsq50gen_SSPdeterminitsicRF.tolist()

    Outdf[
        (
            meta + "bids;q50gen;CDB?" + str(USECDBFCST)
            if USECDBFCST
            else meta + "bids;q50gen;CDB?" + str(USECDBFCST) + "_v1.1"
        )
    ] = bidsq50.tolist()
    Outdf[
        (
            meta + "bids;q40gen;CDB?" + str(USECDBFCST)
            if USECDBFCST
            else meta + "bids;q40gen;CDB?" + str(USECDBFCST) + "_v1.1"
        )
    ] = bidsq40.tolist()
    Outdf[
        (
            meta + "bids;q60gen;CDB?" + str(USECDBFCST)
            if USECDBFCST
            else meta + "bids;q60gen;CDB?" + str(USECDBFCST) + "_v1.1"
        )
    ] = bidsq60.tolist()
    Outdf[
        (
            meta + "bids;optimalRF;CDB?" + str(USECDBFCST)
            if USECDBFCST
            else meta + "bids;optimalRF;CDB?" + str(USECDBFCST) + "_v1.1"
        )
    ] = bidsOptimalRF.tolist()

    csv_path = (
        shared_competition_directory()
        + "Output/Benchmark/"
        + "Benchmark"
        + str(date.year)
        + str(date.month)
        + str(date.day)
        + ".csv"
    )
    Outdf.to_csv(
        csv_path,
        index=False,
    )
    Outdf.set_index("vD", inplace=True)
    write_model_output_to_cdb(date, Outdf, "HEFTcom24.WindSolarQuantilesGBR.Forecast30min", True)
    """
    return predictions


def submit(
    predictions: pd.DataFrame, date: datetime.datetime, real_submit: bool = False
) -> tuple[pd.DataFrame, dict]:
    """
    Format predictions according to competition's instructions and submit.
    """
    real_submit = True
    # 1. Convert to London time zones
    predictions["vD"] = predictions["vD"].dt.tz_convert("Europe/London")
    # 1.1 Manually insert last entry into dataframe
    # NOTE: to see why this is necessary, see: https://stackoverflow.com/questions/73730498/also-ffill-last-value-when-resampling-in-pandas
    last_date = predictions.loc[len(predictions) - 1, "vD"]
    if last_date.minute != 30:
        other_cols = [x for x in predictions.columns if x != "vD"]
        predictions.loc[len(predictions), "vD"] = last_date + pd.Timedelta(0.5, unit="hours")
        predictions.ffill(inplace=True)

    # 2. Get day ahead market times and merge predictions for those dates
    submission = pd.DataFrame({"datetime": comp_utils.day_ahead_market_times(today_date=date)})
    submission = submission.merge(predictions, how="left", left_on="datetime", right_on="vD")

    # 3. Clean: keep only relevant columns, force dtype float64, replace NaNs with 0s
    columns = ["datetime", "market_bid"] + [f"q{quantile}" for quantile in range(10, 100, 10)]
    submission = submission[columns]
    for col in submission.columns:
        if col == "datetime":
            continue
        submission[col] = submission[col].astype("float64", copy=False)
    submission.fillna(0, inplace=True)

    submission_json = comp_utils.prep_submission_in_json_format(submission)

    rebase_api_client = comp_utils.RebaseAPI(api_key=open("team_key.txt").read())

    if real_submit:
        rebase_api_client.submit(submission_json)

    return submission, submission_json


def write_model_output_to_cdb(
    date: datetime.datetime, dataframe: pd.DataFrame, loadsetname: str, isforecast: bool
):
    fD = date
    date_only = fD.replace(hour=0, minute=0, second=0, microsecond=0)
    utils.cdb.CDBLOGGER.modelrun.rundatetime = date_only

    dataframe.index = dataframe.index.strftime("%Y-%m-%dT%H:%M:%S")
    curvetype = 2 if isforecast else 1
    session = cdbwriter.Loadset(loadsetname=loadsetname, curvetype=curvetype)
    session.add(dataframe)
    session.send()


def run_pipeline(USECDBFCST: bool, date: str, real_submit: bool) -> pd.DataFrame:
    """
    Prep, predict, and submit.
    """
    # 1. Logger
    logger = logging.getLogger("heft." + __name__)

    # 2. Parse date, or take today's date if not given
    if date is not None:  # user provided date
        today = False
        yyyy, mm, dd = (int(x) for x in date.split(","))
    else:  # no date provided, use today's date
        today = datetime.date.today()
        yyyy, mm, dd = today.year, today.month, today.day
    date = datetime.datetime(yyyy, mm, dd)
    logger.info("Running submission pipeline for date '{:s}'.".format(date.strftime("%Y-%m-%d")))

    # 3. Fetch all required input data, pre-process (if appicable), and save.
    if not USECDBFCST:
        missing_inputs = prep(date=date)
        if missing_inputs:
            for mi in missing_inputs:
                logger.warning(
                    "Unable to make required input '{:s}' with target file '{:s}'.".format(
                        mi, parse_path(REQUIRED_INPUTS[mi], date)
                    )
                )

    # 4. Run all model predictions
    try:
        predictions = predict(date, USECDBFCST)
    except:  # use EC12 forecast from backup csv file
        predictions = pd.read_csv(
            shared_competition_directory() + f"/Output/backup/{yyyy}{mm}{dd}_EC12.csv"
        )
        predictions["vD"] = pd.to_datetime(predictions["vD"], errors="coerce")

    # 5. Format predictions and submit to API!
    real_submit = True
    if real_submit:
        logger.warning("Submitting entry to competition API!")
    else:
        logger.warning("Not actually submitting entry to competition API!")

    submission_df, submission = submit(
        predictions=predictions, date=date.date(), real_submit=real_submit
    )

    # 6. Plots and debugging printouts
    quantiles = np.linspace(0.1, 0.9, 9)
    y_true = np.zeros(len(submission_df)).reshape(-1, 1)  # dummy actuals
    y_pred = np.array(
        [submission_df[f"q{quantile}"].to_numpy() for quantile in range(10, 100, 10)]
    ).transpose()
    qp = QuantilePrediction(y_true=y_true, y_pred=y_pred, quantiles=quantiles, name="Model")
    plot_pred_vs_actual(
        y_true=submission_df["market_bid"],
        preds=[qp],
        output_file="predictions.png",
        dates=submission_df["datetime"],
        submission_style=True,
    )

    logger.debug(
        "Contents of submission json for 'market_day' : '{:s}'".format(submission["market_day"])
    )
    for entry in submission["submission"]:
        logger.debug("Timestamp : '{:s}'".format(entry["timestamp"]))
        logger.debug(
            "\t probabilistic forecast : {:s}".format(str(entry["probabilistic_forecast"]))
        )
        logger.debug("\t market bid : {:s}".format(str(entry["market_bid"])))

    logger.info("Success!")  # ! make dependent on submission API response

    return submission_df


# TODO: should change below to a continuously growing csv where we append latest energy data
ENERGY_FILE = shared_competition_directory() + "Input/energy/Energy_Data_20240124.csv"
ENERGY_COLS = [
    "Total_MWh_credit",
    "DA_Price",
    "SS_Price",
    "vD",
]


def evaluate():
    """
    Calculate mean pinball loss and revenue for all dates for which data is available.
    """
    logger = logging.getLogger("heft." + __name__)

    energy_data = EnergyData(source_filepath=ENERGY_FILE)
    energy_data = energy_data.data.reset_index()[ENERGY_COLS]

    energy_data = energy_data.loc[energy_data["vD"] >= DATE_RANGES["VAL"][0]]

    dates = np.unique([x.date() for x in energy_data["vD"].to_numpy()])
    predictions = None
    # NOTE: doing this in a loop is not computationally efficient, but enforces a nice consistency between the submission and evaluation pipelines
    for uniq_date in dates:
        date_str = "{:d},{:d},{:d}".format(uniq_date.year, uniq_date.month, uniq_date.day)

        missing_files = prep(uniq_date)
        if missing_files:  # missing inputs, skip
            logger.warning(
                "Missing input(s) for date '{:s}', skipping.".format(uniq_date.strftime("%Y-%m-%d"))
            )
            for mf in missing_files:
                logger.info("\t Input file '{:s}' is missing.".format(mf))
            continue

        preds_date = run_pipeline(USECDBFCST, date=date_str, real_submit=False)
        if predictions is None:
            predictions = preds_date
        else:
            predictions = pd.concat((predictions, preds_date))

    energy_data["vD"] = energy_data["vD"].dt.tz_convert("Europe/London")
    predictions = predictions.merge(energy_data, how="inner", left_on="datetime", right_on="vD")

    for date_range_name, (d_start, d_end) in DATE_RANGES.items():
        if not date_range_name == "VAL":
            continue
        preds_range = predictions[(predictions["vD"] >= d_start) & (predictions["vD"] < d_end)]

        y_true = preds_range["Total_MWh_credit"].to_numpy().reshape(-1, 1)
        y_pred = np.array(
            [preds_range[f"q{quantile}"].to_numpy() for quantile in range(10, 100, 10)]
        ).transpose()
        quantiles = np.linspace(0.1, 0.9, 9)

        pb_loss, pb_loss_unc = pinball_loss_and_unc(
            y_true=y_true, y_pred=y_pred, quantiles=quantiles
        )

        revenue, revenue_unc = BiddingModel.revenue_and_unc(
            bids=preds_range["market_bid"].to_numpy(),
            actuals=preds_range["Total_MWh_credit"].to_numpy(),
            da_price=preds_range["DA_Price"].to_numpy(),
            ss_price=preds_range["SS_Price"].to_numpy(),
        )

        logger.info(
            "Performance over date range '{:s}' (spanning from '{:s}' to '{:s}'):".format(
                date_range_name, d_start.strftime("%Y-%m-%d"), d_end.strftime("%Y-%m-%d")
            )
        )
        logger.info("\t mean pinball loss of {:.3f} +/- {:.3f}.".format(pb_loss, pb_loss_unc))
        logger.info("\t mean revenue of {:.1f} +/- {:.1f}.".format(revenue, revenue_unc))

        n_splits = int(float(len(y_true)) / 125.0) + 1
        y_true_splits = np.array_split(y_true, n_splits)
        y_pred_splits = np.array_split(y_pred, n_splits)
        date_splits = np.array_split(predictions["datetime"], n_splits)
        for i in range(n_splits):
            qp = QuantilePrediction(
                y_true=y_true_splits[i], y_pred=y_pred_splits[i], quantiles=quantiles, name="Model"
            )

            plot_pred_vs_actual(
                y_true=y_true_splits[i].flatten(),
                preds=[qp],
                output_file=shared_competition_directory()
                + "Output/plots/pred_vs_actual_{:s}_{:d}.png".format(date_range_name, i),
                dates=date_splits[i],
            )


if __name__ == "__main__":
    args = parse_arguments()
    logger = setup_logger(args.log_level)

    if not args.evaluate:
        if is_on_terminal():
            try:
                run_pipeline(args.date, args.submit, USECDBFCST)
                # ! make message texts i) show the API submission response, ii) take the filename generically, iii) take when the filename was fetched (some git log)
                message_text = (
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    + " ("
                    + datetime.datetime.now(datetime.timezone.utc).astimezone().tzname()
                    + ")"
                    + ": SUCCESS of scheduled run of "
                    "submission_pipeline_v2.py"
                    " (taken from origin, Jan 30th)."
                )
            except Exception as exc:
                message_text = (
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    + " ("
                    + datetime.datetime.now(datetime.timezone.utc).astimezone().tzname()
                    + ")"
                    + ": FAILURE of scheduled run of "
                    "submission_pipeline_v2.py"
                    " (taken from origin, Jan 30th).\n\n\n" + traceback.format_exc()
                )
            message_text = message_text.replace("\n", "\r\n")
            response = send_message_to_teams(message_text)
        else:
            run_pipeline(USECDBFCST, args.date, args.submit)
    else:
        evaluate()
