import comp_utils
from comp_utils import RebaseAPI
from datetime import datetime, timedelta
from datetime import date
import pandas as pd
import numpy as np
from numpy import dtype
import xarray as xr
import statsmodels.formula.api as smf
from statsmodels.iolib.smpickle import load_pickle

import cdb_pycomm_lib as cdb
from cdb_pycomm_lib import cdbutils as utils
from cdb_pycomm_lib import cdbwriter as pjapi
import cdb_pycomm_lib.cdbreader as pycomm


class PrepareOutput(RebaseAPI):

    def __init__(self):
        super().__init__(api_key=open("team_key.txt").read())
        self.today = date.today()

    def get_dates(self, d1: int, d2: int):
        d1 = self.today + timedelta(days=-d1)
        d1 = d1.strftime("%Y-%m-%d")
        d2 = self.today + timedelta(days=-d2)
        d2 = d2.strftime("%Y-%m-%d")
        start_date = datetime.strptime(d1, "%Y-%m-%d")
        end_date = datetime.strptime(d2, "%Y-%m-%d")
        date_list = pd.date_range(start_date, end_date, freq="D")
        return date_list

    def get_values(self, var: str, date_list, df: pd.DataFrame) -> pd.DataFrame:
        for date in date_list:
            variables = self.get_variable(date.strftime("%Y-%m-%d"), var)
            df = pd.concat([df, variables], axis=0)
        return df


d = PrepareOutput()
vars = {}  # Other variables
variables = [
    "solar_total_production",
    "wind_total_production",
    "day_ahead_price",
    "imbalance_price",
    "market_index",
]

#variables = [
#    "wind_total_production",
#]

for var in variables:
    match var:
        case "solar_total_production":
            date_list = d.get_dates(30, 1)
            df = pd.DataFrame(
                columns=["timestamp_utc", "generation_mw", "installed_capacity_mwp", "capacity_mwp"]
            )
            df = d.get_values(var, date_list, df)
            var_split = var.split("_")
            output = df.set_index("timestamp_utc")
            output["solar_MWh_credit"] = 0.5 * output["generation_mw"]
            output = output.rename(
                columns={
                    "generation_mw": f";generation_mw;{var_split[0]};UTC;MW;A;",
                    "installed_capacity_mwp": f";installed_capacity_mwp;{var_split[0]};UTC;MW;A;",
                    "capacity_mwp": f";capacity_mwp;{var_split[0]};UTC;MW;A;",
                    "solar_MWh_credit": f";solar_MWh_credit;{var_split[0]};UTC;MWh;A;",
                },
                index={
                    "timestamp_utc": "",
                },
            )
            output.index = [t[:-1] for t in output.index]

            # Extend solar capacities and write to CDB
            val_inst_cap = output[output.columns[1]][-1]
            val_cap = output[output.columns[2]][-1]
            dat = output.index[-1]
            dates = pd.date_range(dat, periods=961, freq="30min")  # 961 = 48*20 + 1
            dates = dates.strftime("%Y-%m-%dT%H:%M:%S")
            dates = dates[1:]
            y_inst_cap = np.repeat(val_inst_cap, len(dates))
            y_cap = np.repeat(val_cap, len(dates))
            data = {f"{output.columns[1]}": y_inst_cap, f"{output.columns[2]}": y_cap}
            dataframe = pd.DataFrame(data, index=dates, columns=[k for k in data.keys()])
            session = pjapi.Loadset(loadsetname="Test.Py.HEFTcom24", curvetype=1)
            session.add(dataframe)
            session.send()
            output.drop(columns=[output.columns[1], output.columns[2]], inplace=True)

        case "wind_total_production":
            date_list = d.get_dates(30, 1)
            df = pd.DataFrame(
                columns=[
                    "timestamp_utc",
                    "settlement_date",
                    "settlement_period",
                    "boa",
                    "generation_mw",
                ]
            )
            df = d.get_values(var, date_list, df)
            var_split = var.split("_")
            output = df.set_index("timestamp_utc")
            output["wind_MWh_credit"] = 0.5 * output["generation_mw"] - output["boa"]
            output = output.rename(
                columns={
                    "generation_mw": f";generation_mw;{var_split[0]};UTC;MW;A;",
                    "boa": f";boa;{var_split[0]};UTC;MWh;A;",
                    "wind_MWh_credit": f";wind_MWh_credit;{var_split[0]};UTC;MWh;A;",
                },
                index={
                    "timestamp_utc": "",
                },
            )
            output.index = [t[:-1] for t in output.index]
            output.drop(columns=["settlement_date", "settlement_period"], inplace=True)

        case "day_ahead_price":
            date_list = d.get_dates(30, 1)
            df = pd.DataFrame(
                columns=["timestamp_utc", "settlement_date", "settlement_period", "price"]
            )
            df = d.get_values(var, date_list, df)
            output = df.set_index("timestamp_utc")
            output = output.rename(
                columns={
                    "price": f";day_ahead_price;UTC;EUR/MWh;A;",
                },
                index={
                    "timestamp_utc": "",
                },
            )
            output.index = [t[:-1] for t in output.index]
            output.drop(columns=["settlement_date", "settlement_period"], inplace=True)

        case "imbalance_price":
            date_list = d.get_dates(30, 1)
            df = pd.DataFrame(
                columns=["timestamp_utc", "settlement_date", "settlement_period", "imbalance_price"]
            )
            df = d.get_values(var, date_list, df)
            output = df.set_index("timestamp_utc")
            output = output.rename(
                columns={
                    "imbalance_price": f";imbalance_price;UTC;EUR/MWh;A;",
                },
                index={
                    "timestamp_utc": "",
                },
            )
            output.index = [t[:-1] for t in output.index]
            output.drop(columns=["settlement_date", "settlement_period"], inplace=True)

        case "market_index":
            date_list = d.get_dates(30, 1)
            df = pd.DataFrame(
                columns=[
                    "timestamp_utc",
                    "settlement_date",
                    "settlement_period",
                    "data_provider",
                    "price",
                    "volume",
                ]
            )
            df = d.get_values(var, date_list, df)
            output = df.set_index("timestamp_utc")
            output = output.rename(
                columns={
                    "price": f";price;APXMIDP;UTC;EUR/MWh;A;",
                    "volume": f";volume;APXMIDP;UTC;MWh;A;",
                },
                index={
                    "timestamp_utc": "",
                },
            )
            output.index = [t[:-1] for t in output.index]
            output.drop(
                columns=["settlement_date", "settlement_period", "data_provider"], inplace=True
            )

    # write output to CDB
    session = pjapi.Loadset(loadsetname="Test.Py.HEFTcom24", curvetype=1)

    session.add(output)
    session.send()
