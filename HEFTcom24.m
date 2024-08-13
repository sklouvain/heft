function [status, description] = HEFTcom24(ModelRun, RunDateTime)

CallerGUID = '';

if strcmp(ModelRun, 'WindSolarQuantilesGBR')    
    ForecastSession = CreateOutputSessions('HEFTcom24', ModelRun, 'Forecast30min', RunDateTime, 2, CallerGUID);
    session = ForecastSession{1};
    forecast_date_start = RunDateTime;
    forecast_date_end = RunDateTime;    
elseif ismember(ModelRun, {'EvaluateCDB', 'EvaluateML'})
    ActualSession = CreateOutputSessions('Test', 'Py', 'HEFTcom24', RunDateTime, 1, CallerGUID);
    session = ActualSession{1};
    forecast_date_start = RunDateTime - 8;
    forecast_date_end = forecast_date_start;
    value_date_start = forecast_date_start + 44/48;
    value_date_end = forecast_date_start + 1 + 43/48;
    switch ModelRun
        case 'EvaluateCDB'
            tagRev = 'Pwr.GB;A;PC;PRO;Energy;UTC;Revenue;D.1;D.1;';
            tagPL = 'Pwr.GB;A;PC;PRO;Energy;UTC;PinballLoss;D.1;D.1;';
            % tagPL = 'Pwr.GB;A;PC;PRO;Energy;EC18;UTC;PinballLoss;D.1;D.1;';
            % tagPL = 'Pwr.GB;A;PC;PRO;Energy;UTC;quants;PinballLoss;D.1;D.1;';
            % quantiles =  {'q10';    'q20';   'q30';    'q40';    'q50';    'q60';     q70';    'q80';     'q90'}; 
            quantiles = [157171539 157171538 157171547 157171541 157171546 157171542 157171543 157171540 157171544];
            % quantiles = [157530374 157530372 157530379 157530377 157530373 157530378 157530380 157530375 157530381];
            q50gen_SSPdeterminitsicRF = 157345337;
            q40gen_SSPstochastic = 157345338;
            q60gen_SSPstochastic = 157345340;
            q50gen_SSPstochastic = 157345341;
            optimalRF = 157621367; %157669137
        case 'EvaluateML'
            tagRev = 'Pwr.GB;A;PC;PRO;Energy;ECOp;v1.1;UTC;Revenue;D.1;D.1;';
            tagPL = 'Pwr.GB;A;PC;PRO;Energy;ECOp;v1.1;UTC;PinballLoss;D.1;D.1;';
            %  quantiles =  {'q10';    'q20';   'q30';    'q40';    'q50';    'q60';     q70';    'q80';     'q90'}; 
            %  quantiles = [157321007 157321004 157321008 157321002 157321005 157321009 157321010 157321006 157321003];
            quantiles = [157379591 157379587 157379585 157379589 157379586 157379592 157379588 157379584 157379590];	 
            q50gen_SSPdeterminitsicRF = 157379596; % 157358067;
            q40gen_SSPstochastic = 157379593; % 157358071;
            q60gen_SSPstochastic = 157379597; % 157358070;
            q50gen_SSPstochastic = 157379594; % 157358068;
            optimalRF = 158805426; %157621367; %157669137
    end
else
    return
end

if strcmp(ModelRun, 'WindSolarQuantilesGBR')
    
    % CurveIds for GBRPES10 (solarIds) and Hornsea One (windIds)
    % paths =  {'ECeR00';  'ECeR01';   'ECeR02';   'ECeR03';   'ECeR04';   'ECeR05';   'ECeR06';   'ECeR07';   'ECeR08';   'ECeR09';   'ECeR10';   'ECeR11';   'ECeR12';   'ECeR13';   'ECeR14';   'ECeR15';   'ECeR16';   'ECeR17';   'ECeR18';   'ECeR19';   'ECeR20';   'ECeR21';   'ECeR22';   'ECeR23';   'ECeR24';   'ECeR25';   'ECeR26';   'ECeR27';   'ECeR28';   'ECeR29';   'ECeR30';   'ECeR31';   'ECeR32';   'ECeR33';   'ECeR34';   'ECeR35';   'ECeR36';   'ECeR37';   'ECeR38';   'ECeR39';   'ECeR40';   'ECeR41';   'ECeR42';   'ECeR43';   'ECeR44';   'ECeR45';   'ECeR46';   'ECeR47';   'ECeR48';   'ECeR49';   'ECeR50';};
    % solarIds  =  [156897930	156897966	156897907	156897984	156897899	156897967	156897935	156897972	156897943	156897933	156897961	156897908	156897985	156897948	156897959	156897949	156897896	156897970	156897905	156897942	156897986	156897952	156897897	156897950	156897956	156897947	156897937	156897957	156897944	156897955	156897960	156897906	156897898	156897962	156897946	156897931	156897968	156897938	156897958	156897976	156897977	156897939	156897953	156897978	156897979	156897987	156897973	156897980	156897936	156897934	156897903];
    solarIds  =  [157358623 157358659   157358620   157358677   157358612   157358660   157358628   157358665   157358636   157358626   157358654   157358621   157358678   157358641   157358652   157358642   157358609   157358663   157358618   157358635   157358679   157358645   157358610   157358643   157358649   157358640   157358630   157358650   157358637   157358648   157358653   157358619   157358611   157358655   157358639   157358624   157358661   157358631   157358651   157358669   157358670   157358632   157358646   157358671   157358672   157358680   157358666   157358673   157358629   157358627   157358616];
    windIds   =  [156888194	156888178	156888166	156888202	156888167	156888195	156888204	156888205	156888196	156888158	156888174	156888216	156888185	156888191	156888192	156888179	156888175	156888151	156888162	156888186	156888213	156888172	156888222	156888211	156888219	156888182	156888212	156888163	156888171	156888173	156888217	156888206	156888180	156888207	156888164	156888208	156888200	156888187	156888220	156888203	156888176	156888152	156888168	156888218	156888169	156888188	156888214	156888181	156888165	156888159	156888201];
    % solarqsIds = [156897895	156897981	156897974	156897963	156897929	156897982	156897954	156897902	156897969];
    solarqsIds = [157358608	157358674	157358667	157358656	157358622	157358675	157358647	157358615	157358662];
    windqsIds  = [156888221	156888223	156888157	156888189	156888154	156888209	156888193	156888198	156888160];
    p = [10 20 30 40 50 60 70 80 90]; % quantiles
    
    tag = 'Pwr.GB;MW;F;PC;PRO;Energy;UTC;min.30;H.6;';
    tagqs = 'Pwr.GB;MW;F;PC;PRO;Energy;UTC;quants;min.30;H.6;';

    %  Convert data to 30 min resolution and calculate mean, quantiles
    windFcst = tSeries(windIds, forecast_date_start, forecast_date_end, nan, nan);
    solarFcst = tSeries(solarIds, forecast_date_start, forecast_date_end, nan, nan);
    windqsFcst = tSeries(windqsIds, forecast_date_start, forecast_date_end, nan, nan);
    solarqsFcst = tSeries(solarqsIds, forecast_date_start, forecast_date_end, nan, nan);
    [~,IA,IB] = intersect(windFcst.dates*24, solarFcst.dates*24);
    windFcst = tSeries(windFcst.dates(IA), windFcst.values(IA, :));
    solarFcst = tSeries(solarFcst.dates(IB), solarFcst.values(IB, :)/2);
    windqsFcst = tSeries(windqsFcst.dates(IA), windqsFcst.values(IA, :));
    solarqsFcst = tSeries(solarqsFcst.dates(IB), solarqsFcst.values(IB, :)/2);
    output.datesH = solarFcst.dates;
    totalFcst = windFcst + solarFcst;
    totalqsFcst = windqsFcst + solarqsFcst;
    output.dates30m = output.datesH(1):1/48:(output.datesH(end) + 1/48);
    output.values30m = pchip(output.datesH, totalFcst.values', output.dates30m);
    output.valuesqs30m = pchip(output.datesH, totalqsFcst.values', output.dates30m);
    output.dates30m = output.dates30m';
    output.values30m = output.values30m';
    output.valuesqs30m = output.valuesqs30m';
    output.avg30m = mean(output.values30m, 2);
    output.avgqs30m = mean(output.valuesqs30m, 2);
    output.quant30m = prctile(output.values30m, p, 2);
    output.avg30m = round(output.avg30m, 2);
    output.valuesqs30m = round(output.valuesqs30m, 2);
    output.avgqs30m = round(output.avgqs30m, 2);
    output.quant30m = round(output.quant30m, 2);

    %  Prepare output
    matAll=[output.dates30m output.avg30m output.quant30m output.avgqs30m output.valuesqs30m];
    output.tags = strcat(';', tag,';',{'AvgAcross;', 'Percentile.10;', 'Percentile.20;', 'Percentile.30;', 'Percentile.40;', 'Percentile.50;', 'Percentile.60;', 'Percentile.70;', 'Percentile.80;', 'Percentile.90;'});
    output.tags(end+1:end+10) = strcat(';', tagqs,';',{'AvgAcross;', 'Percentile.10;', 'Percentile.20;', 'Percentile.30;', 'Percentile.40;', 'Percentile.50;', 'Percentile.60;', 'Percentile.70;', 'Percentile.80;', 'Percentile.90;'});
    %matAll=[output.dates30m output.avgqs30m output.valuesqs30m];
    %output.tags = strcat(';', tagqs,';',{'AvgAcross;', 'Percentile.10;', 'Percentile.20;', 'Percentile.30;', 'Percentile.40;', 'Percentile.50;', 'Percentile.60;', 'Percentile.70;', 'Percentile.80;', 'Percentile.90;'});
    
    %  Send data to DWH
    [status, description] = writeModelOutput(session, output, matAll);
    
end

if ismember(ModelRun, {'EvaluateCDB', 'EvaluateML'})
    
    %  Definitions
    %  actuals =  {'wind';  'solar'}; 
    actuals = [156484292 156484285];
    dayahead = 156998806;
    imbalance = 103830604; %156998807;
    
    %  Read data from DWH
    quantilesFcst = tSeries(quantiles, forecast_date_start, forecast_date_end, value_date_start, value_date_end);    
    % bidsFcst = tSeries([q50gen_SSPdeterminitsicRF q40gen_SSPstochastic q60gen_SSPstochastic q50gen_SSPstochastic optimalRF], forecast_date_start, forecast_date_end, value_date_start, value_date_end);
    %bidsFcst = tSeries([optimalRF optimalRF optimalRF optimalRF optimalRF], forecast_date_start, forecast_date_end, value_date_start, value_date_end); 
    prices = tSeries([dayahead imbalance], value_date_start, value_date_end, nan, nan);
    actualsTS = tSeries(actuals, value_date_start, value_date_end, nan, nan);
    actualsTSsum = actualsTS.sum();
    
    %  Calculate Revenue
    revenueq50 = revenue(actualsTSsum.values, quantilesFcst.values(:, 5), prices.values(:, 1), prices.values(:, 2));
    revenueq60 = revenue(actualsTSsum.values, quantilesFcst.values(:, 6), prices.values(:, 1), prices.values(:, 2));
    revenueq40 = revenue(actualsTSsum.values, quantilesFcst.values(:, 4), prices.values(:, 1), prices.values(:, 2));
    %revenueq50gen_SSPdeterminitsicRF = revenue(actualsTSsum.values, bidsFcst.values(:, 1), prices.values(:, 1), prices.values(:, 2));
    %revenueq40gen_SSPstochastic = revenue(actualsTSsum.values, bidsFcst.values(:, 2), prices.values(:, 1), prices.values(:, 2));
    %revenueq60gen_SSPstochastic = revenue(actualsTSsum.values, bidsFcst.values(:, 3), prices.values(:, 1), prices.values(:, 2));
    %revenueq50gen_SSPstochastic = revenue(actualsTSsum.values, bidsFcst.values(:, 4), prices.values(:, 1), prices.values(:, 2));
    %revenueoptimalRF = revenue(actualsTSsum.values, bidsFcst.values(:, 5), prices.values(:, 1), prices.values(:, 2));
    
    %  Calculate Pinball Loss
    p = [10 20 30 40 50 60 70 80 90]; % quantiles
    score = [];
    for i = 1:length(p)
        out = pinball(actualsTSsum.values, quantilesFcst.values(:, i), p(i)/100);
        score = [score out]; %#ok<AGROW>
    end
    pinballscore = sum(score)/length(score);
        
    %  Prepare output
    % matAll = [(forecast_date_start + 1) revenueq50 revenueq60 revenueq40 revenueq50gen_SSPdeterminitsicRF revenueq40gen_SSPstochastic revenueq60gen_SSPstochastic revenueq50gen_SSPstochastic revenueoptimalRF pinballscore]; % market (trading) day = forecast_date_start + 1
    matAll = [(forecast_date_start + 1) revenueq50 revenueq60 revenueq40 pinballscore]; % market (trading) day = forecast_date_start + 1
    % matAll = [(forecast_date_start + 1) revenueoptimalRF]; 
    % matAll = [(forecast_date_start + 1) pinballscore]; % market (trading) day = forecast_date_start + 1

    % output.tags = strcat(';', tagRev,';',{'Percentile.50;', 'Percentile.60;', 'Percentile.40;', 'q50gen_SSPdeterminitsicRF;', 'q40gen_SSPstochastic;', 'q60gen_SSPstochastic;', 'q50gen_SSPstochastic;', 'optimalRF;'});
    output.tags = strcat(';', tagRev,';',{'Percentile.50;', 'Percentile.60;', 'Percentile.40;'});
    % output.tags = strcat(';', tagRev,';',{'optimalRF;calibApr2024;'});
    output.tags(end+1) = strcat(';', tagPL,';',{'AvgAcross;'});
    % output.tags = strcat(';', tagPL,';',{'AvgAcross;'});
    
    %  Send data to DWH
    [status, description] = writeModelOutput(session, output, matAll);
    
end
     
end

function out = pinball(y, q, alpha)
    result = (y - q) .* alpha .* (y - q >= 0) + ...
             (q - y) .* (1 - alpha) .* (y - q < 0);
    out = sum(result)/length(result);
end

function out = revenue(y, q, da, ssp)
    result = q .* da + (y - q) .* (ssp - 0.07 .* (y - q));
    out = sum(result);
end

function [status, description] = writeModelOutput(session, output, matAll)
    if ~isempty(matAll)
        %  Create OutputToSession
        [status, description] = OutputToSession(session, {['Dates' , output.tags], matAll});
        %  Send to DWH
        session.send();
    else
        status = -1;
        description = 'No data to send to DWH';
    end
end