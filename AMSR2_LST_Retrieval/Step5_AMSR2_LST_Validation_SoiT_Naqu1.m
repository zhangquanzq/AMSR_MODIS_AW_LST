%% 使用那曲地区站点观测的温度数据验证AMSR2地表温度.
% 此程序将同一站点的昼夜数据分开做散点图.

%% 功能标记和预设参数.
% 指定白天和晚上的标记. 1表示白天(升轨), 2表示晚上(降轨).
flg2 = 2;

% 昼夜, 过境时间标记.
daynight = {'Day', 'Night'};
daynight = daynight{flg2};

transit = {'1330', '0130'};
transit = transit{flg2};

%% 路径.
% 根目录.
rootDir = 'I:\AMSR2_MODIS_AW_LST';
addpath(fullfile(rootDir, 'Code/Functions/'))
retrievalDir = fullfile(rootDir, 'AMSR2_LST_Retrieval');
dataPath = fullfile(retrievalDir, 'Data');
figPath = fullfile(retrievalDir, 'Figures');

% 输入数据路径.
featureDir = fullfile(dataPath, 'Feature');
siteTrDir = fullfile(dataPath, 'SiteData\Naqu\Large_network');
amsr2LstDir = fullfile(dataPath, 'AMSR2_4_LSTCN_TIF');
modisUpscaleDir = fullfile(dataPath, 'MYD11A1_3_UpscalingCn_TIF');

% 输出数据路径.
siteMatDir = fullfile(dataPath, 'SiteSoilT_Matlab');
if ~exist(siteMatDir, 'dir')
    mkdir(siteMatDir)
end
siteFigureDir = fullfile(figPath, 'SiteSoilT_DaynightSeparate');
if ~exist(siteFigureDir, 'dir')
    mkdir(siteFigureDir)
end

%% 整理站点片区的土壤温度观测数据, 并将其保存为Mat文件.
% 获取站点属性信息.
siteStruct = shaperead(fullfile(featureDir, 'Sites_Location_Naqu.shp'));
siteLocationList = [[siteStruct.X]; [siteStruct.Y]]';
siteNameList = {siteStruct.Name}';
siteNameN = length(siteNameList);

% 获取片区站点文件名称列表.
siteTrTxtList = {dir(fullfile(siteTrDir, '*.txt')).name}';
if length(siteTrTxtList) ~= siteNameN
    error('那曲地区站点的位置矢量文件与观测数据文件的站点数量不匹配, 请检查!')
end

siteMatPath = fullfile(siteMatDir, sprintf('SiteSoilT_Naqu_%s.mat', daynight));
if ~exist(siteMatPath, 'file')
    % 创建存储数据的表, 表包括4个字段, 分别是: 站点名, 站点位置, 时间列表, 温度列表.
    siteDataCell = cell(siteNameN, 4);
    for i = 1: siteNameN
        siteTrTxtPath = fullfile(siteTrDir, siteTrTxtList{i});

        % 读取记录站点观测数据文档的文件头, 获取日期, 时间, 土壤温度属性信息.
        soilTCol = ''; [dateCol, timeCol, headLinesN, fieldN] = deal(0);
        fid = fopen(siteTrTxtPath);
        while true
            txtLine = fgetl(fid);
            headLinesN = headLinesN + 1;
            if contains(txtLine, 'Date-stamp')
                txtLine = split(txtLine, ';');
                dateCol = str2double(txtLine{1}(2:end));
            elseif contains(txtLine, 'Time-stamp')  % UTC
                txtLine = split(txtLine, ';');
                timeCol = str2double(txtLine{1}(2:end));
            elseif contains(txtLine, 'degree C;0.00;0.05')
                txtLine = split(txtLine, ';');
                soilTCol = append(soilTCol, [txtLine{1}(2:end), ' ']);
            elseif contains(txtLine, '$')
                fieldN = headLinesN - 1;
                break
            end
        end
        fclose(fid);
        soilTCol = str2double(split(deblank(soilTCol), ' '))';
        if ismember(0, [dateCol timeCol soilTCol])
            error('数据不完整, 请检查: %s', siteTrTxtPath)
        end

        % 读取站点的观测时刻与温度记录, 将站点记录使用的UTC时间转为当地时(北京时间).
        siteSoilTTable = readtable(siteTrTxtPath, NumHeaderLines=headLinesN);
        siteDateRecords = string(table2array(siteSoilTTable(:, dateCol)));
        siteTimeRecords = string(table2array(siteSoilTTable(:, timeCol)));
        siteDatetimeRecords = datetime(append(siteDateRecords, ' ', siteTimeRecords)) + hours(8);
        siteSoilTRecords = table2array(siteSoilTTable(:, soilTCol));
        siteSoilTRecords(siteSoilTRecords==-99) = nan;
        siteSoilTRecords = mean(siteSoilTRecords, 2, 'omitnan');
        nanIndex = isnan(siteSoilTRecords);
        siteDatetimeRecords(nanIndex) = [];
        siteSoilTRecords(nanIndex) = [];


        % 获取与站点观测日期有重叠的MODIS过境时刻列表.
        modisDatetimeList = strcat(unique(string(siteDatetimeRecords, 'yyyyMMdd')), transit);
        modisDatetimeList = datetime(modisDatetimeList, InputFormat='yyyyMMddHHmm');

        % 获取与MODIS过境时刻最近的站点观测时间和温度列表.
        modisDatetimeN = length(modisDatetimeList);
        siteDatetimeList = strings(modisDatetimeN, 1);
        siteSoilTList = zeros(modisDatetimeN, 1) * nan;
        siteModisDurationIndex = true(modisDatetimeN, 1);
        for k = 1: modisDatetimeN
            datetimeDiff = abs(siteDatetimeRecords - modisDatetimeList(k));
            if min(datetimeDiff) > minutes(15)
                siteModisDurationIndex(k) = false;
                continue
            end
            datetimeDiffMinIndex = find(datetimeDiff == min(datetimeDiff), 1);
            siteDatetimeList(k) = siteDatetimeRecords(datetimeDiffMinIndex);
            siteSoilTList(k) = siteSoilTRecords(datetimeDiffMinIndex);
        end
        modisDatetimeList = modisDatetimeList(siteModisDurationIndex);
        siteDatetimeList = siteDatetimeList(siteModisDurationIndex);
        siteDatetimeList = datetime(siteDatetimeList, InputFormat='yyyy-MM-dd HH:mm:ss');
        siteSoilTList = siteSoilTList(siteModisDurationIndex);

        % 插值MODIS过境时刻的站点温度.
        siteModisSoilTList = interp1(siteDatetimeRecords, siteSoilTRecords, modisDatetimeList, ...
            'spline');

        % 整理站点观测数据.
        siteDataCell{i, 1} = siteNameList{i};
        siteDataCell{i, 2} = siteLocationList(i, :);
        siteDataCell{i, 3} = modisDatetimeList;
        siteDataCell{i, 4} = siteModisSoilTList;
    end

    % 输出站点观测记录到Mat文件.
    tableVarStr = sprintf('site%sDataTable', daynight);
    siteDataTable = cell2table(siteDataCell, ...
        VariableNames=["SiteName" "Location" "Datetime" "SoilTemperature"]);
    assignin('base', tableVarStr, siteDataTable)
    save(siteMatPath, tableVarStr)
end

%% 校正站点观测数据, 并验证反演的AMSR2温度.
% 获取AMSR2 LST影像每个像元的经纬度坐标矩阵, 升尺度后的MODIS LST与AMSR2 LST的信息一样.
amsr2LstPath = fullfile(amsr2LstDir, 'AMSR2_LST_2012XXXX_TIF', 'AMSR2_LST_Day_20120703.tif');
amsr2Ref = geotiffinfo(amsr2LstPath).SpatialRef;
lonMin = amsr2Ref.LongitudeLimits(1);
lonMax = amsr2Ref.LongitudeLimits(2);
latMin = amsr2Ref.LatitudeLimits(1);
latMax = amsr2Ref.LatitudeLimits(2);
cellsizeX = amsr2Ref.CellExtentInLongitude;
cellsizeY = amsr2Ref.CellExtentInLatitude;
lonVector = lonMin + cellsizeX/2: cellsizeX: lonMax - cellsizeX/2;
latVector = latMax - cellsizeY/2: -cellsizeY: latMin + cellsizeY/2;

% 获取站点片区内各站点的名称, 位置, 时间, 温度数据列表.
siteVarStr = sprintf('site%sDataTable', daynight);
load(siteMatPath, siteVarStr); siteDataTable = eval(siteVarStr);
siteNameList = siteDataTable.SiteName;
siteLocationList = siteDataTable.Location;
siteDatetimeRecords = siteDataTable.Datetime;
siteSoilTRecords = siteDataTable.SoilTemperature;
siteNameN = length(siteNameList);

% 获取站点位置在AMSR2 LST数据上的行列号.
lstRowColList = zeros(siteNameN, 2) * nan;
for i = 1: siteNameN
    siteLocation = siteLocationList(i, :);
    lonDiffVector = abs(lonVector - siteLocation(1));
    latDiffVector = abs(latVector - siteLocation(2));
    lstCol = find(lonDiffVector == min(lonDiffVector), 1);
    lstRow = find(latDiffVector == min(latDiffVector), 1);
    lstRowColList(i, :) = [lstRow, lstCol];
end

% 按AMSR2像元分组平均站点观测, 并验证AMSR2 LST.
lstRowColTypes = unique(lstRowColList, 'rows');
for i = 1: length(lstRowColTypes)
    lstRowCol = lstRowColTypes(i, :);
    lstRowColIndexList = find(sum(lstRowColList == lstRowCol, 2) == 2);
    lstRowColIndexN = length(lstRowColIndexList);
    
    % 获取落入同一AMSR2 LST像元的各站点观测日期的合集.
    siteAllDatetimeList = siteDatetimeRecords{lstRowColIndexList(1)};
    for j = 2: lstRowColIndexN
        siteDatetimeList = siteDatetimeRecords{lstRowColIndexList(j)};
        siteAllDatetimeList = union(siteAllDatetimeList, siteDatetimeList);
    end

    % 获取落入同一AMSR2 LST像元的各站点观测日期的土壤温度均值.
    siteAllDatetimeN = length(siteAllDatetimeList);
    siteAllSoilTMeanList = zeros(siteAllDatetimeN, 1);
    for j = 1: siteAllDatetimeN
        siteDatetime = siteAllDatetimeList(j);
        siteSoilTVector = zeros(lstRowColIndexN, 1) * nan;
        for k = 1: lstRowColIndexN
            siteDatetimeList = siteDatetimeRecords{lstRowColIndexList(k)};
            siteDatetimeIndex = find(siteDatetimeList == siteDatetime, 1);
            if length(siteDatetimeIndex) == 1
                siteSoilTList = siteSoilTRecords{lstRowColIndexList(k)};
                siteSoilTVector(k) = siteSoilTList(siteDatetimeIndex);
            end
        end
        siteAllSoilTMeanList(j) = mean(siteSoilTVector, 'omitnan');
    end

    % 分年度验证AMSR2 LST.
    siteYearList = siteAllDatetimeList.Year;
    siteYearTypes = unique(siteYearList);
    for j = 1: length(siteYearTypes)
        siteYear = siteYearTypes(j);

        % 判断是否有站点观测年份的AMSR2 LST.
        amsr2LstYearDir = fullfile(amsr2LstDir, sprintf('AMSR2_LST_%dXXXX_TIF', siteYear));
        if ~exist(amsr2LstYearDir, 'dir')
            continue
        end

        % 获取校正站点温度的系数 ---------------------------------------------------------------------
        % 获取当前年份站点观测日期列表.
        siteYearIndex = (siteYearList == siteYear);
        siteDateInYearList = string(siteAllDatetimeList(siteYearIndex), 'yyyyMMdd');

        % 获取当前年份升尺度后MODIS LST的日期列表.
        modisLstYearDir = fullfile(modisUpscaleDir, sprintf('MYD11A1_%dXXX_TIF', siteYear));
        modisLstNameList = dir(fullfile(modisLstYearDir, sprintf('MYD11A1*_%s.tif', daynight)));
        modisLstNameList = {modisLstNameList.name}';
        modisDateList = cellfun(@(x) x(10:16), modisLstNameList, UniformOutput=false);
        modisDateList = string(cellfun(@yday2ymd, modisDateList, UniformOutput=false));

        % 获取升尺度后MODIS LST和站点观测共同日期列表.
        [~, modisDateIndex, siteDateIndex] = intersect(modisDateList, siteDateInYearList);

        % 获取共同日期的站点温度列表.
        siteSoilTInYearList = siteAllSoilTMeanList(siteYearIndex);
        siteSoilTInYearList = siteSoilTInYearList(siteDateIndex);
        siteSoilTInYearList = siteSoilTInYearList + 273.15;

        % 获取共同日期升尺度后MODIS LST影像中站点所在像元的温度列表.
        modisLstNameList = modisLstNameList(modisDateIndex);
        modisLstNameN = length(modisLstNameList);
        modisLstList = zeros(modisLstNameN, 1) * nan;
        for m = 1: modisLstNameN
            modisLstLayer = readgeoraster(fullfile(modisLstYearDir, modisLstNameList(m)));
            modisLstList(m) = modisLstLayer(lstRowCol(1), lstRowCol(2));
        end

        % 删除NaN值.
        nanIndex = isnan(siteSoilTInYearList) | isnan(modisLstList);
        siteSoilTInYearList(nanIndex) = [];
        modisLstList(nanIndex) = [];

        % 获取使用MODIS LST校正Soil_T的系数.
        p = polyfit(siteSoilTInYearList, modisLstList, 1);

        % 绘制修正站点温度的散点图.
        soilTCaliFigName = sprintf('SoilT_Calibration_Naqu%d_%d_%s.png', i, siteYear, daynight);
        soilTCaliFigPath = fullfile(siteFigureDir, soilTCaliFigName);
        if ~exist(soilTCaliFigPath, 'file')
            f1 = figure;
            % 散点图中要素属性设置.
            xLimit = [240 330]; scatterStyles = {'.r', '.b'}; siteNameColors = {'r', 'b'};

            % 误差统计指标, 及其在图表上的表示文本.
            siteLstInYearList = polyval(p, siteSoilTInYearList);
            r2 = corrcoef(siteSoilTInYearList, modisLstList); r2 = r2(1, 2);
            rmse1 = rmse(siteSoilTInYearList, modisLstList);
            rmse2 = rmse(siteLstInYearList, modisLstList);

            txt1 = sprintf('Naqu %d (%stime)', i, daynight);
            txt2 = sprintf('y = %.2fx %+.2f', p(1), p(2));
            txt3 = sprintf('R^2 = %.2f', r2);
            txt4 = sprintf('RMSE1 = %.2f K', rmse1);
            txt5 = sprintf('RMSE2 = %.2f K', rmse2);
            txt6 = sprintf('N = %d', length(modisLstList));
            txtList = {txt1, txt2, txt3, txt4, txt5, txt6};
            txtX = 0.05; txtY = 0.95; txtYInterval = 0.05;

            plot(siteSoilTInYearList, modisLstList, scatterStyles{flg2}, ...
                xLimit, polyval(p, xLimit), '--k', MarkerSize=10)
            for m = 1: length(txtList)
                t = text(txtX, txtY-txtYInterval*(m-1), txtList{m}, Units='normalized');
                t.FontWeight = 'bold'; if m == 1, t.Color = siteNameColors{flg2}; end
            end
            ax = gca; ax.FontWeight = 'bold'; ax.XLim = xLimit; ax.YLim = xLimit;
            xlabel('Soil Temperature (K)'); ylabel('Upscaled MODIS LST (K)');
            exportgraphics(f1, soilTCaliFigPath, Resolution=200);
        end

        % 校正SoilT, 并验证AMSR2 LST精度 ------------------------------------------------------------
        soilTValidateFigName = sprintf('SoilT_Validate_Naqu%d_%d_%s.png', i, siteYear, daynight);
        soilTValidateFigPath = fullfile(siteFigureDir, soilTValidateFigName);
        if ~exist(soilTValidateFigPath, 'file')
            % 获取当前年份AMSR2 LST的日期列表.
            amsr2LstName = sprintf('AMSR2_LST_%s*.tif', daynight);
            amsr2LstNameList = {dir(fullfile(amsr2LstYearDir, amsr2LstName)).name}';
            amsr2DateList = cellfun(@(x) x(end-11:end-4), amsr2LstNameList, UniformOutput=false);
            amsr2DateList = string(amsr2DateList);

            % 获取AMSR2 LST和站点观测共同日期列表.
            [~, amsr2DateIndex, siteDateIndex] = intersect(amsr2DateList, siteDateInYearList);

            % 获取共同日期的站点温度列表, 并校正.
            siteSoilTInYearList = siteAllSoilTMeanList(siteYearIndex);
            siteSoilTInYearList = siteSoilTInYearList(siteDateIndex) + 273.15;
            siteLstInYearList = polyval(p, siteSoilTInYearList);

            % 获取共同日期AMSR2 LST影像中站点所在像元的温度列表.
            amsr2LstNameList = amsr2LstNameList(amsr2DateIndex);
            amsr2LstNameN = length(amsr2LstNameList);
            amsr2LstList = zeros(amsr2LstNameN, 1) * nan;
            for m = 1: amsr2LstNameN
                amsr2LstLayer = readgeoraster(fullfile(amsr2LstYearDir, amsr2LstNameList(m)));
                amsr2LstList(m) = amsr2LstLayer(lstRowCol(1), lstRowCol(2));
            end

            % 删除NaN值.
            nanIndex = isnan(siteLstInYearList) | (amsr2LstList == 0);
            siteLstInYearList(nanIndex) = [];
            siteSoilTInYearList(nanIndex) = [];
            amsr2LstList(nanIndex) = [];

            % 获取图表参数, 并制图 ------------------------------------------------------------------
            f2 = figure; [xLimit, yLimit] = deal([240 330]);
            scatterStyles = {'.r', '.b'}; siteNameColors = {'r', 'b'};
            % 误差统计指标, 及其在图表上的表示文本.
            bias = mean(siteLstInYearList - amsr2LstList);
            rmse3 = rmse(siteLstInYearList, amsr2LstList);
            rmse4 = rmse(siteSoilTInYearList, amsr2LstList);

            txt1 = sprintf('Naqu%d (%stime)', i, daynight);
            txt2 = sprintf('Bias = %.2f K', bias);
            txt3 = sprintf('RMSE = %.2f K', rmse3);
            txt4 = sprintf('N = %d', length(amsr2LstList));
            txtList = {txt1, txt2, txt3, txt4};
            txtX = 0.05; txtY = 0.95; txtYInterval = 0.05;

            plot(siteLstInYearList, amsr2LstList, scatterStyles{flg2}, ...
                xLimit, yLimit, '--k', MarkerSize=10)
            ax = gca; ax.FontWeight = 'bold'; ax.XLim = xLimit; ax.YLim = xLimit;
            xlabel('Calibrated Soil Temperature (K)'); ylabel('AMSR2 LST (K)');
            for m = 1: length(txtList)
                t = text(txtX, txtY-txtYInterval*(m-1), txtList{m}, Units='normalized');
                t.FontWeight = 'bold'; if m == 1, t.Color = siteNameColors{flg2}; end
            end
            exportgraphics(f2, soilTValidateFigPath, Resolution=200);
        end
        close all
    end
end
