%% 创建AMSR2 LST反演模型.
% 此代码为创建温度反演模型的正常流程, 而Step3_AMSR2_LST_Model_Construction1.m 则用于专门对模型进行优化.

%% 功能标记与预设参数.
% 指定白天和晚上的标记. 1表示白天(升轨), 2表示晚上(降轨).
flg1 = 2;

% AMSR2数据的通道, 极化.
channelList = ["10", "18", "23", "36", "06", "07", "89"];
channelN = length(channelList);
polarizeN = length(["H", "V"]);

% 回归模型中的变量个数(14个通道 + 2个二次项 + 1常数项).
variablesN = channelN * polarizeN + 3;

% 数据年份列表(时间区间2012/07/02-2019/12/31).
yearList = 2012: 2019;

% 各季节[春, 夏, 秋, 冬]包含的月份, 和各月份的名称.
seasonMonthList = {[3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 1, 2]};
seasonNameList = {'Spring', 'Summer', 'Autumn', 'Winter'};
seasonN = length(seasonNameList);
monthList = {'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'};
monthN = length(monthList);

% 昼夜标记.
daynight = {'Day', 'Night'};
daynight = daynight{flg1};

% AMSR2 BT影像中的NoData.
amsr2Nodata = [65534, 65535];

%% 路径.
% 根目录.
rootDir = 'E:\AMSR2_MODIS_AW_LST';
addpath(fullfile(rootDir, 'Code\Functions'));
retrievalDir = fullfile(rootDir, 'AMSR2_LST_Retrieval');
dataDir = fullfile(retrievalDir, 'Data');

% 输入数据路径.
amsr2CnMatDir = fullfile(dataDir, 'AMSR2_2_CN_Matlab');
amsr2MaskMatDir = fullfile(dataDir, 'AMSR2_3_MaskCn_Matlab');
modisLstMaskMatDir = fullfile(dataDir, 'MYD11A1_3_MaskCn_Matlab');
ccsevMatDir = fullfile(dataDir, 'CCSEV_Matlab');

% 输出反演模型的路径.
regressMatDir = fullfile(dataDir, 'Regression_Matlab');
if ~exist(regressMatDir, 'dir')
    mkdir(regressMatDir)
end

% 输出统计图路径.
figDir = fullfile(retrievalDir, 'Figures');
if ~exist(figDir, 'dir')
    mkdir(figDir)
end
lstScatterDir = fullfile(figDir, 'LstScatter');
if ~exist(lstScatterDir, 'dir')
    mkdir(lstScatterDir)
end

%% 回归和输出.
% 分年度建立回归模型AMSR2 LST.
for i = 1: length(yearList)
%     yearStr = num2str(yearList(i));
    yearStr = '2020';

    % 存储纯像元分区回归系数与反演的AMSR2 LST影像数据的Mat文件路径.
    regressPureMatName = sprintf('Regression_Pure_%s_%s.mat', yearStr, daynight);
    regressPureMatPath = fullfile(regressMatDir, regressPureMatName);
    if  exist(regressPureMatPath, 'file')
        continue
    end

    % 创建输出AMSR2 BT(LST)和MODIS LST散点图的文件夹.
    lstScatterYearDir = fullfile(lstScatterDir, sprintf('%s %s', yearStr, daynight));
    if ~exist(lstScatterYearDir, 'dir')
        mkdir(lstScatterYearDir);
    end

    % 从Mat文件中读取中国区原始的AMSR2 BT数据.
    amsr2CnMatName = sprintf('AMSR2_BT_%s_%s.mat', yearStr, daynight);
    amsr2CnMatPath = fullfile(amsr2CnMatDir, amsr2CnMatName);
    load(amsr2CnMatPath, 'amsr2H*YearArray', 'amsr2V*YearArray');

    % 从Mat文件中读取Mask后的AMSR2 BT和MODIS LST数据.
    amsr2MaskMatName = sprintf('AMSR2_BT_MaskCn_%s_%s.mat', yearStr, daynight);
    amsr2MaskMatPath = fullfile(amsr2MaskMatDir, amsr2MaskMatName);
    load(amsr2MaskMatPath, 'amsr2H*MaskYearArray', 'amsr2V*MaskYearArray', 'lstDateFilterList');

    modisLstMaskMatName = sprintf('MYD11A1_MaskCn_%s_%s.mat', yearStr, daynight);
    modisLstMaskMatPath = fullfile(modisLstMaskMatDir, modisLstMaskMatName);
    load(modisLstMaskMatPath, 'amsr2Ref', 'modisLstMaskYearArray');

    % 从Mat文件中读取CCSEV.
    ccsevMatPath = fullfile(ccsevMatDir, sprintf('CCSEV_%s.mat', yearStr));
    load(ccsevMatPath, 'zonesLcCodeList', 'zonesLcFullCodeList', 'lcCodeList', 'lcDateList', ...
        'fixedZonesLcArray');

    % 获取AMSR2 BT影像的行列数.
    [amsr2RowN, amsr2ColN] = size(modisLstMaskYearArray, 1, 2);

    % 获取AMSR2 BT, MODIS LST以及分区数据的共有日期列表.
    [validDateList, lstDateIndex, lcDateIndex] = intersect(lstDateFilterList, lcDateList);
    validYearMonthList = datetime(validDateList , 'InputFormat', 'yyyyMMdd').Month;
    validDateN = length(validDateList);

    % 共有日期的分区矩阵和MODIS LST数据.
    fixedZonesLcArray = fixedZonesLcArray(:, :, lcDateIndex);
    modisLstMaskYearArray = modisLstMaskYearArray(:, :, lstDateIndex); % 数据类型: single

    % 共有日期的AMSR2 BT数据, 将不同通道的数据拆分到单独的矩阵, 并存入元胞数组中.
    % 通道的排序: [10H, 10V, 18H, 18V, 23H, 23V, 36H, 36V, 06H, 06V, 07H, 07V, 89H, 89V]
    [amsr2HMaskYearCell, amsr2VMaskYearCell] = deal(cell(1, channelN));
    for k = 1: channelN
        amsr2HChannelArray = eval(sprintf('amsr2H%sMaskYearArray', channelList{k}));
        amsr2VChannelArray = eval(sprintf('amsr2V%sMaskYearArray', channelList{k}));
        amsr2HMaskYearCell{k} = single(amsr2HChannelArray(:, :, lstDateIndex));
        amsr2VMaskYearCell{k} = single(amsr2VChannelArray(:, :, lstDateIndex));
    end
    clear amsr2HChannelArray amsr2VChannelArray amsr2H*MaskYearArray amsr2V*MaskYearArray

    % 纯像元分区编号列表.
    pureLcCodeList = zonesLcCodeList(zonesLcCodeList < lcCodeList(end));
    pureLcCodeN = length(pureLcCodeList);

    % 存储评价AMSR2 LST反演精度指标的矩阵.
    [rmseYearVector, nYearVector, r2YearVector] = deal(zeros(pureLcCodeN, 1) * nan);
    [rmseSeasonArray, nSeasonArray, r2SeasonArray] = deal(zeros(pureLcCodeN, seasonN) * nan);
    [rmseMonthArray, nMonthArray, r2MonthArray] = deal(zeros(pureLcCodeN, monthN) * nan);

    % 存储反演AMSR2 LST回归系数的矩阵.
    coefficientYearArray = zeros(pureLcCodeN, variablesN);
    coefficientSeasonArray = zeros(pureLcCodeN, variablesN, seasonN); % 4个季节.
    coefficientMonthArray = zeros(pureLcCodeN, variablesN, monthN);  % 12个月.

    % 存储分别使用年尺度, 季节尺度, 和月尺度反演后的AMSR2 LST影像数据的矩阵, 包括掩膜区和全中国区.
    [amsr2LstMaskYearArray1, amsr2LstCnYearArray1, amsr2LstMaskYearArray2, ...
        amsr2LstCnYearArray2, amsr2LstMaskYearArray3, amsr2LstCnYearArray3] = ...
        deal(zeros(amsr2RowN, amsr2ColN, validDateN, 'single'));

    % ==============================================================================================
    % 获取反演AMSR2 LST的系数, 精度, 以及掩膜区AMSR2 LST.
    for j = 1: pureLcCodeN
        zonesLcCode = pureLcCodeList(j);
        zoneName = sprintf('Zone %d', zonesLcCode);
        fprintf('分区%d, %s年.\n', zonesLcCode, yearStr);

        % 获取当前分区的位置索引.
        zonesLcIndexArray = (fixedZonesLcArray == zonesLcCode);

        % 从一整年的数组中提取当前分区的原始AMSR2 BT影像, 以及掩膜后的AMSR2 BT和MODIS LST影像.
        modisLstMaskZoneYearArray = setnan(modisLstMaskYearArray, ~zonesLcIndexArray);

        [amsr2HCnZoneYearCell, amsr2VCnZoneYearCell] = deal(cell(1, channelN));
        [amsr2HMaskZoneYearCell, amsr2VMaskZoneYearCell] = deal(cell(1, channelN));
        for k = 1 : channelN
            amsr2HCnYearArray = single(eval(sprintf('amsr2H%sYearArray', channelList{k})));
            amsr2VCnYearArray = single(eval(sprintf('amsr2V%sYearArray', channelList{k})));
            amsr2HCnYearArray(ismember(amsr2HCnYearArray, amsr2Nodata)) = nan;
            amsr2VCnYearArray(ismember(amsr2VCnYearArray, amsr2Nodata)) = nan;
            amsr2HCnYearArray = amsr2HCnYearArray(:, :, lstDateIndex);
            amsr2VCnYearArray = amsr2VCnYearArray(:, :, lstDateIndex);
            amsr2HCnZoneYearCell{k} = setnan(amsr2HCnYearArray, ~zonesLcIndexArray) / 100;
            amsr2VCnZoneYearCell{k} = setnan(amsr2VCnYearArray, ~zonesLcIndexArray) / 100;
            amsr2HMaskZoneYearCell{k} = setnan(amsr2HMaskYearCell{k}, ~zonesLcIndexArray) / 100;
            amsr2VMaskZoneYearCell{k} = setnan(amsr2VMaskYearCell{k}, ~zonesLcIndexArray) / 100;
        end
        amsr2QdCnZoneYearCell = {
            (amsr2VCnZoneYearCell{4} - amsr2VCnZoneYearCell{3}).^2, ...
            (amsr2VCnZoneYearCell{4} - amsr2VCnZoneYearCell{2}).^2};
        clear amsr2HCnYearArray amsr2VCnYearArray
        
        % ------------------------------------------------------------------------------------------
        % 按年尺度回归AMSR2 BT和MODIS LST, 并获取反演的AMSR2 LST影像.
        % 按照回归函数的格式要求, 将参与回归的影像矩阵重新排列.
        valueIndex = find(~isnan(modisLstMaskZoneYearArray));
        modisLstMaskZoneYearVector = double(modisLstMaskZoneYearArray(valueIndex));
        [amsr2HMaskZoneYearVector, amsr2VMaskZoneYearVector] = deal(cell(1, channelN));
        for k = 1: channelN
            amsr2HMaskZoneYearVector{k} = amsr2HMaskZoneYearCell{k}(valueIndex);
            amsr2VMaskZoneYearVector{k} = amsr2VMaskZoneYearCell{k}(valueIndex);
        end
        amsr2QdMaskZoneYearVector = [
            (amsr2VMaskZoneYearVector{4} - amsr2VMaskZoneYearVector{3}).^2, ...
            (amsr2VMaskZoneYearVector{4} - amsr2VMaskZoneYearVector{2}).^2];
        amsr2BtMaskZoneYearRecords = double(cell2mat([amsr2HMaskZoneYearVector, ...
            amsr2VMaskZoneYearVector, amsr2QdMaskZoneYearVector]));
        clear amsr2HMaskZoneYearVector amsr2VMaskZoneYearVector amsr2QdMaskZoneYearVector;

        % 样本数大于2时才能执行逐步回归, 否则该分区不能反演AMSR2 LST.
        % 系数矩阵 [10H, 18H, 23H, 36H, 06H, 07H, 89H, 10V, 18V, 23V, 36V, 06V, 07V, 89V, 常数].
        modisLstMaskZoneYearVectorN = length(modisLstMaskZoneYearVector);
        if modisLstMaskZoneYearVectorN >= 2
            timestamp = sprintf('%s %s', yearStr, daynight);
            fprintf('分区%d 年尺度 回归: %s.\n', zonesLcCode, timestamp)
            mdl = stepwiselm(amsr2BtMaskZoneYearRecords, modisLstMaskZoneYearVector, ...
                Lower='constant', Upper='linear', Criterion='aic');
            lstRMSE = mdl.RMSE;
            lstR2 = mdl.Rsquared.Ordinary;
            amsr2LstMaskZoneYearVector = mdl.Fitted;
            variableIndex = find(mdl.VariableInfo.InModel == 1);
            pYear = mdl.Coefficients.Estimate;
            pYear = pYear([2:end, 1]);
            coefficientYearArray(j, [variableIndex; end]) = pYear;

            amsr2LstMaskZoneYearVector2 = single(zeros(amsr2RowN * amsr2ColN * validDateN, 1));
            amsr2LstMaskZoneYearVector2(valueIndex) = amsr2LstMaskZoneYearVector;
            amsr2LstMaskZoneYearArray = reshape(amsr2LstMaskZoneYearVector2, ...
                amsr2RowN, amsr2ColN, validDateN);
            clear amsr2LstMaskZoneYearVector2

            % 根据回归系数计算分区内的AMSR2 LST(初始化矩阵为常数项截距).
            amsr2LstCnZoneYearArray = repmat(pYear(end), amsr2RowN, amsr2ColN, validDateN);
            for k = 1: channelN
                amsr2LstCnZoneYearArray = amsr2LstCnZoneYearArray + ...
                    amsr2HCnZoneYearCell{k} * coefficientYearArray(j, k) + ...
                    amsr2VCnZoneYearCell{k} * coefficientYearArray(j, k + channelN);
            end
            amsr2LstCnZoneYearArray = amsr2LstCnZoneYearArray + ...
                amsr2QdCnZoneYearCell{:, 1} * coefficientYearArray(j, k*2+1) + ...
                amsr2QdCnZoneYearCell{:, 2} * coefficientYearArray(j, k*2+2);
            % nan值改为0, 便于算完所有分区AMSR2 LST后将结果相加.
            amsr2LstCnZoneYearArray(isnan(amsr2LstCnZoneYearArray)) = 0;

            % 输出AMSR2 BT(LST)和MODIS LST的年度散点图.
            zoneScatterYearName = sprintf('LstScatter_%s_%s.tif', zoneName, timestamp);
            zoneScatterYearPath = fullfile(lstScatterYearDir, zoneScatterYearName);
            if  ~exist(zoneScatterYearPath, 'file')
                f = lstScatter(amsr2LstMaskZoneYearVector, modisLstMaskZoneYearVector, ...
                    timestamp, zoneName, [lstR2, lstRMSE]);
                exportgraphics(f, zoneScatterYearPath);
                close all;
            end
        else
            lstRMSE = nan; lstR2 = nan;
            amsr2LstMaskZoneYearVector = zeros(modisLstMaskZoneYearVectorN, 1) * nan;
            amsr2LstMaskZoneYearArray = zeros(amsr2RowN, amsr2ColN, validDateN);
            amsr2LstCnZoneYearArray = zeros(amsr2RowN, amsr2ColN, validDateN);
        end
        rmseYearVector(j) = lstRMSE;
        nYearVector(j) = modisLstMaskZoneYearVectorN;
        r2YearVector(j) = lstR2;

        % 将反演的当年AMSR2 LST保存到年度矩阵中.
        amsr2LstMaskYearArray1 = amsr2LstMaskYearArray1 + amsr2LstMaskZoneYearArray;
        amsr2LstCnYearArray1 = amsr2LstCnYearArray1 + amsr2LstCnZoneYearArray;

        % ------------------------------------------------------------------------------------------
        % 按季节尺度回归AMSR2 BT和MODIS LST, 并获取反演的AMSR2 LST影像.
        for n = 1: seasonN
            seasonIndex = ismember(validYearMonthList, seasonMonthList{n});
            seasonIndexN = sum(seasonIndex);

            % 筛选当前季节的数据矩阵.
            modisLstMaskZoneSeasonArray = modisLstMaskZoneYearArray(:, :, seasonIndex);
            [amsr2HMaskZoneSeasonCell, amsr2VMaskZoneSeasonCell] = deal(cell(1, channelN));
            [amsr2HCnZoneSeasonCell, amsr2VCnZoneSeasonCell] = deal(cell(1, channelN));
            for k = 1: channelN
                amsr2HMaskZoneSeasonCell{k} = amsr2HMaskZoneYearCell{k}(:, :, seasonIndex);
                amsr2VMaskZoneSeasonCell{k} = amsr2VMaskZoneYearCell{k}(:, :, seasonIndex);
                amsr2HCnZoneSeasonCell{k} = amsr2HCnZoneYearCell{k}(:, :, seasonIndex);
                amsr2VCnZoneSeasonCell{k} = amsr2VCnZoneYearCell{k}(:, :, seasonIndex);
            end
            amsr2QdCnZoneSeasonCell = {
                (amsr2VCnZoneSeasonCell{4} - amsr2VCnZoneSeasonCell{3}).^2, ...
                (amsr2VCnZoneSeasonCell{4} - amsr2VCnZoneSeasonCell{2}).^2};

            % 按照回归函数的格式要求, 将参与回归的影像矩阵重新排列.
            valueIndex = find(~isnan(modisLstMaskZoneSeasonArray));
            modisLstMaskZoneSeasonVector = double(modisLstMaskZoneSeasonArray(valueIndex));
            [amsr2HMaskZoneSeasonVector, amsr2VMaskZoneSeasonVector] = deal(cell(1, channelN));
            for k = 1: channelN
                amsr2HMaskZoneSeasonVector{k} = amsr2HMaskZoneSeasonCell{k}(valueIndex);
                amsr2VMaskZoneSeasonVector{k} = amsr2VMaskZoneSeasonCell{k}(valueIndex);
            end
            amsr2QdMaskZoneSeasonVector = [
                (amsr2VMaskZoneSeasonVector{4} - amsr2VMaskZoneSeasonVector{3}).^2, ...
                (amsr2VMaskZoneSeasonVector{4} - amsr2VMaskZoneSeasonVector{2}).^2];
            amsr2BtMaskZoneSeasonRecords = double(cell2mat([amsr2HMaskZoneSeasonVector, ...
                amsr2VMaskZoneSeasonVector, amsr2QdMaskZoneSeasonVector]));
            clear amsr2HMaskZoneSeasonCell amsr2VMaskZoneSeasonCell modisLstMaskZoneSeasonArray
            clear amsr2HMaskZoneSeasonVector amsr2VMaskZoneSeasonVector amsr2QdMaskZoneSeasonVector

            % 样本数大于2时才能执行逐步回归, 否则该分区不能反演AMSR2 LST.
            % 系数 [10H, 18H, 23H, 36H, 06H, 07H, 89H, 10V, 18V, 23V, 36V, 06V, 07V, 89V, 常数].
            modisLstMaskZoneSeasonVectorN = length(modisLstMaskZoneSeasonVector);
            if modisLstMaskZoneSeasonVectorN >= 2
                timestamp = sprintf('%s %s %s', yearStr, daynight, seasonNameList{n});
                fprintf('分区%d 季节尺度 回归: %s.\n', zonesLcCode, timestamp)
                mdl = stepwiselm(amsr2BtMaskZoneSeasonRecords, modisLstMaskZoneSeasonVector,...
                    Lower='constant', Upper='linear', Criterion='aic');
                lstRMSE = mdl.RMSE;
                lstR2 = mdl.Rsquared.Ordinary;
                amsr2LstMaskZoneSeasonVector = mdl.Fitted;
                variableIndex = find(mdl.VariableInfo.InModel == 1);
                pSeason = mdl.Coefficients.Estimate;
                pSeason = pSeason([2:end, 1]);
                coefficientSeasonArray(j, [variableIndex; end], n) = pSeason;

                amsr2LstMaskZoneSeasonVector2 = zeros(amsr2RowN * amsr2ColN * seasonIndexN, 1);
                amsr2LstMaskZoneSeasonVector2(valueIndex) = amsr2LstMaskZoneSeasonVector;
                amsr2LstMaskZoneSeasonArray = reshape(amsr2LstMaskZoneSeasonVector2, ...
                    amsr2RowN, amsr2ColN, seasonIndexN);
                clear amsr2LstMaskZoneSeasonVector2

                % 根据回归系数计算分区内的AMSR2 LST(初始化矩阵为常数项截距).
                amsr2LstCnZoneSeasonArray = repmat(pSeason(end),amsr2RowN,amsr2ColN,seasonIndexN);
                for k = 1: channelN
                    amsr2LstCnZoneSeasonArray = amsr2LstCnZoneSeasonArray + ...
                        amsr2HCnZoneSeasonCell{k} * coefficientSeasonArray(j, k, n) + ...
                        amsr2VCnZoneSeasonCell{k} * coefficientSeasonArray(j, k + channelN, n);
                end
                amsr2LstCnZoneSeasonArray = amsr2LstCnZoneSeasonArray + ...
                    amsr2QdCnZoneSeasonCell{:, 1} * coefficientSeasonArray(j, k*2 + 1, n) + ...
                    amsr2QdCnZoneSeasonCell{:, 2} * coefficientSeasonArray(j, k*2 + 2, n);
                % nan值改为0, 便于算完所有分区AMSR2 LST后将结果相加.
                amsr2LstCnZoneSeasonArray(isnan(amsr2LstCnZoneSeasonArray)) = 0;
                clear amsr2HCnZoneSeasonCell amsr2VCnZoneSeasonCell amsr2QdCnZoneSeasonCell

                % 输出AMSR2 BT(LST)和MODIS LST的季节散点图.
                zoneScatterSeasonName = sprintf('LstScatter_%s_%s.tif', zoneName, timestamp);
                zoneScatterSeasonPath = fullfile(lstScatterYearDir, zoneScatterSeasonName);
                if ~exist(zoneScatterSeasonPath, 'file')
                    f = lstScatter(amsr2LstMaskZoneSeasonVector, modisLstMaskZoneSeasonVector, ...
                        timestamp, zoneName, [lstR2, lstRMSE]);
                    exportgraphics(f, zoneScatterSeasonPath);
                    close all;
                end
            else
                lstRMSE = nan; lstR2 = nan;
                amsr2LstMaskZoneSeasonVector = zeros(modisLstMaskZoneSeasonVectorN, 1) * nan;
                amsr2LstMaskZoneSeasonArray = zeros(amsr2RowN, amsr2ColN, seasonIndexN);
                amsr2LstCnZoneSeasonArray = zeros(amsr2RowN, amsr2ColN, seasonIndexN);
            end
            rmseSeasonArray(j, n) = lstRMSE;
            nSeasonArray(j, n) = modisLstMaskZoneSeasonVectorN;
            r2SeasonArray(j, n) = lstR2;

            % 将反演的当前季节AMSR2 LST保存到年度矩阵中.
            zonesLcSeasonIndexArray = zonesLcIndexArray(:, :, seasonIndex);
            zonesLcSeasonIndexArray2 = false(amsr2RowN, amsr2ColN, validDateN);
            zonesLcSeasonIndexArray2(:, :, seasonIndex) = zonesLcSeasonIndexArray;
            amsr2LstMaskYearArray2(zonesLcSeasonIndexArray2) = ...
                amsr2LstMaskZoneSeasonArray(zonesLcSeasonIndexArray);
            amsr2LstCnYearArray2(zonesLcSeasonIndexArray2) = ...
                amsr2LstCnZoneSeasonArray(zonesLcSeasonIndexArray);
            clear zonesLcSeasonIndexArray zonesLcSeasonIndexArray2
            clear amsr2LstMaskZoneSeasonArray amsr2LstCnZoneSeasonArray
        end

        % -------------------------------------------------------------------------------------
        % 按月尺度回归AMSR2 BT和MODIS LST, 并获取反演的AMSR2 LST影像.
        for n = 1: monthN
            monthIndex = (validYearMonthList == n);
            monthIndexN = sum(monthIndex);

            % 筛选当前月份的数据矩阵.
            modisLstMaskZoneMonthArray = modisLstMaskZoneYearArray(:, :, monthIndex);
            [amsr2HMaskZoneMonthCell, amsr2VMaskZoneMonthCell] = deal(cell(1, channelN));
            [amsr2HCnZoneMonthCell, amsr2VCnZoneMonthCell] = deal(cell(1, channelN));
            for k = 1: channelN
                amsr2HMaskZoneMonthCell{k} = amsr2HMaskZoneYearCell{k}(:, :, monthIndex);
                amsr2VMaskZoneMonthCell{k} = amsr2VMaskZoneYearCell{k}(:, :, monthIndex);
                amsr2HCnZoneMonthCell{k} = amsr2HCnZoneYearCell{k}(:, :, monthIndex);
                amsr2VCnZoneMonthCell{k} = amsr2VCnZoneYearCell{k}(:, :, monthIndex);
            end
            amsr2QdCnZoneMonthCell = {
                (amsr2VCnZoneMonthCell{4} - amsr2VCnZoneMonthCell{3}).^2, ...
                (amsr2VCnZoneMonthCell{4} - amsr2VCnZoneMonthCell{2}).^2};

            % 按照回归函数的格式要求, 将参与回归的影像矩阵重新排列.
            valueIndex = find(~isnan(modisLstMaskZoneMonthArray));
            modisLstMaskZoneMonthVector = double(modisLstMaskZoneMonthArray(valueIndex));
            [amsr2HMaskZoneMonthVector, amsr2VMaskZoneMonthVector] = deal(cell(1, channelN));
            for k = 1: channelN
                amsr2HMaskZoneMonthVector{k} = amsr2HMaskZoneMonthCell{k}(valueIndex);
                amsr2VMaskZoneMonthVector{k} = amsr2VMaskZoneMonthCell{k}(valueIndex);
            end
            amsr2QdMaskZoneMonthVector = [
                (amsr2VMaskZoneMonthVector{4} - amsr2VMaskZoneMonthVector{3}).^2, ...
                (amsr2VMaskZoneMonthVector{4} - amsr2VMaskZoneMonthVector{2}).^2];

            amsr2BtMaskZoneMonthRecords = double(cell2mat([amsr2HMaskZoneMonthVector, ...
                amsr2VMaskZoneMonthVector, amsr2QdMaskZoneMonthVector]));
            clear amsr2HMaskZoneMonthCell amsr2VMaskZoneMonthCell modisLstMaskZoneMonthArray
            clear amsr2HMaskZoneMonthVector amsr2VMaskZoneMonthVector amsr2QdMaskZoneMonthVector

            % 样本数大于2时才能执行逐步回归, 否则该分区不能反演AMSR2 LST.
            % 系数 [10H, 18H, 23H, 36H, 06H, 07H, 89H, 10V, 18V, 23V, 36V, 06V, 07V, 89V, 常数].
            modisLstMaskZoneMonthVectorN = length(modisLstMaskZoneMonthVector);
            if modisLstMaskZoneMonthVectorN >= 2
                timestamp = sprintf('%s %s %s', yearStr, daynight, monthList{n});
                fprintf('分区%d 月尺度 回归: %s.\n', zonesLcCode, timestamp)
                mdl = stepwiselm(amsr2BtMaskZoneMonthRecords, modisLstMaskZoneMonthVector, ...
                    Lower='constant', Upper='linear', Criterion='aic');
                lstRMSE = mdl.RMSE;
                lstR2 = mdl.Rsquared.Ordinary;
                amsr2LstMaskZoneMonthVector = mdl.Fitted;
                variableIndex = find(mdl.VariableInfo.InModel == 1);
                pMonth = mdl.Coefficients.Estimate;
                pMonth = pMonth([2:end, 1]);
                coefficientMonthArray(j, [variableIndex; end], n) = pMonth;

                amsr2LstMaskZoneMonthVector2 = zeros(amsr2RowN * amsr2ColN * monthIndexN, 1);
                amsr2LstMaskZoneMonthVector2(valueIndex) = amsr2LstMaskZoneMonthVector;
                amsr2LstMaskZoneMonthArray = reshape(amsr2LstMaskZoneMonthVector2, ...
                    amsr2RowN, amsr2ColN, monthIndexN);
                clear amsr2LstMaskZoneMonthVector2

                % 根据回归系数计算分区内的AMSR2 LST(初始化矩阵为常数项截距).
                amsr2LstCnZoneMonthArray = repmat(pMonth(end), amsr2RowN, amsr2ColN, monthIndexN);
                for k = 1: channelN
                    amsr2LstCnZoneMonthArray = amsr2LstCnZoneMonthArray + ...
                        amsr2HCnZoneMonthCell{k} * coefficientMonthArray(j, k, n) + ...
                        amsr2VCnZoneMonthCell{k} * coefficientMonthArray(j, k + channelN, n);
                end
                amsr2LstCnZoneMonthArray = amsr2LstCnZoneMonthArray + ...
                    amsr2QdCnZoneMonthCell{:, 1} * coefficientMonthArray(j, k*2 + 1, n) + ...
                    amsr2QdCnZoneMonthCell{:, 2} * coefficientMonthArray(j, k*2 + 2, n);
                % nan值改为0, 便于算完所有分区AMSR2 LST后将结果相加.
                amsr2LstCnZoneMonthArray(isnan(amsr2LstCnZoneMonthArray)) = 0;
                clear amsr2HCnZoneMonthCell amsr2VCnZoneMonthCell amsr2QdCnZoneMonthCell

                % 输出AMSR2 BT(LST)和MODIS LST的月份散点图.
                zoneScatterMonthName = sprintf('LstScatter_%s_%s.tif', zoneName, timestamp);
                zoneScatterMonthPath = fullfile(lstScatterYearDir, zoneScatterMonthName);
                if ~exist(zoneScatterMonthPath, 'file')
                    f = lstScatter(amsr2LstMaskZoneMonthVector, modisLstMaskZoneMonthVector,...
                        timestamp, zoneName, [lstR2, lstRMSE]);
                    exportgraphics(f, zoneScatterMonthPath);
                    close all;
                end
            else
                lstRMSE = nan; lstR2 = nan;
                amsr2LstMaskZoneMonthVector = zeros(modisLstMaskZoneMonthVectorN, 1) * nan;
                amsr2LstMaskZoneMonthArray = zeros(amsr2RowN, amsr2ColN, monthIndexN);
                amsr2LstCnZoneMonthArray = zeros(amsr2RowN, amsr2ColN, monthIndexN);
            end
            rmseMonthArray(j, n) = lstRMSE;
            nMonthArray(j, n) = modisLstMaskZoneMonthVectorN;
            r2MonthArray(j, n) = lstR2;

            % 将反演的当前月份AMSR2 LST保存到年度矩阵中.
            zonesLcMonthIndexArray = zonesLcIndexArray(:, :, monthIndex);
            zonesLcMonthIndexArray2 = false(amsr2RowN, amsr2ColN, validDateN);
            zonesLcMonthIndexArray2(:, :, monthIndex) = zonesLcMonthIndexArray;
            amsr2LstMaskYearArray3(zonesLcMonthIndexArray2) = ...
                amsr2LstMaskZoneMonthArray(zonesLcMonthIndexArray);
            amsr2LstCnYearArray3(zonesLcMonthIndexArray2) = ...
                amsr2LstCnZoneMonthArray(zonesLcMonthIndexArray);
            clear zonesLcMonthIndexArray zonesLcMonthIndexArray2
            clear amsr2LstMaskZoneMonthArray amsr2LstCnZoneMonthArray
        end
    end
    % ==============================================================================================
    % 模型优化.
    % 模型优化基于月尺度模型. 当样本数太少导致月尺度模型无法回归或精度较差时, 使用所在季节或年份的模型替换月
    %   尺度模型.

    % 计算各纯像元分区的样本数面积比例. 由于每日的积雪变化, 普通分区(1-62)的面积每日都在变化, 所以使用分区的
    %   平均像元数计算像元面积比.
    monthSampleRatioArray = zeros(pureLcCodeN, monthN);
    seasonSampleRatioArray = zeros(pureLcCodeN, seasonN);
    yearSampleRatioVector = zeros(pureLcCodeN, 1);
    for j = 1: pureLcCodeN
        zonesLcPixelCount = sum(fixedZonesLcArray == pureLcCodeList(j), 'all') / validDateN;
        for k = 1: monthN
            monthSampleRatioArray(j, k) = nMonthArray(j, k) / zonesLcPixelCount;
        end
        for k = 1: seasonN
            seasonSampleRatioArray(j, k) = nSeasonArray(j, k) / zonesLcPixelCount;
        end
        yearSampleRatioVector(j) = nYearVector(j) / zonesLcPixelCount;
    end

    % 分别计算季节尺度和年尺度的RMSE相对于月尺度RMSE的增减幅度.
    rmseSeasonArray2 = [repmat(rmseSeasonArray(:, 1), 1, 3), repmat(rmseSeasonArray(:, 2), 1, 3),...
        repmat(rmseSeasonArray(:, 3), 1, 3), repmat(rmseSeasonArray(:, 4), 1, 3)];
    rmseYearArray2 = repmat(rmseYearVector, 1, 12);
    rmseRateM2sArray = abs(rmseSeasonArray2 - rmseMonthArray) ./ rmseSeasonArray2;
    rmseRateM2yArray = abs(rmseYearArray2 - rmseMonthArray) ./ rmseYearArray2;

    % 当月尺度模型的像元样本数比例小于1时, 鉴于样本代表性问题, 考虑用季节或年尺度模型替换. 若该月所在的季节
    %   模型中像元样本数比例大于3, 仅当样本比例大于0.5且RMSE增减幅度大于0.2时不替换, 其他情况均用季节模型替
    %   换月尺度模型. 若该月所在季节模型样本比例小于3, 进一步看年尺度模型, 替换规则与季节中的一样.
    fixedCoefficientMonthArray = coefficientMonthArray;
    fixedRmseMonthArray = rmseMonthArray;
    fixedR2MonthArray = r2MonthArray;
    fixedNMonthArray = nMonthArray;
    replaceIndexArray = ones(pureLcCodeN, monthN);  % 1表示月, 2表示季节, 3表示年.
    for j = 1: pureLcCodeN
        for k = 1: monthN
            if monthSampleRatioArray(j, k) < 1
                for n = 1: seasonN  % 确定当前月所属季节.
                    if ismember(k, seasonMonthList{n})
                        break 
                    end
                end
                if seasonSampleRatioArray(j, n) > 3
                    if monthSampleRatioArray(j, k) < 0.5 || rmseRateM2sArray(j, k) < 0.2
                        fixedCoefficientMonthArray(j, :, k) = coefficientSeasonArray(j, :, n);
                        fixedRmseMonthArray(j, k) = rmseSeasonArray(j, n);
                        fixedR2MonthArray(j, k) = r2SeasonArray(j, n);
                        fixedNMonthArray(j, k) = nSeasonArray(j, n);
                        monthSampleRatioArray(j, k) = seasonSampleRatioArray(j, n);
                        replaceIndexArray(j, k) = 2;
                    end
                else
                    if monthSampleRatioArray(j, k) < 0.5 || rmseRateM2yArray(j, k) < 0.2
                        fixedCoefficientMonthArray(j, :, k) = coefficientYearArray(j, :);
                        fixedRmseMonthArray(j, k) = rmseYearVector(j);
                        fixedR2MonthArray(j, k) = r2YearVector(j);
                        fixedNMonthArray(j, k) = nYearVector(j);
                        monthSampleRatioArray(j, k) = yearSampleRatioVector(j);
                        replaceIndexArray(j, k) = 3;
                    end
                end
            end
        end
    end

    % 用空间上最临近分区的模型替换没有足够样本建立模型的分区.
    % 注: 目前发现没有模型的分区的相邻分区都有模型, 因此直接用其替换, 没有考虑相邻分区也没有模型的情况.
    coeffIndicatorArray = sum(fixedCoefficientMonthArray == 0, 2);
    for j = 1: pureLcCodeN
        for k = 1: monthN
            if coeffIndicatorArray(j, :, k) == variablesN
                zoneLcDiff = abs(int16(pureLcCodeList) - int16(pureLcCodeList(j)));
                zoneLcDiffMin2 = mink(zoneLcDiff, 2);
                zoneLcIndex = find(zoneLcDiff == zoneLcDiffMin2(2), 1);
                fixedCoefficientMonthArray(j, :, k) = fixedCoefficientMonthArray(zoneLcIndex, :, k);
                fixedRmseMonthArray(j, k) = fixedRmseMonthArray(zoneLcIndex, k);
                fixedR2MonthArray(j, k) = fixedR2MonthArray(zoneLcIndex, k);
                fixedNMonthArray(j, k) = fixedNMonthArray(zoneLcIndex, k);
                replaceIndexArray(j, k) = replaceIndexArray(zoneLcIndex, k);
            end
        end
    end

    % 处理实际不存在(8大分区中的分区没有该地表覆盖类型在AMSR2像元尺度上以60%为阈值的纯像元), 但理论上有其编
    %   号的地表覆盖类型分区. 虽然这些地表类型无法获取对应的LST反演模型, 但反演包含此地表类型的混合像元LST时
    %   却需要该类型的LST反演模型, 因此需要用其他区域对应的同一地表类型LST反演模型代替.
    % 8大区编号: 1,东北东部; 2,华北; 3,华南; 4,西南; 5,西部东部; 6,东北西部; 7,西北西部; 8,青藏高原.
    % 各地表覆盖类型所在大区的替换策略, 分区编号按相似程度从高往低排列:
    %   1: [6, 5, 7, 2, 3, 4, 8]    2: [5, 1, 6, 3, 4, 7, 8]    3: [4, 2, 1, 6, 5, 7, 8]
    %   4: [3, 2, 1, 6, 5, 8, 7]    5: [2, 6, 1, 7, 8, 4, 3]    6: [1, 5, 2, 7, 8, 4, 3]
    %   7: [5, 8, 6, 1, 2, 4, 3]    8: [7, 5, 6, 1, 2, 4, 3]
    % 注: 目前1-62分区, 110-200分区都有模型. 300 ~ 600+的分区存在没有模型的分区, 因此替换分区编码的计算方
    %   法仅考虑这些分区, 可能不适用于1-62分区的计算.
    replaceRegions = {[6 5 7 2 3 4 8]; [5 1 6 3 4 7 8]; [4 2 1 6 5 7 8]; [3 2 1 6 5 8 7]; ...
        [2 6 1 7 8 4 3]; [1 5 2 7 8 4 3]; [5 8 6 1 2 4 3]; [7 5 6 1 2 4 3]};    
    zonesLcFullCodeN = length(zonesLcFullCodeList);
    fixedCoefficientMonthArray2 = zeros(zonesLcFullCodeN, variablesN, monthN);
    [fixedRmseMonthArray2, fixedR2MonthArray2] = deal(zeros(zonesLcFullCodeN, monthN));
    [fixedNMonthArray2, replaceIndexArray2] = deal(zeros(zonesLcFullCodeN, monthN));
    for j = 1: zonesLcFullCodeN
        zonesLcCode = zonesLcFullCodeList(j);
        if ismember(zonesLcCode, pureLcCodeList)
            % 当理论的纯像元分区实际存在, 则直接采用实际分区的模型.
            zoneLcIndex = find(pureLcCodeList == zonesLcCode);
        else
            % 当理论纯像元分区实际不存在, 按上述替换策略, 用实际存在的相似分区模型作为该分区的模型.
            remCode = mod(zonesLcCode, 100);
            lcCode = zonesLcCode - remCode;
            if (300 <= lcCode && lcCode <= 500) || (lcCode == 600 && remCode > 90)
                regionCode = remCode - 90;
                replaceRegionList = replaceRegions{regionCode};
                for k = 1: length(replaceRegionList)
                    replaceRegion = replaceRegionList(k);
                    replaceZoneLcCode = lcCode + replaceRegion + 90;
                    if ismember(replaceZoneLcCode, pureLcCodeList)
                        zoneLcIndex = find(pureLcCodeList == replaceZoneLcCode);
                        break
                    end
                end
            elseif lcCode == 600 && remCode < 90
                zoneLcDiff = abs(int16(zonesLcCode) - int16(pureLcCodeList));
                zoneLcIndex = find(zoneLcDiff == min(zoneLcDiff), 1);
            end
        end
        fixedCoefficientMonthArray2(j, :, :) = fixedCoefficientMonthArray(zoneLcIndex, :, :);
        fixedRmseMonthArray2(j, :) = fixedRmseMonthArray(zoneLcIndex, :);
        fixedR2MonthArray2(j, :) = fixedR2MonthArray(zoneLcIndex, :);
        fixedNMonthArray2(j, :) = fixedNMonthArray(zoneLcIndex, :);
        replaceIndexArray2(j, :) = replaceIndexArray(zoneLcIndex, :);
    end

    % 单独处理个别有问题分区模型.
    if strcmp(yearStr, '2014') && strcmp(daynight, 'Night')
        fixedCoefficientMonthArray2(80:86, :, 12) = fixedCoefficientMonthArray2(80:86, :, 11);
        fixedRmseMonthArray2(80:86, 12) = fixedRmseMonthArray2(80:86, 11);
        fixedR2MonthArray2(80:86, 12) = fixedR2MonthArray2(80:86, 11);
        fixedNMonthArray2(80:86, 12) = fixedNMonthArray2(80:86, 11);
        replaceIndexArray2(80:86, 12) = replaceIndexArray2(80:86, 11);
    end
    if strcmp(yearStr, '2017') && strcmp(daynight, 'Day')
        fixedCoefficientMonthArray2(95, :, 2) = fixedCoefficientMonthArray2(95, :, 1);
        fixedRmseMonthArray2(95, 2) = fixedRmseMonthArray2(95, 1);
        fixedR2MonthArray2(95, 2) = fixedR2MonthArray2(95, 1);
        fixedNMonthArray2(95, 2) = fixedNMonthArray2(95, 1);
        replaceIndexArray2(95, 2) = replaceIndexArray2(95, 1);
    end

    fixedCoefficientMonthArray = fixedCoefficientMonthArray2;
    fixedRmseMonthArray = fixedRmseMonthArray2;
    fixedR2MonthArray = fixedR2MonthArray2;
    fixedNMonthArray = fixedNMonthArray2;
    replaceIndexArray = replaceIndexArray2;
    clear fixed*MonthArray2 replaceIndexArray2

    % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if 0
        % 采用RMSE最小时间尺度上的回归系数.
        fixedCoefficientMonthArray = coefficientMonthArray;
        fixedRmseMonthArray = rmseMonthArray;
        fixedR2MonthArray = r2MonthArray;
        fixedNMonthArray = nMonthArray;
        replaceIndexArray = ones(pureLcCodeN, monthN);  % 1表示月, 2表示季节, 3表示年.
        for j = 1: pureLcCodeN
            for k = 1: monthN
                % 寻找当前月份所属的季节.
                for m = 1: seasonN
                    if ismember(k, seasonMonthList{m})
                        break
                    end
                end
                % 取月尺度, 季节尺度, 年尺度最小RMSE的回归系数, 作为最终反演该月LST的系数.
                rmseGroup = [rmseMonthArray(j, k), rmseSeasonArray(j, m), rmseYearVector(j)];
                r2Group = [r2MonthArray(j, k), r2SeasonArray(j, m), r2YearVector(j)];
                nGroup = [nMonthArray(j, k), nSeasonArray(j, m), nYearVector(j)];
                coeffGroup = [coefficientMonthArray(j, :, k); coefficientSeasonArray(j, :, m); ...
                    coefficientYearArray(j, :)];

                % 如果最小的RMSE不是月尺度, 用最小RMSE的时间尺度替换.
                minRmseIndex = find(rmseGroup == min(rmseGroup), 1);
                if minRmseIndex > 1
                    fixedRmseMonthArray(j, k) = rmseGroup(minRmseIndex);
                    fixedR2MonthArray(j, k) = r2Group(minRmseIndex);
                    fixedNMonthArray(j, k) = nGroup(minRmseIndex);
                    fixedCoefficientMonthArray(j, :, k) = coeffGroup(minRmseIndex, :);
                    replaceIndexArray(j, k) = minRmseIndex;
                end
            end
        end
    end    
    % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    % 保存优化后的回归系数和精度统计指标.
    save(regressPureMatPath, 'nYearVector', 'nSeasonArray', 'nMonthArray', 'r2YearVector', ...
        'r2SeasonArray', 'r2MonthArray', 'rmseYearVector', 'rmseSeasonArray', 'rmseMonthArray', ...
        'coefficientYearArray', 'coefficientSeasonArray', 'coefficientMonthArray', ...
        'fixedNMonthArray','fixedR2MonthArray','fixedRmseMonthArray','fixedCoefficientMonthArray',...
        'amsr2LstMaskYearArray1', 'amsr2LstCnYearArray1', 'amsr2LstMaskYearArray2', ...
        'amsr2LstCnYearArray2', 'amsr2LstMaskYearArray3', 'amsr2LstCnYearArray3',...
        'pureLcCodeList', 'validDateList', 'replaceIndexArray', 'monthSampleRatioArray', ...
        'seasonSampleRatioArray', 'yearSampleRatioVector');
end

% system('shutdown -s -t 60');

%% 自定义函数.
% AMSR2和MODIS地表温度的散点图.
function f = lstScatter(amsr2LstVector, modisLstVector, zoneName, timestamp, r2Rmse)
validIndex =  ~isnan(amsr2LstVector) & ~isnan(modisLstVector);
amsr2LstVector = amsr2LstVector(validIndex);
modisLstVector = modisLstVector(validIndex);

lstBias = mean(amsr2LstVector - modisLstVector);
lstMAE = mean(abs(amsr2LstVector - modisLstVector));

f = figure; f.Visible = false;
plot(amsr2LstVector, modisLstVector, '.k', [220, 360], [220, 360], 'r');
xlabel('AMSR2 LST (K)'); ylabel('MODIS LST (K)');
title(sprintf('AMSR2 LST vs MODIS LST in %s %s', zoneName, timestamp));

txt1 = ['N: ', num2str(sum(~isnan(modisLstVector)))];
txt2 = ['R^2: ', num2str(r2Rmse(1), '%.3f')];
txt3 = ['Bias: ', num2str(lstBias, '%.3f')];
txt4 = ['MAE: ', num2str(lstMAE, '%.3f')];
txt5 = ['RMSE: ', num2str(r2Rmse(2), '%.3f')];
text(0.7, 0.29, txt1, 'Units', 'normalized', 'FontSize', 12);
text(0.7, 0.23, txt2, 'Units', 'normalized', 'FontSize', 12);
text(0.7, 0.17, txt3, 'Units', 'normalized', 'FontSize', 12);
text(0.7, 0.11, txt4, 'Units', 'normalized', 'FontSize', 12);
text(0.7, 0.05, txt5, 'Units', 'normalized', 'FontSize', 12);
end
