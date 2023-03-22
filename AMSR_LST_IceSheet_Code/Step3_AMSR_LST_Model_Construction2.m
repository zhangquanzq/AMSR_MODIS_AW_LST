%% 创建AMSR LST反演模型.
% 1. 2004年11月之后的Antarctic AMSRE BT 89GHz两个极化通道的数据有条带, 导致反演的LST数据也存在条带, 因此将
%   这两个通道的数据排除在反演Antarctic AMSRE LST的算法之外.

%% 功能标记与预设参数.
% 指定研究区的标识. 1表示Antarctic, 2表示Greenland.
flg1 = 1;
% 指定微波数据类型的标识. 1表示AMSRE, 2表示AMSR2.
flg2 = 2;
% 指定白天和晚上的标记. 1表示白天(升轨), 2表示晚上(降轨).
flg3 = 1;
% 指定分区个数的标识. 1表示Antarctic的5, 2表示Antarctic的20, 3表示Greenland的6.
flg4 = 1;

% AMSRE/2类型, 研究区.
region = {'Antarctic', 'Greenland'};
region = region{flg1};

amsrType = {'AMSRE', 'AMSR2'};
amsrType = amsrType{flg2};

% AMSRE/2数据的通道, 极化, 以及回归模型中的变量个数(6/7个通道 + 2个二次项 + 1常数项).
channelList = {["10", "18", "23", "36", "06", "89"], ["10", "18", "23", "36", "06", "07", "89"]};
channelList = channelList{flg2}; if flg1 == 1 && flg2 == 1, channelList = channelList(1:end-1); end
channelN = length(channelList);
polarizeN = length(["H", "V"]);
variablesN = channelN * polarizeN + 3;

% 有数据的年份月份列表(时间区间: 2003/01/01-2011/09/27, 2012/07/02-2020/12/31).
yearList = {2003: 2011, 2012: 2020};
yearList = yearList{flg2};

% 各季节[春, 夏, 秋, 冬]包含的月份, 和各月份的名称.
seasonMonthList = {[3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 1, 2]};
seasonNameList = {'Spring', 'Summer', 'Autumn', 'Winter'};
seasonN = length(seasonNameList);
monthList = {'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'};
monthN = length(monthList);

% 昼夜标记.
daynight = {'Day', 'Night'};
daynight = daynight{flg3};

% AMSRE/2 BT影像中的NoData.
amsrNodata = {[0, 65536], [65534, 65535]};
amsrNodata = amsrNodata{flg1};

% 分区个数, 分区像元Nodata.
zoneN = [5, 20, 6];
zoneN = zoneN(flg4);

% 判断研究区和分区是否匹配.
if (flg1 == 1 && flg4 ==3) || (flg1 == 2 && ismember(flg4, [1, 2]))
    error('指定的研究区和分区个数不匹配!')
end

%% 路径.
% 根目录.
rootDir = 'E:\AMSR_LST_IceSheet\';
addpath(fullfile(rootDir, 'Code\Functions'));
dataDir = fullfile(rootDir, 'Data');

% 输入数据路径.
amsrRegionMatDir = fullfile(dataDir, sprintf('%s_2_BT_Prj%s_Matlab', amsrType, region));
amsrMaskMatDir = fullfile(dataDir, sprintf('%s_3_BT_Mask%s_Matlab', amsrType, region));
modisLstMaskMatDir = fullfile(dataDir, sprintf('MYD11A1_3_Mask%s_Matlab', region));
ccsevMatDir = fullfile(dataDir, sprintf('CCSEV_%s_Matlab', region));

% 输出反演模型的路径.
regressMatDir = fullfile(dataDir, sprintf('Regression_%s_Matlab', region));
if ~exist(regressMatDir, 'dir')
    mkdir(regressMatDir)
end

% 输出统计图路径.
figDir = fullfile(rootDir, 'Figures');
if ~exist(figDir, 'dir')
    mkdir(figDir)
end
lstScatterDir = fullfile(figDir, 'LstScatter');
if ~exist(lstScatterDir, 'dir')
    mkdir(lstScatterDir)
end

%% 回归和输出.
% 分年度建立回归模型AMSRE/2 LST.
for i = 1: length(yearList)
    yearStr = num2str(yearList(i));
%     yearStr = '2005';

    rydStr = strjoin({region, yearStr, daynight}, '_');

    % 存储纯像元分区回归系数与反演的AMSRE/2 LST影像数据的Mat文件路径.
    regressPureMatName = sprintf('Regression_Pure_%s_%dzones.mat', rydStr, zoneN);
    regressPureMatPath = fullfile(regressMatDir, regressPureMatName);
    if  exist(regressPureMatPath, 'file')
        continue
    end

    % 创建输出AMSRE/2 BT(LST)和MODIS LST散点图的文件夹.
    lstScatterYearDir = fullfile(lstScatterDir, sprintf('%s_%dzones', rydStr, zoneN));
    if ~exist(lstScatterYearDir, 'dir')
        mkdir(lstScatterYearDir);
    end

    % 从Mat文件中读取研究区原始的AMSRE/2 BT数据.
    amsrMatName = sprintf('%s_BT_%s.mat', amsrType, rydStr);
    amsrMatPath = fullfile(amsrRegionMatDir, amsrMatName);
    load(amsrMatPath, 'amsr*YearArray');

    % 从Mat文件中读取Mask后的AMSRE/2 BT和MODIS LST数据.
    amsrMaskMatName = sprintf('%s_BT_Mask%s.mat', amsrType, rydStr);
    amsrMaskMatPath = fullfile(amsrMaskMatDir, amsrMaskMatName);
    load(amsrMaskMatPath, 'amsr*MaskYearArray', 'lstDateFilterList');

    modisLstMaskMatName = sprintf('MYD11A1_Mask%s.mat', rydStr);
    modisLstMaskMatPath = fullfile(modisLstMaskMatDir, modisLstMaskMatName);
    load(modisLstMaskMatPath, 'amsr*Ref', 'modisLstMaskYearArray');  % 数据类型: single

    % 从Mat文件中读取CCSEV.
    ccsevMatPath = fullfile(ccsevMatDir, sprintf('CCSEV_%s_%dzones.mat', yearStr, zoneN));
    load(ccsevMatPath, 'zonesLcCodeList', 'zonesLcFullCodeList', 'lcCodeList', 'zonesLcLayer');

    % 获取AMSR2 BT影像的行列数.
    [amsrRowN, amsrColN] = size(modisLstMaskYearArray, 1, 2);

    % 获取AMSR2 BT, MODIS LST以及分区数据的共有日期列表.
    validDateList = lstDateFilterList;
    validYearMonthList = datetime(validDateList , 'InputFormat', 'yyyyMMdd').Month;
    validDateN = length(validDateList);

    % 共有日期的分区矩阵和MODIS LST数据.
    zonesLcArray = repmat(zonesLcLayer, 1, 1, validDateN);

    % 共有日期的AMSRE/2 BT数据, 将不同通道的数据拆分到单独的矩阵, 并存入元胞数组中.
    % 通道的排序: [10H, 10V, 18H, 18V, 23H, 23V, 36H, 36V, 06H, 06V, 07H, 07V, 89H, 89V]
    [amsrHMaskYearCell, amsrVMaskYearCell] = deal(cell(1, channelN));
    for k = 1: channelN
        amsrHChannelArray = eval(sprintf('%sH%sMaskYearArray', lower(amsrType), channelList{k}));
        amsrVChannelArray = eval(sprintf('%sV%sMaskYearArray', lower(amsrType), channelList{k}));
        amsrHMaskYearCell{k} = single(amsrHChannelArray);
        amsrVMaskYearCell{k} = single(amsrVChannelArray);
    end
    clear amsrHChannelArray amsrVChannelArray amsr*MaskYearArray

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
    [amsrLstMaskYearArray1, amsrLstRegionYearArray1, amsrLstMaskYearArray2, ...
        amsrLstRegionYearArray2, amsrLstMaskYearArray3, amsrLstRegionYearArray3] = ...
        deal(zeros(amsrRowN, amsrColN, validDateN, 'single'));

    % ==============================================================================================
    % 获取反演AMSR2 LST的系数, 精度, 以及掩膜区AMSR2 LST.
    for j = 1: pureLcCodeN
        zonesLcCode = pureLcCodeList(j);
        zoneName = sprintf('Zone %d', zonesLcCode);
        fprintf('分区%d, %s年.\n', zonesLcCode, yearStr);

        % 获取当前分区的位置索引.
        zonesLcIndexArray = (zonesLcArray == zonesLcCode);

        % 从一整年的数组中提取当前分区的原始AMSR2 BT影像, 以及掩膜后的AMSR2 BT和MODIS LST影像.
        modisLstMaskZoneYearArray = setnan(modisLstMaskYearArray, ~zonesLcIndexArray);

        [amsrHZoneYearCell, amsrVZoneYearCell] = deal(cell(1, channelN));
        [amsrHMaskZoneYearCell, amsrVMaskZoneYearCell] = deal(cell(1, channelN));
        for k = 1 : channelN
            amsrHYearArrayStr = sprintf('%sH%sYearArray', lower(amsrType), channelList{k});
            amsrVYearArrayStr = sprintf('%sV%sYearArray', lower(amsrType), channelList{k});
            amsrHYearArray = single(eval(amsrHYearArrayStr));
            amsrVYearArray = single(eval(amsrVYearArrayStr));
            amsrHYearArray(ismember(amsrHYearArray, amsrNodata)) = nan;
            amsrVYearArray(ismember(amsrVYearArray, amsrNodata)) = nan;
            amsrHZoneYearCell{k} = setnan(amsrHYearArray, ~zonesLcIndexArray) / 100;
            amsrVZoneYearCell{k} = setnan(amsrVYearArray, ~zonesLcIndexArray) / 100;
            amsrHMaskZoneYearCell{k} = setnan(amsrHMaskYearCell{k}, ~zonesLcIndexArray) / 100;
            amsrVMaskZoneYearCell{k} = setnan(amsrVMaskYearCell{k}, ~zonesLcIndexArray) / 100;
        end
        amsrQdZoneYearCell = {(amsrVZoneYearCell{4} - amsrVZoneYearCell{3}).^2, ...
            (amsrVZoneYearCell{4} - amsrVZoneYearCell{2}).^2};
        clear amsrHYearArray amsrVYearArray
        
        % ------------------------------------------------------------------------------------------
        % 按年尺度回归AMSR2 BT和MODIS LST, 并获取反演的AMSR2 LST影像.
        % 按照回归函数的格式要求, 将参与回归的影像矩阵重新排列.
        valueIndex = find(~isnan(modisLstMaskZoneYearArray));
        modisLstMaskZoneYearVector = double(modisLstMaskZoneYearArray(valueIndex));
        [amsrHMaskZoneYearVector, amsrVMaskZoneYearVector] = deal(cell(1, channelN));
        for k = 1: channelN
            amsrHMaskZoneYearVector{k} = amsrHMaskZoneYearCell{k}(valueIndex);
            amsrVMaskZoneYearVector{k} = amsrVMaskZoneYearCell{k}(valueIndex);
        end
        amsrQdMaskZoneYearVector = [
            (amsrVMaskZoneYearVector{4} - amsrVMaskZoneYearVector{3}).^2, ...
            (amsrVMaskZoneYearVector{4} - amsrVMaskZoneYearVector{2}).^2];
        amsrBtMaskZoneYearRecords = double(cell2mat([amsrHMaskZoneYearVector, ...
            amsrVMaskZoneYearVector, amsrQdMaskZoneYearVector]));
        clear amsrHMaskZoneYearVector amsrVMaskZoneYearVector amsrQdMaskZoneYearVector;

        % 样本数大于2时才能执行逐步回归, 否则该分区不能反演AMSR2 LST.
        % 系数矩阵 [10H, 18H, 23H, 36H, 06H, 07H, 89H, 10V, 18V, 23V, 36V, 06V, 07V, 89V, 常数].
        modisLstMaskZoneYearVectorN = length(modisLstMaskZoneYearVector);
        if modisLstMaskZoneYearVectorN >= 2
            timestamp = sprintf('%s %s', yearStr, daynight);
            fprintf('分区%d 年尺度 回归: %s.\n', zonesLcCode, timestamp)
            mdl = stepwiselm(amsrBtMaskZoneYearRecords, modisLstMaskZoneYearVector, ...
                Lower='constant', Upper='linear', Criterion='aic');
            lstRMSE = mdl.RMSE;
            lstR2 = mdl.Rsquared.Ordinary;
            amsrLstMaskZoneYearVector = mdl.Fitted;
            variableIndex = find(mdl.VariableInfo.InModel == 1);
            pYear = mdl.Coefficients.Estimate;
            pYear = pYear([2:end, 1]);
            coefficientYearArray(j, [variableIndex; end]) = pYear;

            amsrLstMaskZoneYearVector2 = single(zeros(amsrRowN * amsrColN * validDateN, 1));
            amsrLstMaskZoneYearVector2(valueIndex) = amsrLstMaskZoneYearVector;
            amsrLstMaskZoneYearArray = reshape(amsrLstMaskZoneYearVector2, ...
                amsrRowN, amsrColN, validDateN);
            clear amsrLstMaskZoneYearVector2

            % 根据回归系数计算分区内的AMSR2 LST(初始化矩阵为常数项截距).
            amsrLstZoneYearArray = repmat(pYear(end), amsrRowN, amsrColN, validDateN);
            for k = 1: channelN
                amsrLstZoneYearArray = amsrLstZoneYearArray + ...
                    amsrHZoneYearCell{k} * coefficientYearArray(j, k) + ...
                    amsrVZoneYearCell{k} * coefficientYearArray(j, k + channelN);
            end
            amsrLstZoneYearArray = amsrLstZoneYearArray + ...
                amsrQdZoneYearCell{:, 1} * coefficientYearArray(j, k*2+1) + ...
                amsrQdZoneYearCell{:, 2} * coefficientYearArray(j, k*2+2);
            % nan值改为0, 便于算完所有分区AMSR2 LST后将结果相加.
            amsrLstZoneYearArray(isnan(amsrLstZoneYearArray)) = 0;

            % 输出AMSR2 BT(LST)和MODIS LST的年度散点图.
            zoneScatterYearName = sprintf('LstScatter_%s_%s.tif', zoneName, timestamp);
            zoneScatterYearPath = fullfile(lstScatterYearDir, zoneScatterYearName);
            if  ~exist(zoneScatterYearPath, 'file')
                f = lstScatter(amsrLstMaskZoneYearVector, modisLstMaskZoneYearVector, ...
                    timestamp, zoneName, [lstR2, lstRMSE]);
                exportgraphics(f, zoneScatterYearPath);
                close all
            end
        else
            lstRMSE = nan; lstR2 = nan;
            amsrLstMaskZoneYearVector = zeros(modisLstMaskZoneYearVectorN, 1) * nan;
            amsrLstMaskZoneYearArray = zeros(amsrRowN, amsrColN, validDateN);
            amsrLstZoneYearArray = zeros(amsrRowN, amsrColN, validDateN);
        end
        rmseYearVector(j) = lstRMSE;
        nYearVector(j) = modisLstMaskZoneYearVectorN;
        r2YearVector(j) = lstR2;

        % 将反演的当年AMSR2 LST保存到年度矩阵中.
        amsrLstMaskYearArray1 = amsrLstMaskYearArray1 + amsrLstMaskZoneYearArray;
        amsrLstRegionYearArray1 = amsrLstRegionYearArray1 + amsrLstZoneYearArray;

        % ------------------------------------------------------------------------------------------
        % 按季节尺度回归AMSR2 BT和MODIS LST, 并获取反演的AMSR2 LST影像.
        for n = 1: seasonN
            seasonIndex = ismember(validYearMonthList, seasonMonthList{n});
            seasonIndexN = sum(seasonIndex);

            % 筛选当前季节的数据矩阵.
            modisLstMaskZoneSeasonArray = modisLstMaskZoneYearArray(:, :, seasonIndex);
            [amsrHMaskZoneSeasonCell, amsrVMaskZoneSeasonCell] = deal(cell(1, channelN));
            [amsrHZoneSeasonCell, amsrVZoneSeasonCell] = deal(cell(1, channelN));
            for k = 1: channelN
                amsrHMaskZoneSeasonCell{k} = amsrHMaskZoneYearCell{k}(:, :, seasonIndex);
                amsrVMaskZoneSeasonCell{k} = amsrVMaskZoneYearCell{k}(:, :, seasonIndex);
                amsrHZoneSeasonCell{k} = amsrHZoneYearCell{k}(:, :, seasonIndex);
                amsrVZoneSeasonCell{k} = amsrVZoneYearCell{k}(:, :, seasonIndex);
            end
            amsrQdZoneSeasonCell = {(amsrVZoneSeasonCell{4} - amsrVZoneSeasonCell{3}).^2, ...
                (amsrVZoneSeasonCell{4} - amsrVZoneSeasonCell{2}).^2};

            % 按照回归函数的格式要求, 将参与回归的影像矩阵重新排列.
            valueIndex = find(~isnan(modisLstMaskZoneSeasonArray));
            modisLstMaskZoneSeasonVector = double(modisLstMaskZoneSeasonArray(valueIndex));
            [amsrHMaskZoneSeasonVector, amsrVMaskZoneSeasonVector] = deal(cell(1, channelN));
            for k = 1: channelN
                amsrHMaskZoneSeasonVector{k} = amsrHMaskZoneSeasonCell{k}(valueIndex);
                amsrVMaskZoneSeasonVector{k} = amsrVMaskZoneSeasonCell{k}(valueIndex);
            end
            amsrQdMaskZoneSeasonVector = [
                (amsrVMaskZoneSeasonVector{4} - amsrVMaskZoneSeasonVector{3}).^2, ...
                (amsrVMaskZoneSeasonVector{4} - amsrVMaskZoneSeasonVector{2}).^2];
            amsrBtMaskZoneSeasonRecords = double(cell2mat([amsrHMaskZoneSeasonVector, ...
                amsrVMaskZoneSeasonVector, amsrQdMaskZoneSeasonVector]));
            clear amsrHMaskZoneSeasonCell amsrVMaskZoneSeasonCell modisLstMaskZoneSeasonArray
            clear amsrHMaskZoneSeasonVector amsrVMaskZoneSeasonVector amsrQdMaskZoneSeasonVector

            % 样本数大于2时才能执行逐步回归, 否则该分区不能反演AMSR2 LST.
            % 系数 [10H, 18H, 23H, 36H, 06H, 07H, 89H, 10V, 18V, 23V, 36V, 06V, 07V, 89V, 常数].
            modisLstMaskZoneSeasonVectorN = length(modisLstMaskZoneSeasonVector);
            if modisLstMaskZoneSeasonVectorN >= 2
                timestamp = sprintf('%s %s %s', yearStr, daynight, seasonNameList{n});
                fprintf('分区%d 季节尺度 回归: %s.\n', zonesLcCode, timestamp)
                mdl = stepwiselm(amsrBtMaskZoneSeasonRecords, modisLstMaskZoneSeasonVector,...
                    Lower='constant', Upper='linear', Criterion='aic');
                lstRMSE = mdl.RMSE;
                lstR2 = mdl.Rsquared.Ordinary;
                amsrLstMaskZoneSeasonVector = mdl.Fitted;
                variableIndex = find(mdl.VariableInfo.InModel == 1);
                pSeason = mdl.Coefficients.Estimate;
                pSeason = pSeason([2:end, 1]);
                coefficientSeasonArray(j, [variableIndex; end], n) = pSeason;

                amsrLstMaskZoneSeasonVector2 = zeros(amsrRowN * amsrColN * seasonIndexN, 1);
                amsrLstMaskZoneSeasonVector2(valueIndex) = amsrLstMaskZoneSeasonVector;
                amsrLstMaskZoneSeasonArray = reshape(amsrLstMaskZoneSeasonVector2, ...
                    amsrRowN, amsrColN, seasonIndexN);
                clear amsrLstMaskZoneSeasonVector2

                % 根据回归系数计算分区内的AMSR2 LST(初始化矩阵为常数项截距).
                amsrLstZoneSeasonArray = repmat(pSeason(end), amsrRowN, amsrColN, seasonIndexN);
                for k = 1: channelN
                    amsrLstZoneSeasonArray = amsrLstZoneSeasonArray + ...
                        amsrHZoneSeasonCell{k} * coefficientSeasonArray(j, k, n) + ...
                        amsrVZoneSeasonCell{k} * coefficientSeasonArray(j, k + channelN, n);
                end
                amsrLstZoneSeasonArray = amsrLstZoneSeasonArray + ...
                    amsrQdZoneSeasonCell{:, 1} * coefficientSeasonArray(j, k*2 + 1, n) + ...
                    amsrQdZoneSeasonCell{:, 2} * coefficientSeasonArray(j, k*2 + 2, n);
                % nan值改为0, 便于算完所有分区AMSR2 LST后将结果相加.
                amsrLstZoneSeasonArray(isnan(amsrLstZoneSeasonArray)) = 0;
                clear amsrHZoneSeasonCell amsrVZoneSeasonCell amsrQdZoneSeasonCell

                % 输出AMSR2 BT(LST)和MODIS LST的季节散点图.
                zoneScatterSeasonName = sprintf('LstScatter_%s_%s.tif', zoneName, timestamp);
                zoneScatterSeasonPath = fullfile(lstScatterYearDir, zoneScatterSeasonName);
                if ~exist(zoneScatterSeasonPath, 'file')
                    f = lstScatter(amsrLstMaskZoneSeasonVector, modisLstMaskZoneSeasonVector, ...
                        timestamp, zoneName, [lstR2, lstRMSE]);
                    exportgraphics(f, zoneScatterSeasonPath);
                    close all
                end
            else
                lstRMSE = nan; lstR2 = nan;
                amsrLstMaskZoneSeasonVector = zeros(modisLstMaskZoneSeasonVectorN, 1) * nan;
                amsrLstMaskZoneSeasonArray = zeros(amsrRowN, amsrColN, seasonIndexN);
                amsrLstZoneSeasonArray = zeros(amsrRowN, amsrColN, seasonIndexN);
            end
            rmseSeasonArray(j, n) = lstRMSE;
            nSeasonArray(j, n) = modisLstMaskZoneSeasonVectorN;
            r2SeasonArray(j, n) = lstR2;

            % 将反演的当前季节AMSR2 LST保存到年度矩阵中.
            zonesLcSeasonIndexArray = zonesLcIndexArray(:, :, seasonIndex);
            zonesLcSeasonIndexArray2 = false(amsrRowN, amsrColN, validDateN);
            zonesLcSeasonIndexArray2(:, :, seasonIndex) = zonesLcSeasonIndexArray;
            amsrLstMaskYearArray2(zonesLcSeasonIndexArray2) = ...
                amsrLstMaskZoneSeasonArray(zonesLcSeasonIndexArray);
            amsrLstRegionYearArray2(zonesLcSeasonIndexArray2) = ...
                amsrLstZoneSeasonArray(zonesLcSeasonIndexArray);
            clear zonesLcSeasonIndexArray zonesLcSeasonIndexArray2
            clear amsrLstMaskZoneSeasonArray amsrLstZoneSeasonArray
        end

        % -------------------------------------------------------------------------------------
        % 按月尺度回归AMSR2 BT和MODIS LST, 并获取反演的AMSR2 LST影像.
        for n = 1: monthN
            monthIndex = (validYearMonthList == n);
            monthIndexN = sum(monthIndex);

            % 筛选当前月份的数据矩阵.
            modisLstMaskZoneMonthArray = modisLstMaskZoneYearArray(:, :, monthIndex);
            [amsrHMaskZoneMonthCell, amsrVMaskZoneMonthCell] = deal(cell(1, channelN));
            [amsrHZoneMonthCell, amsrVZoneMonthCell] = deal(cell(1, channelN));
            for k = 1: channelN
                amsrHMaskZoneMonthCell{k} = amsrHMaskZoneYearCell{k}(:, :, monthIndex);
                amsrVMaskZoneMonthCell{k} = amsrVMaskZoneYearCell{k}(:, :, monthIndex);
                amsrHZoneMonthCell{k} = amsrHZoneYearCell{k}(:, :, monthIndex);
                amsrVZoneMonthCell{k} = amsrVZoneYearCell{k}(:, :, monthIndex);
            end
            amsrQdZoneMonthCell = {
                (amsrVZoneMonthCell{4} - amsrVZoneMonthCell{3}).^2, ...
                (amsrVZoneMonthCell{4} - amsrVZoneMonthCell{2}).^2};

            % 按照回归函数的格式要求, 将参与回归的影像矩阵重新排列.
            valueIndex = find(~isnan(modisLstMaskZoneMonthArray));
            modisLstMaskZoneMonthVector = double(modisLstMaskZoneMonthArray(valueIndex));
            [amsrHMaskZoneMonthVector, amsrVMaskZoneMonthVector] = deal(cell(1, channelN));
            for k = 1: channelN
                amsrHMaskZoneMonthVector{k} = amsrHMaskZoneMonthCell{k}(valueIndex);
                amsrVMaskZoneMonthVector{k} = amsrVMaskZoneMonthCell{k}(valueIndex);
            end
            amsrQdMaskZoneMonthVector = [
                (amsrVMaskZoneMonthVector{4} - amsrVMaskZoneMonthVector{3}).^2, ...
                (amsrVMaskZoneMonthVector{4} - amsrVMaskZoneMonthVector{2}).^2];

            amsrBtMaskZoneMonthRecords = double(cell2mat([amsrHMaskZoneMonthVector, ...
                amsrVMaskZoneMonthVector, amsrQdMaskZoneMonthVector]));
            clear amsrHMaskZoneMonthCell amsrVMaskZoneMonthCell modisLstMaskZoneMonthArray
            clear amsrHMaskZoneMonthVector amsrVMaskZoneMonthVector amsrQdMaskZoneMonthVector

            % 样本数大于2时才能执行逐步回归, 否则该分区不能反演AMSR2 LST.
            % 系数 [10H, 18H, 23H, 36H, 06H, 07H, 89H, 10V, 18V, 23V, 36V, 06V, 07V, 89V, 常数].
            modisLstMaskZoneMonthVectorN = length(modisLstMaskZoneMonthVector);
            if modisLstMaskZoneMonthVectorN >= 2
                timestamp = sprintf('%s %s %s', yearStr, daynight, monthList{n});
                fprintf('分区%d 月尺度 回归: %s.\n', zonesLcCode, timestamp)
                mdl = stepwiselm(amsrBtMaskZoneMonthRecords, modisLstMaskZoneMonthVector, ...
                    Lower='constant', Upper='linear', Criterion='aic');
                lstRMSE = mdl.RMSE;
                lstR2 = mdl.Rsquared.Ordinary;
                amsrLstMaskZoneMonthVector = mdl.Fitted;
                variableIndex = find(mdl.VariableInfo.InModel == 1);
                pMonth = mdl.Coefficients.Estimate;
                pMonth = pMonth([2:end, 1]);
                coefficientMonthArray(j, [variableIndex; end], n) = pMonth;

                amsrLstMaskZoneMonthVector2 = zeros(amsrRowN * amsrColN * monthIndexN, 1);
                amsrLstMaskZoneMonthVector2(valueIndex) = amsrLstMaskZoneMonthVector;
                amsrLstMaskZoneMonthArray = reshape(amsrLstMaskZoneMonthVector2, ...
                    amsrRowN, amsrColN, monthIndexN);
                clear amsrLstMaskZoneMonthVector2

                % 根据回归系数计算分区内的AMSR2 LST(初始化矩阵为常数项截距).
                amsrLstZoneMonthArray = repmat(pMonth(end), amsrRowN, amsrColN, monthIndexN);
                for k = 1: channelN
                    amsrLstZoneMonthArray = amsrLstZoneMonthArray + ...
                        amsrHZoneMonthCell{k} * coefficientMonthArray(j, k, n) + ...
                        amsrVZoneMonthCell{k} * coefficientMonthArray(j, k + channelN, n);
                end
                amsrLstZoneMonthArray = amsrLstZoneMonthArray + ...
                    amsrQdZoneMonthCell{:, 1} * coefficientMonthArray(j, k*2 + 1, n) + ...
                    amsrQdZoneMonthCell{:, 2} * coefficientMonthArray(j, k*2 + 2, n);
                % nan值改为0, 便于算完所有分区AMSR2 LST后将结果相加.
                amsrLstZoneMonthArray(isnan(amsrLstZoneMonthArray)) = 0;
                clear amsrHZoneMonthCell amsrVZoneMonthCell amsrQdZoneMonthCell

                % 输出AMSR2 BT(LST)和MODIS LST的月份散点图.
                zoneScatterMonthName = sprintf('LstScatter_%s_%s.tif', zoneName, timestamp);
                zoneScatterMonthPath = fullfile(lstScatterYearDir, zoneScatterMonthName);
                if ~exist(zoneScatterMonthPath, 'file')
                    f = lstScatter(amsrLstMaskZoneMonthVector, modisLstMaskZoneMonthVector,...
                        timestamp, zoneName, [lstR2, lstRMSE]);
                    exportgraphics(f, zoneScatterMonthPath);
                    close all;
                end
            else
                lstRMSE = nan; lstR2 = nan;
                amsrLstMaskZoneMonthVector = zeros(modisLstMaskZoneMonthVectorN, 1) * nan;
                amsrLstMaskZoneMonthArray = zeros(amsrRowN, amsrColN, monthIndexN);
                amsrLstZoneMonthArray = zeros(amsrRowN, amsrColN, monthIndexN);
            end
            rmseMonthArray(j, n) = lstRMSE;
            nMonthArray(j, n) = modisLstMaskZoneMonthVectorN;
            r2MonthArray(j, n) = lstR2;

            % 将反演的当前月份AMSR2 LST保存到年度矩阵中.
            zonesLcMonthIndexArray = zonesLcIndexArray(:, :, monthIndex);
            zonesLcMonthIndexArray2 = false(amsrRowN, amsrColN, validDateN);
            zonesLcMonthIndexArray2(:, :, monthIndex) = zonesLcMonthIndexArray;
            amsrLstMaskYearArray3(zonesLcMonthIndexArray2) = ...
                amsrLstMaskZoneMonthArray(zonesLcMonthIndexArray);
            amsrLstRegionYearArray3(zonesLcMonthIndexArray2) = ...
                amsrLstZoneMonthArray(zonesLcMonthIndexArray);
            clear zonesLcMonthIndexArray zonesLcMonthIndexArray2
            clear amsrLstMaskZoneMonthArray amsrLstZoneMonthArray
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
        zonesLcPixelCount = sum(zonesLcArray == pureLcCodeList(j), 'all') / validDateN;
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

    % 保存优化后的回归系数和精度统计指标.
    save(regressPureMatPath, 'nYearVector', 'nSeasonArray', 'nMonthArray', 'r2YearVector', ...
        'r2SeasonArray', 'r2MonthArray', 'rmseYearVector', 'rmseSeasonArray', 'rmseMonthArray', ...
        'coefficientYearArray', 'coefficientSeasonArray', 'coefficientMonthArray', ...
        'fixedNMonthArray','fixedR2MonthArray','fixedRmseMonthArray','fixedCoefficientMonthArray',...
        'amsrLstMaskYearArray1', 'amsrLstRegionYearArray1', 'amsrLstMaskYearArray2', ...
        'amsrLstRegionYearArray2', 'amsrLstMaskYearArray3', 'amsrLstRegionYearArray3',...
        'pureLcCodeList', 'validDateList', 'replaceIndexArray', 'monthSampleRatioArray', ...
        'seasonSampleRatioArray', 'yearSampleRatioVector');
end

system('shutdown -s -t 60')

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
