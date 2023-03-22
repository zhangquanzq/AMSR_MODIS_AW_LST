%% AMSR2地表温度反演.

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

% 各月份的名称.
monthList = {'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'};
monthN = length(monthList);

% 昼夜标记.
daynight = {'Day', 'Night'};
daynight = daynight{flg1};

% LC和分区影像代码的重分类. 主要针对水域, 冰川, 建筑这三个LC区中LST的反演.
%   [阶梯1_东北东部(1): 1-3, 阶梯1_华北(2): 4-5, 阶梯1_华南(3): 6-10]
%   [阶梯2_西南(4): 11-14, 阶梯2_西北东部(5): 15-25, 阶梯2_东北西部(6): 26-29]
%   [阶梯2_西北西部(7): 30-46， 阶梯3_青藏高原(8): 47-62]
% 重分类时, 使用了右开区间, 导致最后一个分区编码71排除在分类结果中, 62+1, 使其包含在重分类结果中.
regionNodes = [1, 4, 6, 11, 15, 26, 30, 47, 62+1];
regionsN = length(regionNodes) - 1;

% AMSR2 BT影像中的NoData.
amsr2Nodata = [65534, 65535];

%% 路径.
% 根目录.
rootDir = 'E:\AMSR2_MODIS_AW_LST';
dataDir = fullfile(rootDir, 'AMSR2_LST_Retrieval\Data\');
addpath(fullfile(rootDir, 'Code\Functions'));

% 输入数据路径.
amsr2CnMatDir = fullfile(dataDir, 'AMSR2_2_CN_Matlab');
amsr2MaskMatDir = fullfile(dataDir, 'AMSR2_3_MaskCn_Matlab');
ccsevMatDir = fullfile(dataDir, 'CCSEV_Matlab');
regressMatDir = fullfile(dataDir, 'Regression_Matlab');

% 输出反演的AMSR2 LST路径.
amsr2LstMatDir = fullfile(dataDir, 'AMSR2_4_LSTCN_Matlab');
if ~exist(amsr2LstMatDir, 'dir')
    mkdir(amsr2LstMatDir)
end
amsr2LstDir = fullfile(dataDir, 'AMSR2_4_LSTCN_TIF');
if ~exist(amsr2LstDir, 'dir')
    mkdir(amsr2LstDir)
end

%% 回归和输出.
% 分年度反演AMSR2 LST.
for i = 1: 1 % length(yearList)
%     yearStr = num2str(yearList(i));
    yearStr = '2020';

    % 输出AMSR2 LST数据路径.
    amsr2LstYearDir = fullfile(amsr2LstDir, sprintf('AMSR2_LST_%sXXXX_TIF', yearStr));
    if ~exist(amsr2LstYearDir, 'dir')
        mkdir(amsr2LstYearDir);
    end
    amsr2LstFilledYearDir = fullfile(amsr2LstDir, sprintf('AMSR2_LST_%sXXXX_Filled_TIF', yearStr));
    if ~exist(amsr2LstFilledYearDir, 'dir')
        mkdir(amsr2LstFilledYearDir);
    end

    amsr2LstYearThumbDir = fullfile(amsr2LstDir, sprintf('AMSR2_LST_%sXXXX_Thumb', yearStr));
    if ~exist(amsr2LstYearThumbDir, 'dir')
        mkdir(amsr2LstYearThumbDir);
    end

    % 从Mat文件中读取中国区原始的AMSR2 BT数据.
    amsr2CnMatPath = fullfile(amsr2CnMatDir, sprintf('AMSR2_BT_%s_%s.mat', yearStr, daynight));
    load(amsr2CnMatPath, 'amsr2H*YearArray', 'amsr2V*YearArray', 'amsr2Ref', 'dateYearList');

    % 从Mat文件中读取CCSEV.
    ccsevMatPath = fullfile(ccsevMatDir, sprintf('CCSEV_%s.mat', yearStr));
    load(ccsevMatPath, 'zonesLcCodeList', 'zonesLcFullCodeList', 'lcCodeList', 'lcDateList', ...
        'fixedZonesLcArray');

    % 从Mat文件中读取纯像元分区的回归系数, 包括实际存在的和理论上存在的所有分区.
    regressPureMatName = sprintf('Regression_Pure_%s_%s.mat', yearStr, daynight);
    regressPureMatPath = fullfile(regressMatDir, regressPureMatName);
    load(regressPureMatPath, 'fixedCoefficientMonthArray');

    % 获取共有日期的分区和数据矩阵.
    [validDateList, lstDateIndex, lcDateIndex] = intersect(dateYearList, lcDateList);
    validYearMonthList = datetime(validDateList , 'InputFormat', 'yyyyMMdd').Month;
    validDateN = length(validDateList);
    fixedZonesLcArray = fixedZonesLcArray(:, :, lcDateIndex);

    % 获取ASMR2 LST数据的行列数.
    [amsr2RowN, amsr2ColN] = size(fixedZonesLcArray, [1 2]);

    % ----------------------------------------------------------------------------------------------
    % 使用优化后的模型反演纯像元分区(1-62; 100, 110, ..., 200; 300, 400, ..., 600)的AMSR2 LST.
    amsr2LstPureYearMatName = sprintf('AMSR2_Lst_%s_%s_Pure.mat', daynight, yearStr);
    amsr2LstPureYearMatPath = fullfile(amsr2LstMatDir, amsr2LstPureYearMatName);
    if ~exist(amsr2LstPureYearMatPath, 'file')
        amsr2LstCnPureLcYearArray = zeros(amsr2RowN, amsr2ColN, validDateN, 'single');
        pureZonesLcCodeList = zonesLcCodeList(zonesLcCodeList < lcCodeList(end));  % < 1000
        for j = 1: length(pureZonesLcCodeList)
            zonesLcCode = pureZonesLcCodeList(j);
            fprintf('反演%s年%s的分区%d的AMSR2地表温度.\n', yearStr, daynight, zonesLcCode);

            % 从年度矩阵中获取纯像元分区AMSR2 BT的影像.
            zonesLcIndexArray = (fixedZonesLcArray == zonesLcCode);
            [amsr2HZoneYearCell, amsr2VZoneYearCell] = deal(cell(channelN, 1));
            for k = 1: channelN
                amsr2HYearArray = single(eval(sprintf('amsr2H%sYearArray', channelList{k})));
                amsr2VYearArray = single(eval(sprintf('amsr2V%sYearArray', channelList{k})));
                amsr2HYearArray(ismember(amsr2HYearArray, amsr2Nodata)) = nan;
                amsr2VYearArray(ismember(amsr2VYearArray, amsr2Nodata)) = nan;
                amsr2HYearArray = amsr2HYearArray(:, :, lstDateIndex) / 100;
                amsr2VYearArray = amsr2VYearArray(:, :, lstDateIndex) / 100;
                amsr2HZoneYearCell{k} = setnan(amsr2HYearArray, ~zonesLcIndexArray);
                amsr2VZoneYearCell{k} = setnan(amsr2VYearArray, ~zonesLcIndexArray);
            end
            clear amsr2HYearArray amsr2VYearArray

            % 反演纯像元分区每个月的AMSR2 LST.
            for k = 1: monthN
                monthIndex = (validYearMonthList == k);

                % 提取当前分区各通道的AMSR2 BT和二次项.
                [amsr2HZoneMonthCell, amsr2VZoneMonthCell] = deal(cell(channelN, 1));
                for n = 1: channelN
                    amsr2HZoneMonthCell{n} = amsr2HZoneYearCell{n}(:, :, monthIndex);
                    amsr2VZoneMonthCell{n} = amsr2VZoneYearCell{n}(:, :, monthIndex);
                end
                amsr2QdZoneMonthCell = {...
                    (amsr2VZoneMonthCell{4} - amsr2VZoneMonthCell{3}) .^ 2, ...
                    (amsr2VZoneMonthCell{4} - amsr2VZoneMonthCell{2}) .^ 2};

                % 反演AMSR2 LST.
                % 首先在理论上所有分区的模型参数矩阵中找到实际分区的系数.
                % 系数: [10H, 18H, 23H, 36H, 06H, 07H, 89H, 10V, 18V, 23V, 36V, 06V, 07V, 89V, 常数].
                zonsLcIndex = (zonesLcFullCodeList == zonesLcCode);
                fixedP = single(fixedCoefficientMonthArray(zonsLcIndex, :, k));
                amsr2LstZoneMonthArray = repmat(fixedP(end), amsr2RowN, amsr2ColN, sum(monthIndex));
                for n = 1: channelN
                    amsr2LstZoneMonthArray = amsr2LstZoneMonthArray + ...
                        amsr2HZoneMonthCell{n} * fixedP(n) + ...
                        amsr2VZoneMonthCell{n} * fixedP(n + channelN);
                end
                amsr2LstZoneMonthArray = amsr2LstZoneMonthArray + ...
                    amsr2QdZoneMonthCell{:, 1} * fixedP(n*2 + 1) + ...
                    amsr2QdZoneMonthCell{:, 2} * fixedP(n*2 + 2);
                amsr2LstZoneMonthArray(isnan(amsr2LstZoneMonthArray)) = 0;

                % 将反演的当前纯像元分区和月份的AMSR2 LST保存到年度矩阵中.
                zonesLcMonthIndexArray = zonesLcIndexArray(:, :, monthIndex);
                zonesLcMonthIndexArray2 = false(amsr2RowN, amsr2ColN, validDateN);
                zonesLcMonthIndexArray2(:, :, monthIndex) = zonesLcMonthIndexArray;
                amsr2LstCnPureLcYearArray(zonesLcMonthIndexArray2) = ...
                    amsr2LstZoneMonthArray(zonesLcMonthIndexArray);
                clear zonesLcMonthIndexArray zonesLcMonthIndexArray2 amsr2LstZoneMonthArray
            end
        end
        save(amsr2LstPureYearMatPath, 'amsr2LstCnPureLcYearArray', 'validDateList', 'amsr2Ref');
    end

    % ----------------------------------------------------------------------------------------------
    % 反演混合像元的AMSR2 LST.
    % 在第一次循环中, 组合包含水体, 冰川, 建筑, 积雪和其他地表覆盖类型的不完全LST. 在第二次循环中, 补上剩下
    %   的沙漠LST部分. 沙漠的比例模型与其他地表覆盖类型不同, 因此他们不能在同一个循环中处理.

    % 循环1: 遍历每个普通分区(1-62), 并寻找其中的混合像元, 然后通过对水域, 冰川, 建筑, 积雪和其他地表覆盖类
    %   型的加权平均计算不完全组分温度.
    amsr2LstMix1YearMatName = sprintf('AMSR2_Lst_%s_%s_Mix1.mat', daynight, yearStr);
    amsr2LstMix1YearMatPath = fullfile(amsr2LstMatDir, amsr2LstMix1YearMatName);
    if ~exist(amsr2LstMix1YearMatPath, 'file')
        % 读取LC的面积比例矩阵.
        load(ccsevMatPath, 'otherPctArray', 'buildingPctArray', 'glacierPctArray', ...
            'waterPctArray', 'snowPctArray');

        % 筛选有效日期的地表覆盖百分比矩阵.
        otherPctArray = otherPctArray(:, :, lcDateIndex);
        buildingPctArray = buildingPctArray(:, :, lcDateIndex);
        glacierPctArray = glacierPctArray(:, :, lcDateIndex);
        waterPctArray = waterPctArray(:, :, lcDateIndex);
        snowPctArray = snowPctArray(:, :, lcDateIndex);

        % 反演不同混合分区的AMSR2 LST.
        amsr2LstCnMix1YearArray = zeros(amsr2RowN, amsr2ColN, validDateN, 'single');
        mixZonesCodeList = zonesLcCodeList(zonesLcCodeList > lcCodeList(end));  % > 1000
        for j = 1: length(mixZonesCodeList)
            % 寻找年度分区矩阵内每个分区的混合像元. 例: 1 + 1000 表示分区1中的混合像元.
            mixZonesCode = mixZonesCodeList(j);
            zoneCode = mixZonesCode - lcCodeList(end);
            fprintf('反演%s年%s的混合像元分区%d的AMSR2地表温度.\n', yearStr, daynight, mixZonesCode);

            % 跳过没有混合像元的分区.
            mixedPixelIndexArray = (fixedZonesLcArray == mixZonesCode);
            if sum(mixedPixelIndexArray, 'all') == 0
                continue
            end

            % 获取当前分区全年所有通道混合像元的AMSR2 BT影像.
            [amsr2HMixPixelYearArray, amsr2VMixPixelYearArray] = deal(cell(1, channelN));
            for k = 1: channelN
                amsr2HYearArray = single(eval(sprintf('amsr2H%sYearArray', channelList{k})));
                amsr2VYearArray = single(eval(sprintf('amsr2V%sYearArray', channelList{k})));
                amsr2HYearArray(ismember(amsr2HYearArray, amsr2Nodata)) = nan;
                amsr2VYearArray(ismember(amsr2VYearArray, amsr2Nodata)) = nan;
                amsr2HYearArray = amsr2HYearArray(:, :, lstDateIndex) / 100;
                amsr2VYearArray = amsr2VYearArray(:, :, lstDateIndex) / 100;
                amsr2HMixPixelYearArray{k} = setnan(amsr2HYearArray, ~mixedPixelIndexArray);
                amsr2VMixPixelYearArray{k} = setnan(amsr2VYearArray, ~mixedPixelIndexArray);
            end
            clear amsr2HYearArray amsr2VYearArray

            % 获取全年当前分区中每个地表覆盖类型(水域, 冰川, 建筑, 积雪, 其他)的面积比例.
            otherPctMixPixelYearArray = setnan(otherPctArray, ~mixedPixelIndexArray);
            waterPctMixPixelYearArray = setnan(waterPctArray, ~mixedPixelIndexArray);
            glacierPctMixPixelYearArray = setnan(glacierPctArray, ~mixedPixelIndexArray);
            buildingPctMixPixelYearArray = setnan(buildingPctArray, ~mixedPixelIndexArray);
            snowPctMixPixelYearArray = setnan(snowPctArray, ~mixedPixelIndexArray);

            % 标记当前混合分区的编码.
            % regionNodes: [1, 4, 6, 11, 15, 26, 30, 47, 62+1]
            % regionCodes: [1, 2, 3, 4, 5, 6, 7, 8] + 90
            % 将zoneCode转换为regionCode, 并寻找地表覆盖在zonesLcCodeList中的索引.
            for k = 1: regionsN
                if (regionNodes(k) <= zoneCode) && (zoneCode < regionNodes(k + 1))
                    regionCode = k + 90;
                    break
                end
            end
            % 积雪在东北, 青藏高原按原来的分区, 在其他地区按大分区, 因此需单独确定积雪编码.
            if ismember(regionCode, [1, 6, 8] + 90)
                snowCode = zoneCode;
            else
                snowCode = regionCode;
            end

            % 使用优化后的月尺度模型计算混合像元的AMSR2 LST.
            for k = 1: monthN
                monthIndex = (validYearMonthList == k);

                % 跳过没有混合像元的月份.
                mixedPixelMonthIndexArray = mixedPixelIndexArray(:, :, monthIndex);
                if sum(mixedPixelMonthIndexArray, 'all') == 0
                    continue
                end

                % 获取当前月份的AMSR2 BT影像.
                [amsr2HMixPixelMonthCell, amsr2VMixPixelMonthCell] = deal(cell(1, channelN));
                for m = 1: channelN
                    amsr2HMixPixelMonthCell{m} = amsr2HMixPixelYearArray{m}(:, :, monthIndex);
                    amsr2VMixPixelMonthCell{m} = amsr2VMixPixelYearArray{m}(:, :, monthIndex);
                end
                amsr2QdMixPixelMonthCell = {
                    (amsr2VMixPixelMonthCell{4} - amsr2VMixPixelMonthCell{3}) .^ 2, ...
                    (amsr2VMixPixelMonthCell{4} - amsr2VMixPixelMonthCell{2}) .^ 2};

                % 获取当前月份和分区中每个地表覆盖类型(水域, 冰川, 建筑, 积雪, 其他)的面积比例.
                otherPctMixPixelMonthArray = otherPctMixPixelYearArray(:, :, monthIndex);
                waterPctMixPixelMonthArray = waterPctMixPixelYearArray(:, :, monthIndex);
                glacierPctMixPixelMonthArray = glacierPctMixPixelYearArray(:, :, monthIndex);
                buildingPctMixPixelMonthArray = buildingPctMixPixelYearArray(:, :, monthIndex);
                snowPctMixPixelMonthArray = snowPctMixPixelYearArray(:, :, monthIndex);

                % 如果该分区包含特定的地表覆盖, 获取其回归系数, 否则将其系数设为0.
                waterCodeIndex = (zonesLcFullCodeList == (lcCodeList(2) + regionCode));
                glacierCodeIndex = (zonesLcFullCodeList == (lcCodeList(3) + regionCode));
                buildingCodeIndex = (zonesLcFullCodeList == (lcCodeList(4) + regionCode));
                snowCodeIndex = (zonesLcFullCodeList == (lcCodeList(5) + snowCode));

                % 系数: [10H, 18H, 23H, 36H, 06H, 07H, 89H, 10V, 18V, 23V, 36V, 06V, 07V, 89V, 常数].
                pOtherMonth = fixedCoefficientMonthArray(j, :, k);
                [pWaterMonth,pGlacierMonth,pSnowMonth,pBuildingMonth] = deal(zeros(1, variablesN));
                if sum(waterCodeIndex) == 1
                    pWaterMonth = fixedCoefficientMonthArray(waterCodeIndex, :, k);
                end
                if sum(glacierCodeIndex) == 1
                    pGlacierMonth = fixedCoefficientMonthArray(glacierCodeIndex, :, k);
                end
                if sum(snowCodeIndex) == 1
                    pSnowMonth = fixedCoefficientMonthArray(snowCodeIndex, :, k);
                end
                if sum(buildingCodeIndex) == 1
                    pBuildingMonth = fixedCoefficientMonthArray(buildingCodeIndex, :, k);
                end

                % 回归.
                coeffArray = zeros(amsr2RowN, amsr2ColN, sum(monthIndex), variablesN, 'single');
                for m = 1: variablesN
                    coeffArray(:, :, :, m) = ...
                        pOtherMonth(m) .* otherPctMixPixelMonthArray + ...
                        pWaterMonth(m) .* waterPctMixPixelMonthArray + ...
                        pGlacierMonth(m) .* glacierPctMixPixelMonthArray + ...
                        pBuildingMonth(m) .* buildingPctMixPixelMonthArray + ...
                        pSnowMonth(m) .* snowPctMixPixelMonthArray;
                end
                amsr2PartialLstMixedLcMonthArray = coeffArray(:, :, :, end);
                for m = 1: channelN
                    amsr2PartialLstMixedLcMonthArray = amsr2PartialLstMixedLcMonthArray + ...
                        amsr2HMixPixelMonthCell{m} .* coeffArray(:, :, :, m) + ...
                        amsr2VMixPixelMonthCell{m} .* coeffArray(:, :, :, m+channelN);
                end
                amsr2PartialLstMixedLcMonthArray = amsr2PartialLstMixedLcMonthArray + ...
                    amsr2QdMixPixelMonthCell{:, 1} .* coeffArray(:, :, :, m*2 + 1) + ...
                    amsr2QdMixPixelMonthCell{:, 2} .* coeffArray(:, :, :, m*2 + 2);
                amsr2PartialLstMixedLcMonthArray(isnan(amsr2PartialLstMixedLcMonthArray)) = 0;

                mixPixelMonthIndexArray2 = false(amsr2RowN, amsr2ColN, validDateN);
                mixPixelMonthIndexArray2(:, :, monthIndex) = mixedPixelMonthIndexArray;
                amsr2LstCnMix1YearArray(mixPixelMonthIndexArray2) = ...
                    amsr2PartialLstMixedLcMonthArray(mixedPixelMonthIndexArray);
                clear coeffArray mixPixelMonthIndexArray2 mixedPixelMonthIndexArray;
                clear amsr2PartialLstMixedLcMonthArray;
            end
        end
        save(amsr2LstMix1YearMatPath, 'amsr2LstCnMix1YearArray', 'validDateList', 'amsr2Ref')
    end

    % 循环2. 将剩余的沙漠部分LST添加到循环1获取的不完全LST中. 遍历每个沙漠分区(100, 110, ..., 200).
    % 获取沙漠分区编码.
    amsr2LstMix2YearMatName = sprintf('AMSR2_Lst_%s_%s_Mix2.mat', daynight, yearStr);
    amsr2LstMix2YearMatPath = fullfile(amsr2LstMatDir, amsr2LstMix2YearMatName);
    if  ~exist(amsr2LstMix2YearMatPath, 'file')
        % 读取沙漠分区编码数据.
        load(ccsevMatPath, 'desertCodeLayer', 'desertPctArray');
        load(amsr2LstMix1YearMatPath, 'amsr2LstCnMix1YearArray')

        desertCodeList = unique(desertCodeLayer);
        desertCodeList(desertCodeList == 0) = [];
        desertCodeArray = repmat(desertCodeLayer, [1 1 validDateN]);
        desertPctArray = desertPctArray(:, :, lcDateIndex);
        mixedLcsIndexArray = (fixedZonesLcArray > lcCodeList(end));
        amsr2LstCnMix2YearArray = amsr2LstCnMix1YearArray;
        for j = 1: length(desertCodeList)
            desertCode = desertCodeList(j);
            fprintf('反演%s年%s沙漠区%d混合像元的AMSR2地表温度.\n', yearStr, daynight, desertCode);

            % 确定每个沙漠区的混合像元.
            desertCodeMixIndexArray = (desertCodeArray == desertCode) & mixedLcsIndexArray;

            % 获取每个沙漠区的混合像元AMSR2 BT数据.
            desertPctMixYearArray = setnan(desertPctArray, ~desertCodeMixIndexArray);
            [amsr2HMixDesertYearCell, amsr2VMixDesertYearCell] = deal(cell(channelN, 1));
            for k = 1 : channelN
                amsr2HYearArray = single(eval(sprintf('amsr2H%sYearArray', channelList{k})));
                amsr2VYearArray = single(eval(sprintf('amsr2V%sYearArray', channelList{k})));
                amsr2HYearArray(ismember(amsr2HYearArray, amsr2Nodata)) = nan;
                amsr2VYearArray(ismember(amsr2VYearArray, amsr2Nodata)) = nan;
                amsr2HYearArray = amsr2HYearArray(:, :, lstDateIndex) / 100;
                amsr2VYearArray = amsr2VYearArray(:, :, lstDateIndex) / 100;
                amsr2HMixDesertYearCell{k} = setnan(amsr2HYearArray, ~desertCodeMixIndexArray);
                amsr2VMixDesertYearCell{k} = setnan(amsr2VYearArray, ~desertCodeMixIndexArray);
            end
            clear amsr2HYearArray amsr2VYearArray

            % 反演每个沙漠区混合像元的沙漠地表温度组分.
            for k = 1: monthN
                monthIndex = (validYearMonthList == k);
                desertCodeMixMonthIndexArray = desertCodeMixIndexArray(:, :, monthIndex);

                % 跳过没有混合像元沙漠分区的月份.
                if sum(desertCodeMixMonthIndexArray, 'all') == 0
                    continue
                end

                % 获取每个月份当前沙漠区的混合像元AMSR2 BT数据.
                desertPctMixMonthArray = desertPctMixYearArray(:, :, monthIndex);
                [amsr2HMixDesertMonthCell, amsr2VMixDesertMonthCell] = deal(cell(channelN, 1));
                for m = 1: channelN
                    amsr2HMixDesertMonthCell{m} = amsr2HMixDesertYearCell{m}(:, :, monthIndex);
                    amsr2VMixDesertMonthCell{m} = amsr2VMixDesertYearCell{m}(:, :, monthIndex);
                end
                amsr2QdMixDesertMonthCell = {
                    (amsr2VMixDesertMonthCell{4} - amsr2VMixDesertMonthCell{3}).^2, ...
                    (amsr2VMixDesertMonthCell{4} - amsr2VMixDesertMonthCell{2}).^2};

                % 确定沙漠区反演AMSR2 LST的回归系数.
                % 系数: [10H, 18H, 23H, 36H, 06H, 07H, 89H, 10V, 18V, 23V, 36V, 06V, 07V, 89V, 常数].
                pMonth = zeros(1, variablesN);
                desertCodeIndex = find(zonesLcFullCodeList == desertCode);
                if ~isempty(desertCodeIndex)
                    pMonth = fixedCoefficientMonthArray(desertCodeIndex, :, k);
                end

                % 回归.
                amsr2LstMix2MonthArray = repmat(pMonth(end), [amsr2RowN amsr2ColN sum(monthIndex)]);
                for m = 1: channelN
                    amsr2LstMix2MonthArray = amsr2LstMix2MonthArray + ...
                        amsr2HMixDesertMonthCell{m} .* pMonth(m) + ...
                        amsr2VMixDesertMonthCell{m} .* pMonth(m + channelN);
                end
                amsr2LstMix2MonthArray = desertPctMixMonthArray.*(amsr2LstMix2MonthArray + ...
                    amsr2QdMixDesertMonthCell{:, 1} .* pMonth(m*2 + 1) + ...
                    amsr2QdMixDesertMonthCell{:, 2} .* pMonth(m*2 + 2));
                amsr2LstMix2MonthArray(isnan(amsr2LstMix2MonthArray)) = 0;

                desertCodeMixMonthIndexArray2 = false(amsr2RowN, amsr2ColN, validDateN);
                desertCodeMixMonthIndexArray2(:, :, monthIndex) = desertCodeMixMonthIndexArray;
                amsr2LstCnMix2YearArray(desertCodeMixMonthIndexArray2) = ...
                    amsr2LstCnMix2YearArray(desertCodeMixMonthIndexArray2) + ...
                    amsr2LstMix2MonthArray(desertCodeMixMonthIndexArray);
                clear desertCodeMixMonthIndexArray2 desertCodeMixMonthIndexArray
                clear amsr2LstMix2MonthArray
            end
        end
        save(amsr2LstMix2YearMatPath, 'amsr2LstCnMix2YearArray', 'validDateList', 'amsr2Ref');
    end

    % ----------------------------------------------------------------------------------------------
    % 合并AMSR2 LST纯像元和混合像元, 去掉异常值.
    amsr2LstYearMatName = sprintf('AMSR2_Lst_%s_%s_CN.mat', daynight, yearStr);
    amsr2LstYearMatPath = fullfile(amsr2LstMatDir, amsr2LstYearMatName);
    if ~exist(amsr2LstYearMatPath, 'file')
        load(amsr2LstPureYearMatPath, 'amsr2LstCnPureLcYearArray');
        load(amsr2LstMix2YearMatPath, 'amsr2LstCnMix2YearArray');
        amsr2LstCnYearArray = amsr2LstCnPureLcYearArray + amsr2LstCnMix2YearArray;

        % 去掉AMSR2 LST中的异常值.

        save(amsr2LstYearMatPath, 'amsr2LstCnYearArray', 'validDateList', 'amsr2Ref');
    end

    % 输出反演的AMSR2 LST.
    fprintf('输出反演的%s年%s的AMSR2地表温度.\n', yearStr, daynight);
    load(amsr2LstYearMatPath, 'amsr2LstCnYearArray')
    for j = 1: validDateN
        amsr2LstLayer = amsr2LstCnYearArray(:, :, j);

        % AMSR2 LST数据影像.
        amsr2LstName = sprintf('AMSR2_LST_%s_%s.tif', daynight, validDateList{j});
        amsr2LstPath = fullfile(amsr2LstYearDir, amsr2LstName);
        if ~exist(amsr2LstPath, 'file')
            geotiffwrite(amsr2LstPath, amsr2LstLayer, amsr2Ref, ...
                TiffTags=struct('Compression','LZW'));
        end

        % AMSR2 LST数据缩略图.
        amsr2LstThumbName = sprintf('AMSR2_LST_%s_%s.png', daynight, validDateList{j});
        amsr2LstThumbPath = fullfile(amsr2LstYearThumbDir, amsr2LstThumbName);
        if ~exist(amsr2LstThumbPath, 'file')
            valueIndexLayer = amsr2LstLayer ~= 0;
            if sum(valueIndexLayer, 'all') > 0
                cmapLim = round(prctile(amsr2LstLayer(amsr2LstLayer ~= 0), [0.5, 99.5]));
            else
                cmapLim = [0 1];
            end
            f = figure; f.Visible = 'off'; f.Units = 'normalized'; f.Position = [0.05 0.08 0.8 0.6];
            imagesc(amsr2LstLayer, cmapLim);
            axis equal tight; ax = gca; ax.Visible = 'off';
            exportgraphics(f, amsr2LstThumbPath);
            close all
        end
    end

    continue

    % ----------------------------------------------------------------------------------------------
    % 修复AMSR2 LST影像中轨道间隙的数据空缺, 并输出每日的TIF文件.
    amsr2LstFilledMatName = sprintf('AMSR2_Lst_%s_%s_Gapfilled.mat', daynight, yearStr);
    amsr2LstFilledMatPath = fullfile(amsr2LstMatDir, amsr2LstFilledMatName);
    if ~exist(amsr2LstFilledMatPath, 'file')
        load(amsr2LstYearMatPath, 'amsr2LstCnYearArray');
        % 用于插值的时间间隔区间.
        halfInterval = 4;
        interval = halfInterval*2 + 1;

        % 给AMSR2 LST矩阵添加首尾图层, 便于时空插值.
        amsr2LstHeadArray = amsr2LstCnYearArray(:, :, end-halfInterval+1: end);
        amsr2LstTailArray = amsr2LstCnYearArray(:, :, 1 : halfInterval);
        amsr2LstYearExtArray = cat(3,amsr2LstHeadArray, amsr2LstCnYearArray, amsr2LstTailArray);

        % 使用像元前后几天的时间序列的插值来填补每日AMSR2 LST影像的轨道间隙空缺.
        amsr2LstCnFilledYearArray = zeros(size(amsr2LstCnYearArray), 'single');
        for j = 1: validDateN
            fprintf('填补%s的AMSR2 LST轨道间隙.\n', validDateList{j});

            % 准备用于插值的数据.
            fixedZonesLcLayer = fixedZonesLcArray(:, :, j);
            amsr2LstDailyLayer = amsr2LstCnYearArray(:, :, j);
            amsr2LstIntervalArray = amsr2LstYearExtArray(:, :, j: j + halfInterval*2);
            amsr2LstGapIndexList = find(amsr2LstDailyLayer == 0 & fixedZonesLcLayer ~= 128);

            % 插值.
            for k = 1: length(amsr2LstGapIndexList)
                gapIndex = amsr2LstGapIndexList(k);
                yVector = zeros(interval, 1) * nan;
                for m = 1: interval
                    amsr2LstDailyExtLayer = amsr2LstIntervalArray(:, :, m);
                    yVector(m) = amsr2LstDailyExtLayer(gapIndex);
                end
                xVector = (1: interval)';
                xVector(yVector == 0) = [];
                yVector(yVector == 0) = [];
                if isempty(xVector)
                    amsr2LstGapValue = nan;
                elseif length(xVector) == 1
                    amsr2LstGapValue = yVector;
                else
                    amsr2LstGapValue = interp1(xVector, yVector, halfInterval + 1);
                end
                amsr2LstDailyLayer(gapIndex) = amsr2LstGapValue;
            end
            amsr2LstCnFilledYearArray(:, :, j) = amsr2LstDailyLayer;
        end
        save(amsr2LstFilledMatPath, 'amsr2LstCnFilledYearArray', 'validDateList', 'amsr2Ref');
    end

    % 输出修复轨道间隙空缺后的AMSR2 LST为TIF格式.
    fprintf('输出填补过轨道间隙的%s年%s的AMSR2地表温度.\n', yearStr, daynight);
    load(amsr2LstFilledMatPath, 'amsr2LstCnFilledYearArray')
    for j = 1: validDateN
        amsr2LstName = sprintf('AMSR2_LST_%s_%s.tif', daynight, validDateList{j});
        amsr2LstPath = fullfile(amsr2LstFilledYearDir, amsr2LstName);
        if ~exist(amsr2LstPath, 'file')
            geotiffwrite(amsr2LstPath, amsr2LstCnFilledYearArray(:, :, j), amsr2Ref, ...
                TiffTags=struct('Compression','LZW'));
        end
    end
end

system('shutdown -s -t 60')
