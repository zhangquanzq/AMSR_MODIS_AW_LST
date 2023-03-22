%% AMSR 地表温度反演.

%% 功能标记与预设参数.
% 指定研究区的标识. 1表示Antarctic, 2表示Greenland.
flg1 = 1;
% 指定微波数据类型的标识. 1表示AMSRE, 2表示AMSR2.
flg2 = 2;
% 指定白天和晚上的标记. 1表示白天(升轨), 2表示晚上(降轨).
flg3 = 1;
% 指定分区个数的标识. 1表示Antarctic的5, 2表示Antarctic的20, 3表示Greenland的6.
flg4 = 1;

% 研究区, AMSRE/2类型, 空间参考变量字符串.
region = {'Antarctic', 'Greenland'};
region = region{flg1};

amsrType = {'AMSRE', 'AMSR2'};
amsrType = amsrType{flg2};

amsrRefStr = sprintf('%sRef', lower(amsrType));

% AMSR2数据的通道, 极化.
channelList = {{'10', '18', '23', '36', '06', '89'}, {'10', '18', '23', '36', '06', '07', '89'}};
channelList = channelList{flg2}; if flg1 == 1 && flg2 == 1, channelList = channelList(1:end-1); end
channelN = length(channelList);
polarizeN = length(["H", "V"]);

% 回归模型中的变量个数(12/14个通道 + 2个二次项 + 1常数项).
variablesN = channelN * polarizeN + 3;

% 数据年份列表(AMSR-E时间区间2003-2011, AMSR2时间区间2012-2020).
yearList = {2003: 2011, 2012: 2020};
yearList = yearList{flg2};

% 各月份的名称.
monthList = {'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'};
monthN = length(monthList);

% 昼夜标记.
daynight = {'Day', 'Night'};
daynight = daynight{flg3};

% AMSR2 BT影像中的NoData.
amsr2Nodata = [65534, 65535, 0];

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
dataDir = fullfile(rootDir, 'Data');
addpath(fullfile(rootDir, 'Code\Functions'));

% 输入数据路径.
amsrPrjMatDir = fullfile(dataDir, sprintf('%s_2_BT_Prj%s_Matlab', amsrType, region));
amsrPrjTifDir = fullfile(dataDir, sprintf('%s_2_BT_Prj%s_TIF', amsrType, region));
ccsevMatDir = fullfile(dataDir, sprintf('CCSEV_%s_Matlab', region));
regressMatDir = fullfile(dataDir, sprintf('Regression_%s_Matlab', region));

% 输出反演的AMSRE/2 LST路径.
amsrLstMatDir = fullfile(dataDir, sprintf('%s_4_LST_%s_Matlab_%dzones', amsrType, region, zoneN));
if ~exist(amsrLstMatDir, 'dir')
    mkdir(amsrLstMatDir)
end
amsrLstDir = fullfile(dataDir, sprintf('%s_4_LST_%s_TIF_%dzones', amsrType, region, zoneN));
if ~exist(amsrLstDir, 'dir')
    mkdir(amsrLstDir)
end

% AMSRE/2 BT 示例数据, 用于获取坐标系统标记.
amsrBtExampleName = {sprintf('%s_D25_%d0101A_v03_06H.tif', amsrType, yearList(end)), ...
    sprintf('GW1AM2_%d0101_01D_EQMA_L3SGT06HA2220220_BtH.tif', yearList(end))};
amsrBtExamplePath = fullfile(amsrPrjTifDir, sprintf('%s_%dXXXX', amsrType, yearList(end)), ...
    sprintf('%s_%d0101', amsrType, yearList(end)), amsrBtExampleName{flg2});
geoTag = geotiffinfo(amsrBtExamplePath).GeoTIFFTags.GeoKeyDirectoryTag;

%% 回归和输出.
% 分年度反演AMSRE/2 LST.
for i = 1: length(yearList)
%     yearStr = num2str(yearList(i));
    yearStr = '2020';
    rydStr = sprintf('%s_%s_%s', region, yearStr, daynight);

    % 输出AMSRE/2 LST数据的路径.
    amsrLstYearDir = fullfile(amsrLstDir, sprintf('%s_LST_%sXXXX_TIF', amsrType, yearStr));
    if ~exist(amsrLstYearDir, 'dir')
        mkdir(amsrLstYearDir);
    end

    amsrLstYearThumbDir = fullfile(amsrLstDir, sprintf('%s_LST_%sXXXX_Thumb', amsrType, yearStr));
    if ~exist(amsrLstYearThumbDir, 'dir')
        mkdir(amsrLstYearThumbDir);
    end

%     amsrLstFillYearDir = fullfile(amsrLstDir, sprintf('%s_LST_%sXXXX_Fill', amsrType, yearStr));
%     if ~exist(amsrLstFillYearDir, 'dir')
%         mkdir(amsrLstFillYearDir);
%     end

    % 从Mat文件中读取研究区原始AMSRE/2 BT数据.
    amsrPrjMatName = sprintf('%s_BT_%s.mat', amsrType, rydStr);
    amsrPrjMatPath = fullfile(amsrPrjMatDir, amsrPrjMatName);
    load(amsrPrjMatPath, 'amsr*YearArray', amsrRefStr, 'dateYearList');

    % 从Mat文件中读取CCSEV.
    ccsevMatPath = fullfile(ccsevMatDir, sprintf('CCSEV_%s_%dzones.mat', yearStr, zoneN));
    load(ccsevMatPath, 'zonesLcCodeList', 'lcCodeList', 'zonesLcLayer');

    % 从Mat文件中读取纯像元分区的回归系数.
    regressPureMatName = sprintf('Regression_Pure_%s_%dzones.mat', rydStr, zoneN);
    regressPureMatPath = fullfile(regressMatDir, regressPureMatName);
    load(regressPureMatPath, 'fixedCoefficientMonthArray', 'pureLcCodeList');

    % 获取当前年份有效数据日期的天数及其所属月份.
    dateN = length(dateYearList);
    yearMonthList = datetime(dateYearList , InputFormat='yyyyMMdd').Month;

    % 获取ASMRE/2 LST数据的行列数, 扩展分区编码数据层数为当前年份有数据的天数.
    [amsrRowN, amsrColN] = size(zonesLcLayer);
    zonesLcArray = repmat(zonesLcLayer, 1, 1, dateN);

    % ----------------------------------------------------------------------------------------------
    % 使用优化后的模型反演纯像元分区的AMSRE/2 LST.
    amsrLstPureYearMatName = sprintf('%s_LST_%s_%s_Pure.mat', amsrType, daynight, yearStr);
    amsrLstPureYearMatPath = fullfile(amsrLstMatDir, amsrLstPureYearMatName);
    if ~exist(amsrLstPureYearMatPath, 'file')
        amsrLstPureLcYearArray = zeros(amsrRowN, amsrColN, dateN, 'single');
        for j = 1: length(pureLcCodeList)
            zonesLcCode = pureLcCodeList(j);
            fprintf('反演%s年分区%d %s的%s LST.\n', yearStr, zonesLcCode, daynight, amsrType);

            % 从年度矩阵中获取纯像元分区AMSRE/2 BT的影像.
            zonesLcIndexArray = (zonesLcArray == zonesLcCode);
            [amsrHZoneYearCell, amsrVZoneYearCell] = deal(cell(channelN, 1));
            for k = 1: channelN
                amsrHYearArrayStr = sprintf('%sH%sYearArray', lower(amsrType), channelList{k});
                amsrVYearArrayStr = sprintf('%sV%sYearArray', lower(amsrType), channelList{k});
                amsrHYearArray = single(eval(amsrHYearArrayStr));
                amsrVYearArray = single(eval(amsrVYearArrayStr));
                amsrHYearArray(ismember(amsrHYearArray, amsr2Nodata)) = nan;
                amsrVYearArray(ismember(amsrVYearArray, amsr2Nodata)) = nan;
                amsrHZoneYearCell{k} = setnan(amsrHYearArray / 100, ~zonesLcIndexArray);
                amsrVZoneYearCell{k} = setnan(amsrVYearArray / 100, ~zonesLcIndexArray);
            end
            clear amsrHYearArray amsrVYearArray

            % 反演纯像元分区每个月的AMSRE/2 LST.
            for k = 1: monthN
                monthIndex = (yearMonthList == k);

                % 提取当前分区各通道的AMSRE/2 BT和二次项.
                [amsrHZoneMonthCell, amsrVZoneMonthCell] = deal(cell(channelN, 1));
                for n = 1: channelN
                    amsrHZoneMonthCell{n} = amsrHZoneYearCell{n}(:, :, monthIndex);
                    amsrVZoneMonthCell{n} = amsrVZoneYearCell{n}(:, :, monthIndex);
                end
                amsrQdZoneMonthCell = {...
                    (amsrVZoneMonthCell{4} - amsrVZoneMonthCell{3}) .^ 2, ...
                    (amsrVZoneMonthCell{4} - amsrVZoneMonthCell{2}) .^ 2};

                % 反演AMSRE/2 LST.
                % AMSRE系数: [10H 18H 23H 36H 06H 89H 10V 18V 23V 36V 06V 89V 常数].
                % AMSR2系数: [10H 18H 23H 36H 06H 07H 89H 10V 18V 23V 36V 06V 07V 89V 常数].
                zonsLcIndex = (pureLcCodeList == zonesLcCode);
                fixedP = single(fixedCoefficientMonthArray(zonsLcIndex, :, k));
                amsrLstZoneMonthArray = repmat(fixedP(end), amsrRowN, amsrColN, sum(monthIndex));
                for n = 1: channelN
                    amsrLstZoneMonthArray = amsrLstZoneMonthArray + ...
                        amsrHZoneMonthCell{n} * fixedP(n) + ...
                        amsrVZoneMonthCell{n} * fixedP(n + channelN);
                end
                amsrLstZoneMonthArray = amsrLstZoneMonthArray + ...
                    amsrQdZoneMonthCell{:, 1} * fixedP(n*2 + 1) + ...
                    amsrQdZoneMonthCell{:, 2} * fixedP(n*2 + 2);
                amsrLstZoneMonthArray(isnan(amsrLstZoneMonthArray)) = 0;

                % 将反演的当前纯像元分区和月份的AMSRE/2 LST保存到年度矩阵中.
                zonesLcMonthIndexArray = zonesLcIndexArray(:, :, monthIndex);
                zonesLcMonthIndexArray2 = false(amsrRowN, amsrColN, dateN);
                zonesLcMonthIndexArray2(:, :, monthIndex) = zonesLcMonthIndexArray;
                amsrLstPureLcYearArray(zonesLcMonthIndexArray2) = ...
                    amsrLstZoneMonthArray(zonesLcMonthIndexArray);
                clear zonesLcMonthIndexArray zonesLcMonthIndexArray2 amsrLstZoneMonthArray
            end
        end
        save(amsrLstPureYearMatPath, 'amsrLstPureLcYearArray', 'dateYearList', amsrRefStr);
    end

    % ----------------------------------------------------------------------------------------------
    % 反演混合像元的AMSRE/2 LST.
    % 在循环中, 组合包含水体, 冰川, 建筑和其他地表覆盖类型的不完全LST.

    % 遍历每个普通分区, 并寻找其中的混合像元, 然后通过对水域, 冰川, 建筑和其他地表覆盖类型的加权平均计算不完
    %   全组分温度.
    amsrLstMixYearMatName = sprintf('%s_LST_%s_%s_Mix.mat', amsrType, daynight, yearStr);
    amsrLstMixYearMatPath = fullfile(amsrLstMatDir, amsrLstMixYearMatName);
    if ~exist(amsrLstMixYearMatPath, 'file')
        % 读取LC的面积比例矩阵.
        load(ccsevMatPath, 'othersPctLayer', 'buildingPctLayer', 'glacierPctLayer', ...
            'waterPctLayer');

        % 扩展LC面积比例数据层数为有效日期数.
        otherPctArray = repmat(othersPctLayer, 1, 1, dateN);
        buildingPctArray = repmat(buildingPctLayer, 1, 1, dateN);
        glacierPctArray = repmat(glacierPctLayer, 1, 1, dateN);
        waterPctArray = repmat(waterPctLayer, 1, 1, dateN);

        % 反演不同混合分区的AMSRE/2 LST.
        amsrLstMixYearArray = zeros(amsrRowN, amsrColN, dateN, 'single');
        mixZonesCodeList = zonesLcCodeList(zonesLcCodeList > lcCodeList(end));  % > 1000
        for j = 1: length(mixZonesCodeList)
            % 寻找年度分区矩阵内每个分区的混合像元. 例: 1 + 1000 表示分区1中的混合像元.
            mixZoneCode = mixZonesCodeList(j);
            zoneCode = mixZoneCode - lcCodeList(end);
            fprintf('反演%s年混合像元分区%d %s的%s LST.\n', yearStr, mixZoneCode, daynight, amsrType);

            % 跳过没有混合像元的分区.
            mixedPixelIndexArray = (zonesLcArray == mixZoneCode);
            if sum(mixedPixelIndexArray, 'all') == 0
                continue
            end

            % 获取当前分区全年所有通道混合像元的AMSRE/2 BT影像.
            [amsrHMixPixelYearArray, amsrVMixPixelYearArray] = deal(cell(1, channelN));
            for k = 1: channelN
                amsrHYearArrayStr = sprintf('%sH%sYearArray', lower(amsrType), channelList{k});
                amsrVYearArrayStr = sprintf('%sV%sYearArray', lower(amsrType), channelList{k});
                amsrHYearArray = single(eval(amsrHYearArrayStr));
                amsrVYearArray = single(eval(amsrVYearArrayStr));
                amsrHYearArray(ismember(amsrHYearArray, amsr2Nodata)) = nan;
                amsrVYearArray(ismember(amsrVYearArray, amsr2Nodata)) = nan;
                amsrHMixPixelYearArray{k} = setnan(amsrHYearArray / 100, ~mixedPixelIndexArray);
                amsrVMixPixelYearArray{k} = setnan(amsrVYearArray / 100, ~mixedPixelIndexArray);
            end
            clear amsrHYearArray amsrVYearArray

            % 获取当前分区中全年每个地表覆盖类型(水域, 冰川, 建筑, 其他)的面积比例.
            otherPctMixPixelYearArray = setnan(otherPctArray, ~mixedPixelIndexArray);
            waterPctMixPixelYearArray = setnan(waterPctArray, ~mixedPixelIndexArray);
            glacierPctMixPixelYearArray = setnan(glacierPctArray, ~mixedPixelIndexArray);
            buildingPctMixPixelYearArray = setnan(buildingPctArray, ~mixedPixelIndexArray);

            % 使用优化后的月尺度模型计算混合像元的AMSRE/2 LST.
            for k = 1: monthN
                monthIndex = (yearMonthList == k);

                % 跳过没有混合像元的月份.
                mixedPixelMonthIndexArray = mixedPixelIndexArray(:, :, monthIndex);
                if sum(mixedPixelMonthIndexArray, 'all') == 0
                    continue
                end

                % 获取当前月份的AMSRE/2 BT数据和二次项矩阵.
                [amsrHMixPixelMonthCell, amsrVMixPixelMonthCell] = deal(cell(1, channelN));
                for m = 1: channelN
                    amsrHMixPixelMonthCell{m} = amsrHMixPixelYearArray{m}(:, :, monthIndex);
                    amsrVMixPixelMonthCell{m} = amsrVMixPixelYearArray{m}(:, :, monthIndex);
                end
                amsrQdMixPixelMonthCell = {
                    (amsrVMixPixelMonthCell{4} - amsrVMixPixelMonthCell{3}) .^ 2, ...
                    (amsrVMixPixelMonthCell{4} - amsrVMixPixelMonthCell{2}) .^ 2};

                % 获取当前月份和分区中每个地表覆盖类型(水域, 冰川, 建筑, 其他)的面积比例.
                otherPctMixPixelMonthArray = otherPctMixPixelYearArray(:, :, monthIndex);
                waterPctMixPixelMonthArray = waterPctMixPixelYearArray(:, :, monthIndex);
                glacierPctMixPixelMonthArray = glacierPctMixPixelYearArray(:, :, monthIndex);
                buildingPctMixPixelMonthArray = buildingPctMixPixelYearArray(:, :, monthIndex);

                % 如果该分区包含特定的地表覆盖, 获取其回归系数, 否则将其系数设为0.
                waterCodeIndex = (pureLcCodeList == (lcCodeList(2) + zoneCode));
                glacierCodeIndex = (pureLcCodeList == (lcCodeList(3) + zoneCode));
                buildingCodeIndex = (pureLcCodeList == (lcCodeList(4) + zoneCode));

                % AMSRE系数: [10H 18H 23H 36H 06H 89H 10V 18V 23V 36V 06V 89V 常数].
                % AMSR2系数: [10H 18H 23H 36H 06H 07H 89H 10V 18V 23V 36V 06V 07V 89V 常数].
                pOtherMonth = fixedCoefficientMonthArray(j, :, k);
                [pWaterMonth, pGlacierMonth, pBuildingMonth] = deal(zeros(1, variablesN));
                if sum(waterCodeIndex) == 1
                    pWaterMonth = fixedCoefficientMonthArray(waterCodeIndex, :, k);
                end
                if sum(glacierCodeIndex) == 1
                    pGlacierMonth = fixedCoefficientMonthArray(glacierCodeIndex, :, k);
                end
                if sum(buildingCodeIndex) == 1
                    pBuildingMonth = fixedCoefficientMonthArray(buildingCodeIndex, :, k);
                end

                % 回归.
                coeffArray = zeros(amsrRowN, amsrColN, sum(monthIndex), variablesN, 'single');
                for m = 1: variablesN
                    coeffArray(:, :, :, m) = ...
                        pOtherMonth(m) .* otherPctMixPixelMonthArray + ...
                        pWaterMonth(m) .* waterPctMixPixelMonthArray + ...
                        pGlacierMonth(m) .* glacierPctMixPixelMonthArray + ...
                        pBuildingMonth(m) .* buildingPctMixPixelMonthArray;
                end
                amsrPartialLstMixedLcMonthArray = coeffArray(:, :, :, end);
                for m = 1: channelN
                    amsrPartialLstMixedLcMonthArray = amsrPartialLstMixedLcMonthArray + ...
                        amsrHMixPixelMonthCell{m} .* coeffArray(:, :, :, m) + ...
                        amsrVMixPixelMonthCell{m} .* coeffArray(:, :, :, m+channelN);
                end
                amsrPartialLstMixedLcMonthArray = amsrPartialLstMixedLcMonthArray + ...
                    amsrQdMixPixelMonthCell{:, 1} .* coeffArray(:, :, :, m*2 + 1) + ...
                    amsrQdMixPixelMonthCell{:, 2} .* coeffArray(:, :, :, m*2 + 2);
                amsrPartialLstMixedLcMonthArray(isnan(amsrPartialLstMixedLcMonthArray)) = 0;

                mixPixelMonthIndexArray2 = false(amsrRowN, amsrColN, dateN);
                mixPixelMonthIndexArray2(:, :, monthIndex) = mixedPixelMonthIndexArray;
                amsrLstMixYearArray(mixPixelMonthIndexArray2) = ...
                    amsrPartialLstMixedLcMonthArray(mixedPixelMonthIndexArray);
                clear coeffArray mixPixelMonthIndexArray2 mixedPixelMonthIndexArray;
                clear amsrPartialLstMixedLcMonthArray;
            end
        end
        save(amsrLstMixYearMatPath, 'amsrLstMixYearArray', 'dateYearList', amsrRefStr)
    end

    % ----------------------------------------------------------------------------------------------
    % 合并AMSRE/2 LST纯像元和混合像元, 去掉异常值.
    amsrLstYearMatName = sprintf('%s_LST_%s_%s_CN.mat', amsrType, daynight, yearStr);
    amsrLstYearMatPath = fullfile(amsrLstMatDir, amsrLstYearMatName);
    if ~exist(amsrLstYearMatPath, 'file')
        load(amsrLstPureYearMatPath, 'amsrLstPureLcYearArray');
        load(amsrLstMixYearMatPath, 'amsrLstMixYearArray');
        amsrLstYearArray = amsrLstPureLcYearArray + amsrLstMixYearArray;

        % 去掉AMSRE/2 LST中的异常值.

        save(amsrLstYearMatPath, 'amsrLstYearArray', 'dateYearList', amsrRefStr);
    end

    % 输出反演的AMSRE/2 LST.
    load(amsrLstYearMatPath, 'amsrLstYearArray')
    for j = 1: dateN
        amsrLstLayer = amsrLstYearArray(:, :, j);

        % AMSRE/2 LST数据影像.
        amsrLstName = sprintf('%s_LST_%s_%s.tif', amsrType, daynight, dateYearList{j});
        amsrLstPath = fullfile(amsrLstYearDir, amsrLstName);
        if ~exist(amsrLstPath, 'file')
            fprintf('输出反演的%s %s的%s LST.\n', dateYearList{j}, daynight, amsrType);
            geotiffwrite(amsrLstPath, amsrLstLayer, eval(amsrRefStr), ...
                TiffTags=struct('Compression','LZW'), GeoKeyDirectoryTag=geoTag);
        end

        % AMSRE/2 LST数据缩略图.
        amsrLstThumbName = sprintf('%s_LST_%s_%s.png', amsrType, daynight, dateYearList{j});
        amsrLstThumbPath = fullfile(amsrLstYearThumbDir, amsrLstThumbName);
        if ~exist(amsrLstThumbPath, 'file')
            valueIndexLayer = amsrLstLayer ~= 0;
            if sum(valueIndexLayer, 'all') > 0
                cmapLim = round(prctile(amsrLstLayer(amsrLstLayer ~= 0), [0.5, 99.5]));
            else
                cmapLim = [0 1];
            end
            f = figure; f.Visible = 'off'; f.Units = 'normalized'; f.Position = [0.05 0.08 0.8 0.6];
            imagesc(amsrLstLayer, cmapLim);
            axis equal tight; ax = gca; ax.Visible = 'off';
            exportgraphics(f, amsrLstThumbPath);
            close all
        end
    end

    continue

    % ----------------------------------------------------------------------------------------------
    % 修复AMSRE/2 LST影像中轨道间隙的数据空缺, 并输出每日的TIF文件.
    amsrLstFillMatName = sprintf('AMSR2_LST_%s_%s_Gapfilled.mat', daynight, yearStr);
    amsrLstFillMatPath = fullfile(amsrLstMatDir, amsrLstFillMatName);
    if ~exist(amsrLstFillMatPath, 'file')
        load(amsrLstYearMatPath, 'amsrLstYearArray');
        % 用于插值的时间间隔区间.
        halfInterval = 4;
        interval = halfInterval*2 + 1;

        % 给AMSR2 LST矩阵添加首尾图层, 便于时空插值.
        amsrLstHeadArray = amsrLstYearArray(:, :, end-halfInterval+1: end);
        amsrLstTailArray = amsrLstYearArray(:, :, 1: halfInterval);
        amsrLstYearExtArray = cat(3, amsrLstHeadArray, amsrLstYearArray, amsrLstTailArray);

        % 使用像元前后几天的时间序列的插值来填补每日AMSRE/2 LST影像的轨道间隙空缺.
        amsrLstFillYearArray = zeros(size(amsrLstYearArray), 'single');
        for j = 1: dateN
            fprintf('填补%s的%s LST轨道间隙.\n', dateYearList{j}, amsrType);

            % 准备用于插值的数据.
            fixedZonesLcLayer = zonesLcArray(:, :, j);
            amsrLstDailyLayer = amsrLstYearArray(:, :, j);
            amsrLstIntervalArray = amsrLstYearExtArray(:, :, j: j + halfInterval*2);
            amsrLstGapIndexList = find(amsrLstDailyLayer == 0 & fixedZonesLcLayer ~= 128);

            % 插值.
            for k = 1: length(amsrLstGapIndexList)
                gapIndex = amsrLstGapIndexList(k);
                yVector = zeros(interval, 1) * nan;
                for m = 1: interval
                    amsrLstDailyExtLayer = amsrLstIntervalArray(:, :, m);
                    yVector(m) = amsrLstDailyExtLayer(gapIndex);
                end
                xVector = (1: interval)';
                xVector(yVector == 0) = [];
                yVector(yVector == 0) = [];
                if isempty(xVector)
                    amsrLstGapValue = nan;
                elseif length(xVector) == 1
                    amsrLstGapValue = yVector;
                else
                    amsrLstGapValue = interp1(xVector, yVector, halfInterval + 1);
                end
                amsrLstDailyLayer(gapIndex) = amsrLstGapValue;
            end
            amsrLstFillYearArray(:, :, j) = amsrLstDailyLayer;
        end
        save(amsrLstFillMatPath, 'amsrLstFillYearArray', 'dateYearList', amsrRefStr);
    end

    % 输出修复轨道间隙空缺后的AMSRE/2 LST为TIF格式.
    fprintf('输出填补过轨道间隙的%s年%s的%s地表温度.\n', yearStr, daynight, amsrType);
    load(amsrLstFillMatPath, 'amsrLstFillYearArray')
    for j = 1: dateN
        amsrLstName = sprintf('%s_LST_%s_%s.tif', amsrType, daynight, dateYearList{j});
        amsrLstPath = fullfile(amsrLstFillYearDir, amsrLstName);
        if ~exist(amsrLstPath, 'file')
            geotiffwrite(amsrLstPath, amsrLstFillYearArray(:, :, j), eval(amsrRefStr), ...
                TiffTags=struct('Compression','LZW'), GeoKeyDirectoryTag=geoTag);
        end
    end
end

system('shutdown -s -t 60')
