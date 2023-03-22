%% 构建AMSR2 LST反演中的环境变量综合分类体系.
% 1. 本版本使用Landsat地表覆盖类型, 以及云覆盖区数据填充后的MODIS积雪覆盖数据作为建立分类体系的标准.
% 2. 积雪覆盖从8大综合分区改为62各基础地形植被分区反演地表温度.

%% 预设参数.
% 有数据的年份列表 (时间区间：2012/07/02-2019/12/31).
yearList = 2012: 2020;

% Landsat LC数据年份.
lcYearList = [2000, 2010, 2020];

% 重采样时, 纯像元阈值.
lcThreshold = 0.6;

% LC和分区影像代码的重分类. 主要针对水域, 冰川, 建筑, 积雪这四类LC.
%   [阶梯1_东北东部(1): 1-3, 阶梯1_华北(2): 4-5, 阶梯1_华南(3): 6-10]
%   [阶梯2_西南(4): 11-14, 阶梯2_西北东部(5): 15-25, 阶梯2_东北西部(6): 26-29]
%   [阶梯2_西北西部(7): 30-46, 阶梯3_青藏高原(8): 47-62]
% 重分类时, 使用了右开区间, 导致最后一个分区编码71排除在分类结果中, 62+1, 使其包含在重分类结果中.
regionNodes = [1, 4, 6, 11, 15, 26, 30, 47, 62+1];
regionNodesN = uint16(length(regionNodes) - 1);
snowSplitRegions = [1, 6, 8];

%% 路径.
% 根目录.
rootDir = 'F:\AMSR2_MODIS_AW_LST';
dataDir = fullfile(rootDir, 'AMSR2_LST_Retrieval\Data');
addpath(fullfile(rootDir, 'Code\Functions'));

% 创建保存环境变量综合分类体系数据的文件夹.
ccsevDir = fullfile(dataDir, 'CCSEV_Matlab');
if ~exist(ccsevDir, 'dir')
    mkdir(ccsevDir)
end

% 分区, 地表覆盖, 沙漠, 积雪, AMSR2 BT示例数据路径, 用于获取数据的属性信息和空间参考.
% 分区和沙漠长期不变, 地表覆盖每年一幅, 积雪每日一幅.
zonesPath = fullfile(dataDir, 'Zones', 'GeographicalZones_62_Merged.tif');
lcPath = fullfile(dataDir, 'LandsatLC_2_MosaicCN_TIF', 'LandsatLC_CN_Update_2000_0.01.tif');
desertPath = fullfile(dataDir, 'Landcover', 'Desert_CN_gcs.tif');
cgfSnowPath = fullfile(dataDir, 'CGF_Snow_2_CN_TIF\2012', ...
    'NIEER_CGF-MODIS_SCE_20120101_DAILY_500m_V02.tif');
amsr2BtPath = fullfile(dataDir, 'AMSR2_3_MaskCn_TIF/AMSR2_2012XXXX_TIF/AMSR2_20120703', ...
    'GW1AM2_20120703_01D_EQMA_L3SGT06HA2220220_BtH.tif');

%% 读取数据参考与属性.
% 读取掩膜后的AMSR2 BT图像的空间参考. 所有掩膜后AMSR2 BT与MODIS LST影像的空间参考都相同.
amsr2Ref = georasterinfo(amsr2BtPath).RasterReference;
amsr2RowN = amsr2Ref.RasterSize(1); amsr2ColN = amsr2Ref.RasterSize(2);

% 读取Landsat LC图像的空间参考. 沙漠图像与土地覆盖图像的空间参考相同.
lcRef = georasterinfo(lcPath).RasterReference;
lcRowN = lcRef.RasterSize(1); lcColN = lcRef.RasterSize(2);

% 读取CGF积雪图像的空间参考.
cgfSnowRef = georasterinfo(cgfSnowPath).RasterReference;
snowRowN = cgfSnowRef.RasterSize(1); snowColN = cgfSnowRef.RasterSize(2);

% 获取LC和Snow影像左上角第一个滑动窗口的边界行列号, 行列数.
% 滑动窗口的大小为AMSR2影像的像元尺寸. 滑动窗口的行列数是LC影像和AMSR2影像的分辨率倍数. 
% 边界行列号: [topRow, bottomRow, leftCol, rightCol],  行列数: [blockRowN, blockColN].
[lcBlockBdy, lcBlockSize] = getStartBlockRowCol(lcRef, amsr2Ref);
[snowBlockBdy, snowBlockSize] = getStartBlockRowCol(cgfSnowRef, amsr2Ref);

%% 合并分区, 地表覆盖, 沙漠和积雪分区影像.
% 获取地形, 植被, 裸地分区影像.
zonesLayer = uint16(readgeoraster(zonesPath));
zonesNodata = georasterinfo(zonesPath).MissingDataIndicator;  % Nodata是128.
zonesList = unique(zonesLayer);
zonesList = zonesList(zonesList ~= zonesNodata);

% 获取LC编码. [0: 其他(裸地: 100, 植被: 200), 水体: 300, 冰川: 400, 建筑: 500, 积雪: 600, 混合: 1000]
lcLayer = uint16(readgeoraster(lcPath)) * 100;
lcNodata = georasterinfo(lcPath).MissingDataIndicator * 100;
lcCodeList = unique(lcLayer);
lcCodeList = lcCodeList(lcCodeList ~= lcNodata);
majorLcCodeList = lcCodeList(1:2);  % 裸地, 植被.
minorLcCodeList = lcCodeList(3:5);  % 水体, 冰川, 建筑.
lcCodeList = [0; minorLcCodeList; 600; 1000];

% 获取沙漠影像和编码. [110, ..., 200]
desertLayer = uint16(readgeoraster(desertPath));
desertNodata = georasterinfo(desertPath).MissingDataIndicator;  % Nodata是255.
desertCodeList = unique(desertLayer);
desertCodeList = desertCodeList(desertCodeList ~= desertNodata);
desertCodeN = length(desertCodeList);

% 理论上的zonesLc编码.
[lcCodeGrid, regionGrid] = meshgrid(lcCodeList(2:4), 1: regionNodesN);
lcRegionList = reshape(lcCodeGrid + regionGrid + 90, [], 1);
snowZonesList = lcCodeList(5) + zonesList;
for i = 1: regionNodesN
    if ~ismember(i, snowSplitRegions)
        lcIndexVector = (snowZonesList >= lcCodeList(5) + regionNodes(i)) & ...
            (snowZonesList < lcCodeList(5) + regionNodes(i + 1));
        snowZonesList(lcIndexVector) = lcCodeList(5) + i + 90;
    end
end
zonesLcFullCodeList = [zonesList; desertCodeList; lcRegionList; unique(snowZonesList)];

% 分年度合并LC, 并升尺度到AMSR2数据的分辨率.
for i = 1: length(yearList)
    yearNum = yearList(i);
%     yearNum = 2013;

    % 环境变量综合分类体系数据路径.
    outLcPctArrayMatPath = fullfile(ccsevDir, ['CCSEV_', num2str(yearNum), '.mat']);
    if ~exist(outLcPctArrayMatPath, 'file')
        continue
    end
    
    fprintf('创建 %d 年的环境变量综合分类体系(CCSEV).\n', yearNum)
    % 获取离当年最近年份的地表覆盖类型影像. 移除原分类数据中的裸地和植被类型, 然后将剩余的地表覆盖类型与沙漠
    %   叠置. 当水体, 冰雪, 建筑与沙漠有重叠时, 保持他们不变.
    lcYearDiff = abs(lcYearList - yearNum);
    lcYear = lcYearList(find(lcYearDiff == min(lcYearDiff), 1));
    lcName = sprintf('LandsatLC_CN_Update_%d_0.01.tif', lcYear);
    lcPath = fullfile(dataDir, 'LandsatLC_2_MosaicCN_TIF', lcName);
    lcLayer = uint16(readgeoraster(lcPath)) * 100;
    lcLayer(ismember(lcLayer, majorLcCodeList)) = 0; % 将裸地, 植被编码替换为其他.
    desertIndex = (desertLayer ~= desertNodata) & ~ismember(lcLayer, minorLcCodeList);
    lcLayer(desertIndex) = desertLayer(desertIndex);

    % 叠置地表覆盖影像和分区影像. 结合除积雪之外的其他地表覆盖类型与分区影像. 混合像元定义为像元内没有任何地
    %   表覆盖类型的面积超过60%的像元. 合并的地表覆盖编码定义为: 分区代码 + 地表覆盖代码. 例如: 310表示在分
    %   区10的水域(300).
    
    % 计算中国范围AMSR2数据像元范围内各地表覆盖类型的面积比例, 同时获取AMSR2像元尺度的沙漠编码数据层.
    [othersPctLayer, desertPctLayer, waterPctLayer, glacierPctLayer, buildingPctLayer] = ...
        deal(zeros(amsr2RowN, amsr2ColN, 'single') * nan);
    desertCodeLayer = zeros(amsr2RowN, amsr2ColN, 'uint8');
    for m = 1: amsr2RowN
        for n = 1: amsr2ColN
            % 跳过非中国区域像元.
            if zonesLayer(m, n) == zonesNodata
                continue
            end
            % 定位每一个LC滑动窗口.
            blockTopRow = lcBlockBdy(1) + lcBlockSize(1) * (m - 1);
            blockBottomRow = lcBlockBdy(2) + lcBlockSize(1) * (m - 1);
            blockLeftCol = lcBlockBdy(3) + lcBlockSize(2) * (n - 1);
            blockRightCol = lcBlockBdy(4) + lcBlockSize(2) * (n - 1);
            if blockRightCol > lcColN || blockBottomRow > lcRowN
                continue
            end
            % 窗口计算, 排除LC中的Nodata像元.
            lcBlock = lcLayer(blockTopRow: blockBottomRow, blockLeftCol: blockRightCol);
            lcBlockRestN = prod(lcBlockSize) - sum(lcBlock == lcNodata, 'all');
            lcBlock(lcBlock == lcNodata) = [];
            othersPctLayer(m, n) = sum(lcBlock == lcCodeList(1), 'all') / lcBlockRestN;
            desertPctLayer(m, n) = sum(ismember(lcBlock, desertCodeList), 'all') / lcBlockRestN;
            waterPctLayer(m, n) = sum(lcBlock == lcCodeList(2), 'all') / lcBlockRestN;
            glacierPctLayer(m, n) = sum(lcBlock == lcCodeList(3), 'all') / lcBlockRestN;
            buildingPctLayer(m, n) = sum(lcBlock == lcCodeList(4), 'all') / lcBlockRestN;
            % 获取与AMSR2影像分辨率相同的沙漠编码数据层, 当窗口内存在多个沙漠分区, 取面积最大的沙漠分区为该
            %   像元的沙漠分区.
            if desertPctLayer(m, n) > 0
                desertPixelCount = zeros(desertCodeN, 1);
                for k = 1: desertCodeN
                    desertPixelCount(k) = sum(lcBlock == desertCodeList(k), 'all');
                end
                majorDesertIndex = find(desertPixelCount == max(desertPixelCount), 1);
                desertCodeLayer(m, n) = desertCodeList(majorDesertIndex);
            end
        end
    end

    % 获取不包括积雪覆盖的zonesLcLayer, 用于统计模型质量空间分布情况.
    zonesLcLayer = ones(amsr2RowN, amsr2ColN, 'uint16') * zonesNodata;
    for m = 1: amsr2RowN
        for n = 1: amsr2ColN
            % 跳过非中国区域像元.
            if zonesLayer(m, n) == zonesNodata
                continue
            end
            % 选取面积比例最大的LC作为升尺度后的分区地表覆盖数据层的LC.
            lcPctList = [othersPctLayer(m, n), desertPctLayer(m, n), waterPctLayer(m, n), ...
                glacierPctLayer(m, n), buildingPctLayer(m, n)]';
            maxIndex = find(lcPctList == max(lcPctList), 1);
            if maxIndex == 1 % 为其他地表覆盖类型(植被, 裸地), 编码不变.
                zonesLcLayer(m, n) = zonesLayer(m, n);
            elseif maxIndex == 2 % 沙漠.
                zonesLcLayer(m, n) = desertCodeLayer(m, n);
            elseif ismember(maxIndex, [3, 4, 5]) % 水域, 冰川, 建筑.
                zonesLcLayer(m, n) = zonesLayer(m, n) + lcCodeList(maxIndex-1);
            end
        end
    end
    for j = 1: regionNodesN
        for k = 2: length(lcCodeList) - 1
            lcIndexLayer = (zonesLcLayer >= lcCodeList(k) + regionNodes(j)) & ...
                (zonesLcLayer < lcCodeList(k) + regionNodes(j + 1));
            zonesLcLayer(lcIndexLayer) = lcCodeList(k) + j + 90;
        end
    end

    save(outLcPctArrayMatPath, 'zonesLcLayer', '-append');
    continue

    % 读取每天的MODIS积雪图像, 将其混合到分区地表覆盖图像中, 并修正土地覆盖百分比.
    cgfSnowYearDir = fullfile(dataDir, 'CGF_Snow_2_CN_TIF', num2str(yearNum));
    cgfSnowDailyList = {dir(fullfile(cgfSnowYearDir, 'NIEER*.tif')).name}';
    cgfSnowDailyN = length(cgfSnowDailyList);
    lcDateList = cell(cgfSnowDailyN, 1);
    
    [otherPctArray, desertPctArray, glacierPctArray, waterPctArray, buildingPctArray, ...
        snowPctArray] = deal(zeros(amsr2RowN, amsr2ColN, cgfSnowDailyN, 'single'));
    zonesLcArray = ones(amsr2RowN, amsr2ColN, cgfSnowDailyN, 'uint16') * zonesNodata;
    for j = 1: cgfSnowDailyN
        cgfSnowDaily = cgfSnowDailyList{j};
        fprintf('修正积雪覆盖比例: %s\n', cgfSnowDaily);

        % 获取有MODIS积雪覆盖数据的日期.
        yearDateStr = split(cgfSnowDaily, '_');
        lcDateList{j} = yearDateStr{4};

        % 获取AMSR2像元尺度的积雪比例数据层.
        cgfSnowLayer = readgeoraster(fullfile(cgfSnowYearDir, cgfSnowDaily));
        snowPctLayer = zeros(amsr2RowN, amsr2ColN, 'single');
        for m = 1: amsr2RowN
            for n = 1: amsr2ColN
                % 跳过非中国区域像元.
                if zonesLayer(m, n) == zonesNodata
                    continue
                end
                % 定位每一个snow滑动窗口.
                blockTopRow = snowBlockBdy(1) + snowBlockSize(1) * (m - 1);
                blockBottomRow = snowBlockBdy(2) + snowBlockSize(1) * (m - 1);
                blockLeftCol = snowBlockBdy(3) + snowBlockSize(2) * (n - 1);
                blockRightCol = snowBlockBdy(4) + snowBlockSize(2) * (n - 1);
                if blockRightCol > snowColN || blockBottomRow > snowRowN
                    continue
                end
                % 积雪滑动窗口计算.
                snowBlock = cgfSnowLayer(blockTopRow: blockBottomRow, blockLeftCol: blockRightCol);
                snowPixelIndex = ismember(snowBlock, [1, 2, 3]);
                snowPctLayer(m, n) = sum(snowPixelIndex, 'all') / numel(snowBlock);
            end
        end

        % 获取积雪修正后的每日地表覆盖比例数据.
        otherPctArray(:, :, j) = othersPctLayer .* (1 - snowPctLayer);
        desertPctArray(:, :, j) = desertPctLayer .* (1 - snowPctLayer);
        waterPctArray(:, :, j) = waterPctLayer .* (1 - snowPctLayer);
        glacierPctArray(:, :, j) = glacierPctLayer .* (1 - snowPctLayer);
        buildingPctArray(:, :, j) = buildingPctLayer .* (1 - snowPctLayer);
        snowPctArray(:, :, j) = snowPctLayer;

        % 如果单个窗口(1个AMSR2像元)内的某土地覆盖类型面积超过窗口面积的60%, 则将其视为纯像元, 否则视为混合
        %   像元.
        zonesLcLayer2 = ones(amsr2RowN, amsr2ColN, 'uint16') * zonesNodata;
        for m = 1: amsr2RowN
            for n = 1: amsr2ColN
                % 跳过非中国区域像元.
                if zonesLayer(m, n) == zonesNodata
                    continue
                end
                % 确定每个窗口的地表覆盖类型编码.
                lcPctList = [otherPctArray(m, n, j), desertPctArray(m, n, j), ...
                    waterPctArray(m, n, j), glacierPctArray(m, n, j), buildingPctArray(m, n, j), ...
                    snowPctArray(m, n, j)]';
                maxLcPct = max(lcPctList);
                maxIndex = find(lcPctList == maxLcPct);
                if length(maxIndex) == 1  % 最大值数量只有1个时, 单个土地覆盖类型面积比例有超过0.6的可能.
                    if maxLcPct >= lcThreshold  % 视为纯像元.
                        if maxIndex == 1 % 为其他地表覆盖类型(植被, 裸地), 编码不变.
                            zonesLcLayer2(m, n) = zonesLayer(m, n);
                        elseif maxIndex == 2 % 沙漠.
                            zonesLcLayer2(m, n) = desertCodeLayer(m, n);
                        elseif ismember(maxIndex, [3, 4, 5, 6]) % 水域, 冰川, 建筑, 积雪.
                            zonesLcLayer2(m, n) = zonesLayer(m, n) + lcCodeList(maxIndex-1);
                        end
                    else  % 混合像元.
                        zonesLcLayer2(m, n) = zonesLayer(m, n) + lcCodeList(end);
                    end
                else  % 最大值数量超过1个时, 肯定是混合像元.
                    zonesLcLayer2(m, n) = zonesLayer(m, n) + lcCodeList(end);
                end
            end
        end
        zonesLcArray(:, :, j) = zonesLcLayer2;
    end

    % LC和分区影像代码的重分类.
    %   [阶梯1_东北东部(1): 1-3, 阶梯1_华北(2): 4-5, 阶梯1_华南(3): 6-10]
    %   [阶梯2_西南(4): 11-14, 阶梯2_西北东部(5): 15-25, 阶梯2_东北西部(6): 26-29]
    %   [阶梯2_西北西部(7): 30-46, 阶梯3_青藏高原(8): 47-62]
    fprintf('中国8大地形分区: %d.\n', yearNum);
    zonesLcCodeList = unique(zonesLcArray);
    zonesLcCodeList = zonesLcCodeList(zonesLcCodeList ~= zonesNodata);
    fixedZonesLcArray = zonesLcArray;
    for j = 1: regionNodesN
        % lcCodeList: 0(裸地, 植被), 300(水体), 400(冰川), 500(建筑), 600(积雪), 1000(混合像元)
        % 裸地, 植被, 仍按原来的分区; 水体, 冰川, 建筑归类为8个大区;
        % 积雪在东北, 青藏高原按原来的分区, 在其他地区按8大分区.
        % 8大分区编码为 j+90, 以示与原地形分区编码的区别.
        for k = 2: length(lcCodeList) - 1
            if k == 5 && ismember(j, snowSplitRegions)
                continue
            end
            lcIndexVector = (zonesLcCodeList >= lcCodeList(k) + regionNodes(j)) & ...
                (zonesLcCodeList < lcCodeList(k) + regionNodes(j + 1));
            zonesLcCodeList(lcIndexVector) = lcCodeList(k) + j + 90;
            lcIndexArray = (zonesLcArray >= lcCodeList(k) + regionNodes(j)) & ...
                (zonesLcArray < lcCodeList(k) + regionNodes(j + 1));
            fixedZonesLcArray(lcIndexArray) = lcCodeList(k) + j + 90;
        end
    end
    zonesLcCodeList = unique(zonesLcCodeList);

    fprintf('保存%d年的CCESV.\n', yearNum)
    save(outLcPctArrayMatPath, 'zonesLcLayer', 'otherPctArray', 'desertPctArray', ...
        'waterPctArray', 'glacierPctArray', 'buildingPctArray', 'snowPctArray', ...
        'fixedZonesLcArray', 'desertCodeLayer', 'zonesLcCodeList', 'zonesLcFullCodeList', ...
        'lcCodeList', 'lcDateList');
end

