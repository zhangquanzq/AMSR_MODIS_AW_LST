%% 构建两极冰盖AMSRE/2 LST反演的环境变量综合分类体系(CCSEV).

%% 标识与预设参数.
% 指定研究区的标识. 1表示Antarctic, 2表示Greenland.
flg1 = 1;
% 指定微波数据类型的标识. 1表示AMSRE, 2表示AMSR2.
flg2 = 2;
% 指定分区个数的标识. 1表示Antarctic的5, 2表示Antarctic的20, 3表示Greenland的6.
flg3 = 1;

% AMSRE/2类型, 研究区.
region = {'Antarctic', 'Greenland'};
region = region{flg1};

amsrType = {'AMSRE', 'AMSR2'};
amsrType = amsrType{flg2};

% 有数据的年份月份列表(时间区间: 2003/01/01-2011/09/27, 2012/07/02-2020/12/31).
yearList = {2003: 2011, 2012: 2020};
yearList = yearList{flg2};

% 重采样时的纯像元阈值.
lcThreshold = 0.6;

% 分区个数, 分区像元Nodata.
zoneN = [5, 20, 6];
zoneN = zoneN(flg3);

zoneNodata = [15, 128, 15];
zoneNodata = zoneNodata(flg3);

% 判断研究区和分区是否匹配.
if (flg1 == 1 && flg3 ==3) || (flg1 == 2 && ismember(flg3, [1, 2]))
    error('指定的研究区和分区个数不匹配!')
end

%% 路径.
% 根目录.
rootDir = 'K:\AMSR_LST_IceSheet';
dataDir = fullfile(rootDir, 'Data');
addpath(fullfile(rootDir, 'Code\Functions'));

% 创建保存环境变量综合分类体系数据的文件夹.
ccsevDir = fullfile(dataDir, sprintf('CCSEV_%s_Matlab', region));
if ~exist(ccsevDir, 'dir')
    mkdir(ccsevDir)
end

% 分区, 地表覆盖, AMSRE/2 BT示例数据路径, 用于获取数据的属性信息和空间参考.
%   zones长期不变, MODIS地表覆盖每年一幅.
zonesPath = fullfile(dataDir, 'Zones', sprintf('%s_%s_Zones_%d.tif', amsrType, region, zoneN));
lcPath = fullfile(dataDir, sprintf('MCD12Q1_2_Mosaic%s_TIF', region), ...
    'MCD12Q1.A2001001.006_reclsfy_pt.tif');
amsrName = {sprintf('AMSRE_D25_%d0703A_v03_06H.tif', yearList(1)), ...
    sprintf('GW1AM2_%d0703_01D_EQMA_L3SGT06HA2220220_BtH.tif', yearList(1))};
amsrName = amsrName{flg2};
amsrBtPath = fullfile(dataDir, sprintf('%s_3_BT_Mask%s_TIF/%s_%dXXXX_TIF/%s_%d0703', amsrType, ...
    region, amsrType, yearList(1), amsrType, yearList(1)), amsrName);

%% 读取数据参考与属性.
% 读取掩膜后的AMSRE/2 BT图像的空间参考. 所有掩膜后AMSRE/2 BT与MODIS LST影像的空间参考都相同.
amsr2Ref = georasterinfo(amsrBtPath).RasterReference;
amsr2RowN = amsr2Ref.RasterSize(1); amsr2ColN = amsr2Ref.RasterSize(2);

% 读取MODIS LC图像的空间参考.
lcRef = georasterinfo(lcPath).RasterReference;
lcRowN = lcRef.RasterSize(1); lcColN = lcRef.RasterSize(2);

% 获取LC和Snow影像左上角第一个滑动窗口的边界行列号, 行列数.
% 滑动窗口的大小为AMSRE/2影像的像元尺寸. 滑动窗口的行列数是LC影像和AMSRE/2影像的分辨率倍数. 
% 边界行列号: [topRow, bottomRow, leftCol, rightCol],  行列数: [blockRowN, blockColN].
[lcBlockBdy, lcBlockSize, skipLocation] = getStartBlockRowCol(lcRef, amsr2Ref);
lcBlockN = prod(lcBlockSize);

%% 合并地表覆盖分区影像.
% 获取地形分区影像.  !!! Nodata, LC编码乘以10后, 是否够用 !!!
zonesLayer = uint16(readgeoraster(zonesPath));
% zonesNodata = georasterinfo(zonesPath).MissingDataIndicator;  % Nodata是128.
zonesList = unique(zonesLayer);
zonesList = zonesList(zonesList ~= zoneNodata);

% 获取LC编码. [0: 其他(裸地: 100, 植被: 200), 水体: 300, 冰川: 400, 建筑: 500, 混合: 1000]
lcLayer = uint16(readgeoraster(lcPath)) * 100;
lcNodata = uint16(georasterinfo(lcPath).MissingDataIndicator) * 100;
lcCodeList = unique(lcLayer);
lcCodeList = lcCodeList(lcCodeList ~= lcNodata);
majorLcCodeList = lcCodeList(1:2);  % 裸地, 植被.
minorLcCodeList = lcCodeList(3:5);  % 水体, 冰川, 建筑.
lcCodeList = [0; minorLcCodeList; 1000];

% 理论上的zonesLc编码.
[lcCodeGrid, zoneGrid] = meshgrid(lcCodeList(2:4), zonesList);
lcZoneList = reshape(lcCodeGrid + zoneGrid, [], 1);
zonesLcFullCodeList = [zonesList; lcZoneList];

% 分年度合并LC, 并升尺度到AMSR2数据的分辨率.
for i = 1: length(yearList)
    yearNum = yearList(i);
%     yearNum = 2013;

    % 环境变量综合分类体系数据路径.
    ccsevMatPath = fullfile(ccsevDir, sprintf('CCSEV_%d_%dzones.mat', yearNum, zoneN));
    if exist(ccsevMatPath, 'file')
        continue
    end
    
    fprintf('创建%s %d年的CCSEV.\n', region, yearNum)
    % 获取当年的地表覆盖类型影像. 移除原分类数据中的裸地和植被类型.
    lcName = sprintf('MCD12Q1.A%d001.006_reclsfy_pt.tif', yearNum);
    lcPath = fullfile(dataDir, sprintf('MCD12Q1_2_Mosaic%s_TIF', region), lcName);
    lcLayer = uint16(readgeoraster(lcPath)) * 100;
    lcLayer(ismember(lcLayer, majorLcCodeList)) = 0; % 将裸地, 植被编码替换为其他.

    % 叠置地表覆盖影像和分区影像. 除了积雪之外的其他地表覆盖类型与分区影像在此处结合. 混合像元定义为像元内没
    %   有任何地表覆盖类型的面积超过60%的像元.
    %   合并的地表覆盖编码定义为: 分区代码 + 地表覆盖代码. 例如: 301表示在分区1的水域(300).
    
    % 计算研究区范围AMSRE/2数据像元范围内各地表覆盖类型的面积比例.
    [othersPctLayer, waterPctLayer, glacierPctLayer, buildingPctLayer] = ...
        deal(zeros(amsr2RowN, amsr2ColN, 'single') * nan);
    for m = 1: amsr2RowN
        for n = 1: amsr2ColN
            % 跳过非研究区像元.
            if zonesLayer(m, n) == zoneNodata
                continue
            end
            % 定位每一个LC滑动窗口.
            blockTopRow = lcBlockBdy(1) + lcBlockSize(1) * (m - skipLocation(1));
            blockBottomRow = lcBlockBdy(2) + lcBlockSize(1) * (m - skipLocation(1));
            blockLeftCol = lcBlockBdy(3) + lcBlockSize(2) * (n - skipLocation(2));
            blockRightCol = lcBlockBdy(4) + lcBlockSize(2) * (n - skipLocation(2));

            % 跳过不在AMSRE/2影像范围的MODIS LST像元.
            if blockLeftCol <= 0 || blockTopRow <= 0
                continue
            end
            if blockRightCol > lcColN || blockBottomRow > lcRowN
                continue
            end
            % 窗口计算, 排除LC中的Nodata像元.
            lcBlock = lcLayer(blockTopRow: blockBottomRow, blockLeftCol: blockRightCol);
            lcBlockRestN = lcBlockN - sum(lcBlock == lcNodata, 'all');
            lcBlock(lcBlock == lcNodata) = [];
            othersPctLayer(m, n) = sum(lcBlock == lcCodeList(1), 'all') / lcBlockRestN;
            waterPctLayer(m, n) = sum(lcBlock == lcCodeList(2), 'all') / lcBlockRestN;
            glacierPctLayer(m, n) = sum(lcBlock == lcCodeList(3), 'all') / lcBlockRestN;
            buildingPctLayer(m, n) = sum(lcBlock == lcCodeList(4), 'all') / lcBlockRestN;
        end
    end
    
    % 若单个窗口(1 AMSRE/2像元)内的某土地覆盖类型面积超过窗口面积的60%，则将其视为纯像元, 否则视为混合像元.
    zonesLcLayer = zeros(amsr2RowN, amsr2ColN, 'uint16');
    for m = 1: amsr2RowN
        for n = 1: amsr2ColN
            % 跳过非研究区像元.
            if zonesLayer(m, n) == zoneNodata
                continue
            end

            lcPctList = [othersPctLayer(m, n), waterPctLayer(m, n), glacierPctLayer(m, n), ...
                buildingPctLayer(m, n)]';
            maxLcPct = max(lcPctList);
            maxIndex = find(lcPctList == maxLcPct);
            if length(maxIndex) == 1  % 最大值数量只有1个时, 单个土地覆盖类型面积比例有超过0.6的可能.
                if maxLcPct >= lcThreshold  % 视为纯像元.
                    if maxIndex == 1 % 为其他地表覆盖类型(植被, 裸地), 编码不变.
                        zonesLcLayer(m, n) = zonesLayer(m, n);
                    elseif ismember(maxIndex, [2, 3, 4]) % 水域, 冰川, 建筑.
                        zonesLcLayer(m, n) = zonesLayer(m, n) + lcCodeList(maxIndex);
                    end
                else  % 混合像元.
                    zonesLcLayer(m, n) = zonesLayer(m, n) + lcCodeList(end);
                end
            else  % 最大值数量超过1个时, 肯定是混合像元.
                zonesLcLayer(m, n) = zonesLayer(m, n) + lcCodeList(end);
            end
        end
    end
    zonesLcCodeList = unique(zonesLcLayer);
    zonesLcCodeList(zonesLcCodeList == 0) = [];

    fprintf('保存%s %d年的CCSEV.\n', region, yearNum)
    save(ccsevMatPath, 'zonesLcLayer', 'lcCodeList', 'othersPctLayer', 'waterPctLayer', ...
        'glacierPctLayer', 'buildingPctLayer', 'zonesLcCodeList', 'zonesLcFullCodeList');
end
