%%
%{
2019/01/10 说明
  本实验将modispixelpct阈值设置为0.6.

2022/09/03 说明
  以>30000(31000)和非冰川, 湖泊区域<18000为阈值, 去除AMSR2 BT中的异常值.
  Step3_AMSR2_LST_Model_Construction.m 中用到此阈值去除AMSR2 BT中异常值.
%}

%% 标识与预设参数.
% 指定微波数据类型的标识. 1表示AMSRE, 2表示AMSR2.
flg1 = 1;
% 指定研究区的标识. 1表示Antarctic, 2表示Greenland.
flg2 = 1;
% 指定轨道的标识. 1表示升轨(A, 白天), 2表示降轨(D, 晚上).
flg3 = 2;

% AMSRE/2类型, 研究区, 昼夜标记.
amsrType = {'AMSRE', 'AMSR2'};
amsrType = amsrType{flg1};

region = {'Antarctic', 'Greenland'};
region = region{flg2};

dayNight = {'Day', 'Night'};
dayNight = dayNight{flg3};

% AMSRE/2通道, 极化, 年份列表.
channelList = {{'10', '18', '23', '36', '06', '89'}, {'10', '18', '23', '36', '06', '07', '89'}};
channelList = channelList{flg1};
channelN = length(channelList);

polarList = {'H', 'V'};
polarN = length(polarList);
cpN = channelN * polarN;

yearList = {2003: 2011, 2012: 2020};
yearList = yearList{flg1};

% AMSR/2 Mat文件里的变量名, 以及TIF文件名中的日期字符串位置.
amsrRefVar = {'amsreRef', 'amsr2Ref'};
amsrRefVar = amsrRefVar{flg1};

amsrPathMatrixVar = {'amsrePathMatrix', 'amsr2PathMatrix'};
amsrPathMatrixVar = amsrPathMatrixVar{flg1};

dateIndex = {11:18, 8:15};
dateIndex = dateIndex{flg1};

% 标记MODIS LST像元质量的QC值.
% QC值含义见文件: 'J:\AMSR2_MODIS_AW_LST\AMSR2_LST_Retrieval\Doc\MYD11A1_QC.csv'
qcBad1 = [2, 3]; qcBad2 = 192;

% 筛选有效AMSRE/2像元的阈值, 定义为一个AMSRE/2像元范围内的MODIS像元的百分比.
modisPixelPct = 0.6;

% AMSRE/2 BT影像中的NoData.
amsrNodata = {[0, 65536], [65534, 65535]};
amsrNodata = amsrNodata{flg1};

% MODIS LST影像中的Nodata.
modisLstNodata = [65535, 0];

% MODIS LC湖泊, 冰川编号.
waterLcCode = [3, 4];

%% 路径.
rootDir = 'F:\AMSR_LST_IceSheet\';
dataDir = fullfile(rootDir, 'Data');
addpath(fullfile(rootDir, 'Code', 'Functions'));

% 输入数据AMSRE/2 BT, MODIS LST文件夹路径.
amsrMatDir = fullfile(dataDir, sprintf('%s_2_BT_Prj%s_Matlab', amsrType, region));
modisLstPrjDir = fullfile(dataDir, sprintf('MYD11A1_2_Prj%s_TIF', region));
modisLcDir = fullfile(dataDir, sprintf('MCD12Q1_2_Mosaic%s_TIF', region));

% 输出数据路径, 同日期经过空间交集掩膜的AMSRE/2 BT和MODIS LST数据.
amsrMaskTifDir = fullfile(dataDir, sprintf('%s_3_BT_Mask%s_TIF', amsrType, region));
if ~exist(amsrMaskTifDir, 'dir')
    mkdir(amsrMaskTifDir)
end

amsrMaskMatDir = fullfile(dataDir, sprintf('%s_3_BT_Mask%s_Matlab', amsrType, region));
if ~exist(amsrMaskMatDir, 'dir')
    mkdir(amsrMaskMatDir)
end

modisLstMaskTifDir = fullfile(dataDir, sprintf('MYD11A1_3_Mask%s_TIF', region));
if ~exist(modisLstMaskTifDir, 'dir')
    mkdir(modisLstMaskTifDir)
end

modisLstMaskMatDir = fullfile(dataDir, sprintf('MYD11A1_3_Mask%s_Matlab', region));
if ~exist(modisLstMaskMatDir, 'dir')
    mkdir(modisLstMaskMatDir)
end

%% 读取数据参考和属性信息.
% 获取样例AMSRE/2 BT影像的空间参考信息.
amsrBtMatName = sprintf('%s_BT_%s_%d_%s.mat', amsrType, region, yearList(1), dayNight);
amsrBtMatPath = fullfile(dataDir, sprintf('%s_2_BT_Prj%s_Matlab', amsrType, region), amsrBtMatName);
load(amsrBtMatPath, amsrRefVar); amsrRef = eval(sprintf('%s', amsrRefVar));
amsrRowN = amsrRef.RasterSize(1); amsrColN = amsrRef.RasterSize(2);

% 获取样例MODIS LST影像的空间参考信息.
modisLstName = sprintf('MYD11A1.A%d185.LST_Day.tif', yearList(1));
modisLstPath = fullfile(modisLstPrjDir, sprintf('MYD11A1_%dXXX_TIF', yearList(1)), modisLstName);
modisLstRef = geotiffinfo(modisLstPath).SpatialRef;
geoTag = geotiffinfo(modisLstPath).GeoTIFFTags.GeoKeyDirectoryTag;

% 从MODIS影像中获取与研究区AMSRE/2影像边界对齐的起始和结束的行号和列号.
[modisLStartRow, modisLEndRow, modisLStartCol, modisLEndCol] = getBdyRowCol(modisLstRef, amsrRef);

% 裁剪后研究区范围内的MODIS LST行列数.
modisLstColN = modisLEndCol - modisLStartCol + 1;
modisLstRowN = modisLEndRow - modisLStartRow + 1;

% 获取研究区的范围栅格.
regionExtentName = sprintf('%s_%s_Extent.tif', amsrType, region);
regionExtentPath = fullfile(dataDir, 'Projection_Example', regionExtentName);
regionMaskLayer = readgeoraster(regionExtentPath) == 1;

% 获取用于升尺度MODIS像元的移动滑块的行列数.
% 获取与研究区范围AMSRE/2影像左上角边界对齐的滑动块的左右边界列号与上下边界行号.
% 边界行列号: [topRow, bottomRow, leftCol, rightCol],  行列数: [blockRowN, blockColN].
[lstBlockBdy, lstBlockSize, skipLocation] = getStartBlockRowCol(modisLstRef, amsrRef);
lstBlockN = prod(lstBlockSize);

%% 读取各年份的数据, 筛选相同日期数据, 升尺度MODIS LST, 并输出空间交集数据.
% 分年份读取MODIS LST和AMSRE/2 BT的文件.
for i = 1: length(yearList)
    yearStr = num2str(yearList(i));
%     yearStr = '2003';
    disp(['数据筛选年份:', yearStr]);

    % 输出的掩膜后年度MODIS LST和AMSRE/2 BT的矩阵的mat文件路径.
    modisLstMaskMatName = sprintf('MYD11A1_Mask%s_%s_%s.mat', region, yearStr, dayNight);
    modisLstMaskMatPath = fullfile(modisLstMaskMatDir, modisLstMaskMatName);
    amsrMaskMatName = sprintf('%s_BT_Mask%s_%s_%s.mat', amsrType, region, yearStr, dayNight);
    amsrMaskMatPath = fullfile(amsrMaskMatDir, amsrMaskMatName);
    modisMatExist = exist(modisLstMaskMatPath, 'file');
    amsrMatExist = exist(amsrMaskMatPath, 'file');

    % 读取MODIS LST, QC的文件列表.
    modisLstYearFolder = sprintf('MYD11A1_%sXXX_TIF', yearStr);
    modisLstPrjYearDir = fullfile(modisLstPrjDir, modisLstYearFolder);

    modisLstYearList = dir(fullfile(modisLstPrjYearDir, sprintf('*LST_%s.tif', dayNight)));
    modisLstYearList = {modisLstYearList.name}';
    modisLstYearN = length(modisLstYearList);

    modisQcYearList = dir(fullfile(modisLstPrjYearDir, sprintf('*QC_%s.tif', dayNight)));
    modisQcYearList = {modisQcYearList.name}';
    modisQcYearN = length(modisQcYearList);

    % 判断MODIS LST, QC文件是否一一对应.
    if ~isequal(modisLstYearN, modisQcYearN)
        error('%s年的MODIS LST, QC数据不匹配, 请检查!', yearStr)
    end
    [modisLstDateList, modisQcDateList] = deal(cell(modisLstYearN, 1));
    for j = 1: modisLstYearN
        modisLstDate = split(modisLstYearList{j}, '.');
        modisLstDateList{j} = yday2ymd(modisLstDate{2}(2:end));

        modisQcDate = split(modisQcYearList{j}, '.');
        modisQcDateList{j} = yday2ymd(modisQcDate{2}(2:end));
    end
    if  ~isequal(modisLstDateList, modisQcDateList)
        error('%s年的MODIS LST与QC数据不匹配, 请检查!', yearStr)
    end

    % 从mat文件中读取AMSRE/2 BT数据各通道文件列表, 数据本身及空间参考.
    % 变量 amsr2PathMatrix 为存储AMSR2 BT TIF文件路径字符串的cell矩阵. 矩阵的行表示不同的日期, 列表示不同
    %   的极化通道. 第一行为表头, 存储2极化和7通道的名称: ['10H', '10V', '18H', '18V', '23H', '23V',
    %   '36H', '36V', '6H', '6V', '7H', '7V', '89H', '89V']. 其他行依次存储这些极化通道对应的不同日期的
    %   AMSR2 BT TIF文件的路径字符串.
    amsrMatName = sprintf('%s_BT_%s_%s_%s.mat', amsrType, region, yearStr, dayNight);
    amsrMatPath = fullfile(amsrMatDir, amsrMatName);
    load(amsrMatPath, amsrPathMatrixVar, 'amsr*YearArray')
    amsrPathMatrix = eval(sprintf('%s', amsrPathMatrixVar));
    amsrPathMatrix = replace(amsrPathMatrix, 'F:\AMSR_LST_IceSheet\Data\', dataDir);

    % 输入AMSRE/2 BT数据日期列表.
    amsrPathMatrix = amsrPathMatrix(2:end, :);
    [~, amsrNameMatrix, ~] = fileparts(amsrPathMatrix);
    amsrDateList = cellfun(@(a) a(dateIndex), amsrNameMatrix(:, 1), UniformOutput=false);

    % AMSRE/2 BT数据异常值检测.
    % A) 以MODIS LC 5分区为基准, 去除每个分区中AMSRE/2 BT离差超过3倍标准差的值(暂时不用, 用break语句跳过).
    % !!! 此处代码从中国地区AMSRE/2 BT Filter代码复制过来, 暂时没动. !!!
%     modisLcName = sprintf("MCD12Q1.A%s001.006_gcs_reclsfy_upscaled.tif", yearStr);
%     modisLcLayer = readgeoraster(fullfile(modisLcDir, modisLcName));
    modisLcLayer = zeros(amsrRowN, amsrColN);
    lcTypeList = unique(modisLcLayer); lcTypeList(lcTypeList == 0) = [];
    for j = 1: channelN % 顺序: {'10', '18', '23', '36', '06', '07', '89'}.
        break
        channel = channelList{j};
        amsrHVarName = sprintf('%sH%sYearArray', amsrType, channel);
        amsrVVarName = sprintf('%sV%sYearArray', amsrType, channel);
        amsrHArray = eval(amsrHVarName);
        amsrVArray = eval(amsrVVarName);
        amsrHIndexArray = ~ismember(amsrHArray, amsrNodata);
        amsrVIndexArray = ~ismember(amsrVArray, amsrNodata);
        [amsrHArray2, amsrVArray2] = deal(zeros(size(amsrHArray), 'int16'));
        for k = 1: length(lcTypeList)
            lcType = lcTypeList(k);
            lcIndexArray = repmat((modisLcLayer == lcType), 1, 1, size(amsrHArray, 3));

            amsrHLcIndexArray = lcIndexArray & amsrHIndexArray;
            amsrHLcArray = single(amsrHArray); amsrHLcArray(~amsrHLcIndexArray) = nan;
            amsrHStd = std(amsrHLcArray, 0, 'all', 'omitnan');
            amsrHMean = mean(amsrHLcArray, 'all', 'omitnan');
            outlierIndexArray = abs(amsrHLcArray - amsrHMean) > 3 * amsrHStd;
            amsrHLcIndexArray2 = amsrHLcIndexArray & ~outlierIndexArray;
            amsrHArray2(amsrHLcIndexArray2) = amsrHArray(amsrHLcIndexArray2);

            amsrVLcIndexArray = lcIndexArray & amsrVIndexArray;
            amsrVLcArray = single(amsrVArray); amsrVLcArray(~amsrVLcIndexArray) = nan;
            amsrVStd = std(amsrVLcArray, 0, 'all', 'omitnan');
            amsrVMean = mean(amsrVLcArray, 'all', 'omitnan');
            outlierIndexArray = abs(amsrVLcArray - amsrVMean) > 3 * amsrVStd;
            amsrVLcIndexArray2 = amsrVLcIndexArray & ~outlierIndexArray;
            amsrVArray2(amsrVLcIndexArray2) = amsrVArray(amsrVLcIndexArray2);

            % 异常值去除前后各LC分区的AMSRE/2 BT值直方图分布.
            f1 = figure; histogram(amsrVLcArray, Normalization='pdf');
            title([sprintf('%sV %sGHz BT in Landcover %d Original', amsrType, channel, lcType)]);
            f2 = figure; histogram(amsrVArray(amsrVLcIndexArray2), Normalization='pdf');
            title([sprintf('%sV %sGHz BT in Landcover %d filtered', amsrType, channel, lcType)]);

            f3 = figure; histogram(amsrHLcArray, Normalization='pdf');
            title([sprintf('%sH %sGHz BT in Landcover %d Original', amsrType, channel, lcType)]);
            f4 = figure; histogram(amsrHArray(amsrHLcIndexArray2), Normalization='pdf');
            title([sprintf('%sH %sGHz BT in Landcover %d filtered', amsrType, channel, lcType)]);
        end
        close all
        assignin('base', amsrHVarName, amsrHArray2);
        assignin('base', amsrVVarName, amsrVArray2);
    end

    % B) 去除值大于30000的像元值, 以及非湖泊, 冰川区小于18000的像元值.
    % !!! 此处代码从中国地区AMSRE/2 BT Filter代码复制过来, 暂时没动. !!!
%     waterBufferName = sprintf("MCD12Q1.A%s001.006_gcs_reclsfy_upscaled_waterBuffer.tif", yearStr);
%     waterBufferLayer = readgeoraster(fullfile(modisLcDir, waterBufferName));
    waterBufferLayer = zeros(amsrRowN, amsrColN);
    waterBufferIndexLayer = ismember(waterBufferLayer, waterLcCode);
    for j = 1: channelN  % 顺序: {'10', '18', '23', '36', '06', '07', '89'}.
        break
        amsrHVarName = sprintf('%sH%sYearArray', lower(amsrType), channelList{j});
        amsrVVarName = sprintf('%sV%sYearArray', lower(amsrType), channelList{j});
        amsrHArray = eval(amsrHVarName);
        amsrVArray = eval(amsrVVarName);
        waterBufferIndexArray = repmat(waterBufferIndexLayer, 1, 1, size(amsrHArray, 3));
        amsrHOutlierArray = amsrHArray > 30000 | (~waterBufferIndexArray & amsrHArray < 18000);
        amsrVOutlierArray = amsrVArray > 31000 | (~waterBufferIndexArray & amsrVArray < 18000);
        amsrHArray(amsrHOutlierArray) = amsrNodata(1);
        amsrVArray(amsrVOutlierArray) = amsrNodata(1);
        assignin('base', amsrHVarName, amsrHArray);
        assignin('base', amsrVVarName, amsrVArray);
    end

    % 获取MODIS LST和AMSRE/2 BT均有数据日期的文件路径和数据矩阵.
    [lstDateFilterList, modisDateIndex, amsrDateIndex] = intersect(modisLstDateList, amsrDateList);
    lstDateFilterN = length(lstDateFilterList);
    modisLstPathFilterList = cellfun(@fullfile, repmat({modisLstPrjYearDir}, lstDateFilterN, 1), ...
        modisLstYearList(modisDateIndex), UniformOutput=false);
    modisQcPathFilterList = cellfun(@fullfile, repmat({modisLstPrjYearDir}, lstDateFilterN, 1), ...
        modisQcYearList(modisDateIndex), UniformOutput=false);
    amsrPathFilterMatrix = amsrPathMatrix(amsrDateIndex, :);
    for j = 1: channelN % 顺序: {'10', '18', '23', '36', '06', '07', '89'}.
        amsrHVarName = sprintf('%sH%sYearArray', lower(amsrType), channelList{j});
        amsrVVarName = sprintf('%sV%sYearArray', lower(amsrType), channelList{j});
        amsrHArray = eval(amsrHVarName);
        amsrVArray = eval(amsrVVarName);
        assignin('base', amsrHVarName, amsrHArray(:, :, amsrDateIndex));
        assignin('base', amsrVVarName, amsrVArray(:, :, amsrDateIndex));
    end
    
    % 创建存储掩膜后MODIS LST和AMSRE/2 BT数据的年路径.
    modisLstMaskTifYearDir = fullfile(modisLstMaskTifDir, modisLstYearFolder);
    if ~exist(modisLstMaskTifYearDir, 'dir')
        mkdir(modisLstMaskTifYearDir)
    end
    amsrMaskTifYearDir = fullfile(amsrMaskTifDir, sprintf('%s_%sXXXX_TIF', amsrType, yearStr));
    if ~exist(amsrMaskTifYearDir, 'dir')
        mkdir(amsrMaskTifYearDir)
    end

    % 创建存储掩膜后年度MODIS LST和AMSRE/2 BT的矩阵.
    if ~modisMatExist || ~amsrMatExist
        modisLstMaskYearArray = zeros(amsrRowN, amsrColN, lstDateFilterN, 'single') * nan;
        [amsrHMaskYearArray, amsrVMaskYearArray] = ...
            deal(zeros(amsrRowN, amsrColN, channelN * lstDateFilterN, 'uint16'));
        startN = 1;
    end

    % 读取MODIS LST和AMSRE/2 BT数据, 并掩膜输出.
    for j = 1: lstDateFilterN
        % 掩膜后的MODIS LST文件输出路径.
        modisLstMaskFileName = split(modisLstPathFilterList{j}, '\');
        modisLstMaskFileName = modisLstMaskFileName{end};
        modisLstMaskFilePath = fullfile(modisLstMaskTifYearDir, modisLstMaskFileName);

        % 掩膜后的各通道AMSRE/2 BT文件输出路径.
        amsrMaskYearDateFolder = sprintf('%s_%s', amsrType, lstDateFilterList{j});
        amsrMaskYearDateDir = fullfile(amsrMaskTifYearDir, amsrMaskYearDateFolder);
        if ~exist(amsrMaskYearDateDir, 'dir')
            mkdir(amsrMaskYearDateDir);
        end
        amsrMaskFileNameList = split(amsrPathFilterMatrix(j, :), '\');
        amsrMaskFileNameList = amsrMaskFileNameList(:, :, end);
        amsrMaskFilePathList = fullfile(amsrMaskYearDateDir, amsrMaskFileNameList);

        % 输出的TIF数据是否存在的标记.
        amsrMaskFileExistList = false(cpN, 1);
        for k = 1: cpN
            amsrMaskFileExistList(k) = exist(amsrMaskFilePathList{k}, 'file');
        end
        amsrMaskFileN = sum(logical(amsrMaskFileExistList));

        % 在输出的掩膜后TIF文件存在的前提下, 如果Mat文件也存在, 则直接进入下次循环, 否则将当前日期TIF数据存
        %   储在年度矩阵中. 如果掩膜后的TIF文件不存在, 则Mat文件肯定不存在, 此时进行掩膜处理, 并保存TIF文件
        %   和Mat文件.
        if exist(modisLstMaskFilePath, 'file') && (amsrMaskFileN == cpN)
            if modisMatExist && amsrMatExist
                continue
            else
                modisLstMaskYearArray(:, :, j) = readgeoraster(modisLstMaskFilePath);
                [amsrHArray, amsrVArray] = deal(zeros(amsrRowN, amsrColN, channelN));
                for k = 1: channelN
                    amsrHArray(:, :, k) = readgeoraster(amsrMaskFilePathList{2*k-1});
                    amsrVArray(:, :, k) = readgeoraster(amsrMaskFilePathList{2*k});
                end
                amsrHMaskYearArray(:, :, startN:startN+channelN-1) = amsrHArray;
                amsrVMaskYearArray(:, :, startN:startN+channelN-1) = amsrVArray;
                startN = startN + channelN;
                continue
            end
        end

        % 读取和裁剪MODIS LST, QC数据层.
        modisLstLayer = readgeoraster(modisLstPathFilterList{j});
        modisLstLayer = modisLstLayer(modisLStartRow: modisLEndRow, modisLStartCol: modisLEndCol);

        modisQcLayer = readgeoraster(modisQcPathFilterList{j});
        modisQcLayer = modisQcLayer(modisLStartRow: modisLEndRow, modisLStartCol: modisLEndCol);

        % 将MODIS LST和发射率DN值转为实际值.
        modisLstLayer = double(modisLstLayer) * 0.02;
        % MYD11A1 V061 QC中, 原本代表质量好的0值不知为何被赋值为Nodata, Matlab将Nodata读成256, 因此QcBad
        %   中需要排除256. V006中不存在此问题.
        qcBadIndexLayer = ismember(modisQcLayer, qcBad1) | (modisQcLayer >= qcBad2 & ...
            modisQcLayer ~= 256);
        modisLstNodataIndexLayer = ismember(modisLstLayer, modisLstNodata * 0.02);
        modisLstLayer(qcBadIndexLayer | modisLstNodataIndexLayer) = nan;

        % MODIS LST影像升尺度.
        upscaledModisLstLayer = zeros(amsrRowN, amsrColN) * nan;
        for ii = 1: amsrRowN
            for jj = 1: amsrColN
                % 定位滑动块的位置.
                lstBlockTopRow = lstBlockBdy(1) + lstBlockSize(1) * (ii - skipLocation(1));
                lstBlockBottomRow = lstBlockBdy(2) + lstBlockSize(1) * (ii - skipLocation(1));
                lstBlockLeftCol = lstBlockBdy(3) + lstBlockSize(2) * (jj - skipLocation(2));
                lstBlockRightCol = lstBlockBdy(4) + lstBlockSize(2) * (jj - skipLocation(2));

                % 跳过不在AMSRE/2影像范围的MODIS LST像元.
                if lstBlockLeftCol <= 0 || lstBlockTopRow <= 0
                    continue
                end
                if lstBlockRightCol > modisLstColN || lstBlockBottomRow > modisLstRowN
                    continue
                end

                % 块计算.
                modisLstBlock = modisLstLayer(lstBlockTopRow: lstBlockBottomRow, ...
                    lstBlockLeftCol: lstBlockRightCol);
                if sum(~isnan(modisLstBlock), 'all') / lstBlockN < modisPixelPct
                    continue
                end
                upscaledModisLstLayer(ii, jj) = mean(modisLstBlock, 'all', 'omitnan');
            end
        end

        % AMSRE/2 BT质量控制, 排除PR值大于1和轨道间隙像元.
        [amsrHArray, amsrVArray] = deal(zeros(amsrRowN, amsrColN, channelN, 'uint16'));
        for k = 1: channelN % 顺序: {'10', '18', '23', '36', '06', '07', '89'}.
            amsrHArrayName = sprintf('%sH%sYearArray(:, :, j)', lower(amsrType), channelList{k});
            amsrVArrayName = sprintf('%sV%sYearArray(:, :, j)', lower(amsrType), channelList{k});
            amsrHArray(:, :, k) = eval(amsrHArrayName);
            amsrVArray(:, :, k) = eval(amsrVArrayName);
        end
        amsrPrArray = single(amsrHArray) ./ single(amsrVArray);
        amsrPrIndexLayer = logical(sum(amsrPrArray > 1, 3));
        amsrNodataIndexArray = ismember(amsrHArray, amsrNodata) | ismember(amsrVArray, amsrNodata);
        amsrNodataIndexLayer = logical(sum(amsrNodataIndexArray, 3));
        amsrNanIndexLayer = amsrPrIndexLayer | amsrNodataIndexLayer;
        amsrNanIndexArray = repmat(amsrNanIndexLayer, 1, 1, channelN);
        amsrHArray(amsrNanIndexArray) = amsrNodata(1);
        amsrVArray(amsrNanIndexArray) = amsrNodata(1);

        % 获取中国范围内AMSRE/2 BT数据和升尺度后MODIS LST数据中的共有可用像元.
        nanIndexLayer = isnan(upscaledModisLstLayer) | amsrNanIndexLayer | ~regionMaskLayer;
        nanIndexArray = repmat(nanIndexLayer, 1, 1, channelN);
        upscaledModisLstLayer(nanIndexLayer) = nan;
        amsrHArray(nanIndexArray) = amsrNodata(1);
        amsrVArray(nanIndexArray) = amsrNodata(1);

        % 输出质量控制且升尺度后的MODIS LST和AMSRE/2 BT数据.
        fprintf('输出%s %s的MODIS LST, %s BT掩膜数据.\n', lstDateFilterList{j}, dayNight, amsrType)
        geotiffwrite(modisLstMaskFilePath, single(upscaledModisLstLayer), amsrRef, ...
            GeoKeyDirectoryTag=geoTag, TiffTags=struct('Compression','LZW'));
        for k = 1: channelN
            geotiffwrite(amsrMaskFilePathList{2 * k - 1}, amsrHArray(:, :, k), amsrRef, ...
                GeoKeyDirectoryTag=geoTag, TiffTags=struct('Compression','LZW'));
            geotiffwrite(amsrMaskFilePathList{2 * k}, amsrVArray(:, :, k), amsrRef, ...
                GeoKeyDirectoryTag=geoTag, TiffTags=struct('Compression','LZW'));
        end

        % 年度掩膜MODIS LST和AMSRE/2 BT矩阵.
        if ~modisMatExist || ~amsrMatExist
            modisLstMaskYearArray(:, :, j) = upscaledModisLstLayer;
            amsrHMaskYearArray(:, :, startN: startN + channelN - 1) = amsrHArray;
            amsrVMaskYearArray(:, :, startN: startN + channelN - 1) = amsrVArray;
            startN = startN + channelN;
        end
    end

    % 存储掩膜后年度MODIS LST和AMSRE/2 BT的矩阵的mat文件.
    if ~modisMatExist || ~amsrMatExist
        fprintf('保存%s年的MODIS LST, %s BT Mat文件.\n', amsrType, yearStr)
        for j = 1: channelN
            channelIndexVector = j: channelN: channelN * lstDateFilterN;
            amsrHVarName = sprintf('%sH%sMaskYearArray', lower(amsrType), channelList{j});
            amsrVVarName = sprintf('%sV%sMaskYearArray', lower(amsrType), channelList{j});
            assignin('base', amsrHVarName, amsrHMaskYearArray(:, :, channelIndexVector));
            assignin('base', amsrVVarName, amsrVMaskYearArray(:, :, channelIndexVector));
        end
        clear amsrHMaskYearArray amsrVMaskYearArray
        save(modisLstMaskMatPath, amsrRefVar, 'modisLstMaskYearArray', 'lstDateFilterList');
        save(amsrMaskMatPath, amsrRefVar, 'amsr*MaskYearArray', 'lstDateFilterList');
    end
end

% system('shutdown -s -t 60')