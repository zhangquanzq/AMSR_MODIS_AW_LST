%%
%{
2019/01/10 说明
  本实验将modispixelpct阈值设置为0.6, 导致东北等地的AMSR2和MODIS像素匹配度较高的地方很少, 无法构建回归模型.
采用类似土地覆被区域的回归模型的策略被证实是不恰当的, 因为它们会导致巨大的误差.
  计划使用一个新的根据景观的异质性来设置阈值策略. 在土地覆被类型均匀的缓坡或平坦地区(如东北地区), 阈值可以设
置的越小越好, 而在陡峭的地形或高度异质性的地区, 阈值可以设置的越大.
  由于AMSR2 BT图像没有应的QC层来控制质量, 因此在检索的AMSRE LST图像中会出现大量的异常像素. 设法想出处理这个
问题的计策. 利用AMSR2 LST块像素的STD或一个像素的时间序列.

2022/09/03 说明
  以>30000(31000)和非冰川, 湖泊区域<18000为阈值, 去除AMSR2 BT中的异常值.
  Step3_AMSR2_LST_Model_Construction.m 中用到此阈值去除AMSR2 BT中异常值.
%}

%% 标识与预设参数.
% 指定的轨道的标识. 1标识升轨(A, 白天), 2标识降轨(D, 晚上).
flg1 = 2;

% 昼夜标记.
dayNight = {'Day', 'Night'};
dayNight = dayNight{flg1};

% 年份列表.
% yearList = 2012: 2021;
yearList = 2020: 2021;

% 通道, 极化列表.
channelList = {'10', '18', '23', '36', '06', '07', '89'};
channelN = length(channelList);
polarList = {'H', 'V'};
polarN = length(polarList);
cpN = channelN * polarN;

% 标记MODIS LST像元质量的QC值.
% QC值含义见文件: 'J:\AMSR2_MODIS_AW_LST\AMSR2_LST_Retrieval\Doc\MYD11A1_QC.csv'
qcBad1 = [2, 3]; qcBad2 = 192;

% 筛选有效AMSR2像元的阈值, 定义为一个AMSR2像元范围内的MODIS像元的百分比.
modisPixelPct = 0.6;

% AMSR2 BT影像中的NoData.
amsr2Nodata = [65534, 65535];

% MODIS LST影像中的Nodata.
modisLstNodata = [65535, 0];

% MODIS LC湖泊, 冰川编号.
waterLcCode = [3, 4];

%% 路径.
rootDir = 'E:\AMSR2_MODIS_AW_LST\';
dataDir = fullfile(rootDir, 'AMSR2_LST_Retrieval', 'Data');
addpath(fullfile(rootDir, 'Code', 'Functions'));

% 输入数据AMSR2 BT, MODIS LST文件夹路径.
amsr2CnTifDir = fullfile(dataDir, 'AMSR2_2_CN_TIF', 'L3.TB6GHz_10\2012\07\');
amsr2CnMatDir = fullfile(dataDir, 'AMSR2_2_CN_Matlab');
modisLstPrjCnDir = fullfile(dataDir, 'MYD11A1_2_PrjCN_TIF');
modisLcDir = fullfile(dataDir, 'MCD12Q1_2_MosaicCN_TIF');

% 输出数据路径, 同日期经过空间交集掩膜的AMSR2 BT和MODIS LST数据.
amsr2MaskTifDir = fullfile(dataDir, 'AMSR2_3_MaskCn_TIF');
if ~exist(amsr2MaskTifDir, 'dir')
    mkdir(amsr2MaskTifDir)
end

amsr2MaskMatDir = fullfile(dataDir, 'AMSR2_3_MaskCn_Matlab');
if ~exist(amsr2MaskMatDir, 'dir')
    mkdir(amsr2MaskMatDir)
end

modisLstMaskTifDir = fullfile(dataDir, 'MYD11A1_3_MaskCn_TIF');
if ~exist(modisLstMaskTifDir, 'dir')
    mkdir(modisLstMaskTifDir)
end

modisLstMaskMatDir = fullfile(dataDir, 'MYD11A1_3_MaskCn_Matlab');
if ~exist(modisLstMaskMatDir, 'dir')
    mkdir(modisLstMaskMatDir)
end

%% 读取数据参考和属性信息.
% 获取样例AMSR2 BT影像的空间参考信息.
amsr2BtPath = fullfile(amsr2CnTifDir, 'GW1AM2_20120700_01M_EQMA_L3SGT06HA2220220_BtH.tif');
amsr2Ref = georasterinfo(amsr2BtPath).RasterReference;
amsr2RowN = amsr2Ref.RasterSize(1); amsr2ColN = amsr2Ref.RasterSize(2);

% 获取样例MODIS LST影像的空间参考信息.
modisLstPath = fullfile(modisLstPrjCnDir, 'MYD11A1_2002XXX_TIF', 'MYD11A1.A2002185.LST_Day.tif');
modisLstRef = georasterinfo(modisLstPath).RasterReference;

% 读取MODIS像元面积和SRTM坡度数据.
modisAreaLayer = readgeoraster(fullfile(modisLstPrjCnDir, 'MYD11A1.PixelArea.tif'));
[srtmSlpLayer, srtmRef] = readgeoraster(fullfile(dataDir, 'SRTM', 'SRTM_0d01_CN_Slp.tif'));

% 从MODIS, SRTM影像中获取与中国区AMSR2影像边界对齐的起始和结束的行号和列号.
[modisLStartRow, modisLEndRow, modisLStartCol, modisLEndCol] = getBdyRowCol(modisLstRef, amsr2Ref);
[srtmStartRow, srtmEndRow, srtmStartCol, srtmEndCol] = getBdyRowCol(srtmRef, amsr2Ref);

% 裁剪后中国范围内的MODIS LST行列数.
modisLstColN = modisLEndCol - modisLStartCol + 1;
modisLstRowN = modisLEndRow - modisLStartRow + 1;

% 获取中国区域的范围栅格, MODIS像元面积影像, SRTM坡度数据.
cnMaskLayer = readgeoraster(fullfile(dataDir, 'Zones', 'ExtentCN_0d1.tif')) == 1;

modisAreaLayer = modisAreaLayer(modisLStartRow: modisLEndRow, modisLStartCol: modisLEndCol);
modisAreaLayer = double(modisAreaLayer) * 0.01;

srtmSlpLayer = srtmSlpLayer(srtmStartRow: srtmEndRow, srtmStartCol: srtmEndCol);
cosGamaLayer = cosd(srtmSlpLayer);

% 获取用于升尺度MODIS像元的移动滑块的行列数.
% 获取与中国区范围AMSR2影像左上角边界对齐的滑动块的左右边界列号与上下边界行号.
% 边界行列号: [topRow, bottomRow, leftCol, rightCol],  行列数: [blockRowN, blockColN].
[~, lstBlockSize] = getStartBlockRowCol(modisLstRef, amsr2Ref);
lstBlockN = prod(lstBlockSize);
lstBlockBdy = [1, lstBlockSize(1), 1, lstBlockSize(2)];

%% 读取各年份的数据, 筛选相同日期数据, 升尺度MODIS LST, 并输出空间交集数据.
% 分年份读取MODIS LST和AMSR2 BT的文件.
for i = 1: length(yearList)
    yearStr = num2str(yearList(i));
%     yearStr = '2020';
    disp(['数据筛选年份:', yearStr]);

    modisVersion = {'006', '061'};
    modisVersion = modisVersion{strcmp(yearStr, '2021')+1};

    % 输出的掩膜后年度MODIS LST和AMSR2 BT的矩阵的mat文件路径.
    modisLstMaskMatName = sprintf('MYD11A1_MaskCn_%s_%s.mat', yearStr, dayNight);
    modisLstMaskMatPath = fullfile(modisLstMaskMatDir, modisLstMaskMatName);
    amsr2MaskMatName = sprintf('AMSR2_BT_MaskCn_%s_%s.mat', yearStr, dayNight);
    amsr2MaskMatPath = fullfile(amsr2MaskMatDir, amsr2MaskMatName);
    modisMatExist = exist(modisLstMaskMatPath, 'file');
    amsr2MatExist = exist(amsr2MaskMatPath, 'file');

    % 读取MODIS LST, QC以及Emis 31, 32的文件列表.
    modisLstYearFolder = sprintf('MYD11A1_%sXXX_TIF', yearStr);
    modisLstPrjYearDir = fullfile(modisLstPrjCnDir, modisLstYearFolder);

    modisLstYearList = dir(fullfile(modisLstPrjYearDir, sprintf('*LST_%s.tif', dayNight)));
    modisLstYearList = {modisLstYearList.name}';
    modisLstYearN = length(modisLstYearList);

    modisQcYearList = dir(fullfile(modisLstPrjYearDir, sprintf('*QC_%s.tif', dayNight)));
    modisQcYearList = {modisQcYearList.name}';
    modisQcYearN = length(modisQcYearList);

    modisEmis31YearList = dir(fullfile(modisLstPrjYearDir, '*Emis_31.tif'));
    modisEmis31YearList = {modisEmis31YearList.name}';
    modisEmis31YearN = length(modisEmis31YearList);

    modisEmis32YearList = dir(fullfile(modisLstPrjYearDir, '*Emis_32.tif'));
    modisEmis32YearList = {modisEmis32YearList.name}';
    modisEmis32YearN = length(modisEmis32YearList);

    % 判断MODIS LST, QC以及Emiss文件是否一一对应.
    if ~isequal(modisLstYearN, modisQcYearN, modisEmis31YearN, modisEmis32YearN)
        error('%s年的MODIS LST, QC, Emis31, Emis32数据不匹配, 请检查!', yearStr)
    end
    [modisLstDateList, modisQcDateList, modisEmis31DateList, modisEmis32DateList] = ...
        deal(cell(modisLstYearN, 1));
    for j = 1: modisLstYearN
        modisLstDate = split(modisLstYearList{j}, '.');
        modisLstDateList{j} = yday2ymd(modisLstDate{2}(2:end));

        modisQcDate = split(modisQcYearList{j}, '.');
        modisQcDateList{j} = yday2ymd(modisQcDate{2}(2:end));

        modisEmis31Date = split(modisEmis31YearList{j}, '.');
        modisEmis31DateList{j} = yday2ymd(modisEmis31Date{2}(2:end));

        modisEmis32Date = split(modisEmis32YearList{j}, '.');
        modisEmis32DateList{j} = yday2ymd(modisEmis32Date{2}(2:end));
    end
    if  ~isequal(modisLstDateList, modisQcDateList, modisEmis31DateList, modisEmis32DateList)
        error('%s年的MODIS LST与QC数据不匹配, 请检查!', yearStr)
    end

    % 从mat文件中读取AMSR2 BT数据各通道文件列表, 数据本身及空间参考.
    % 变量 amsr2PathMatrix 为存储AMSR2 BT TIF文件路径字符串的cell矩阵. 矩阵的行表示不同的日期, 列表示不同
    %   的极化通道. 第一行为表头, 存储2极化和7通道的名称: ['10H', '10V', '18H', '18V', '23H', '23V',
    %   '36H', '36V', '6H', '6V', '7H', '7V', '89H', '89V']. 其他行依次存储这些极化通道对应的不同日期的
    %   AMSR2 BT TIF文件的路径字符串.
    amsr2CnMatName = sprintf('AMSR2_BT_%s_%s.mat', yearStr, dayNight);
    amsr2CnMatPath = fullfile(amsr2CnMatDir, amsr2CnMatName);
    load(amsr2CnMatPath, 'amsr2PathMatrix', 'amsr2Ref', 'amsr2*YearArray')
    amsr2PathMatrix = replace(amsr2PathMatrix, 'F:\AMSR_MODIS_Fusion\Data\', dataDir);

    % 输入AMSR2 BT数据日期列表.
    amsr2PathMatrix = amsr2PathMatrix(2:end, :);
    [~, amsr2Name, ~] = fileparts(amsr2PathMatrix);
    amsr2DateMatrix = split(amsr2Name, '_');
    amsr2DateList = amsr2DateMatrix(:, 1, 2);
%     amsr2Datetime = datetime(cell2mat(amsr2DateList), InputFormat='yyyyMMdd');

    % AMSR2 BT数据异常值检测.
    % A) 以MODIS LC 5分区为基准, 去除每个分区中AMSR2 BT离差超过3倍标准差的值 (暂时不用, 用break语句跳过).
    modisLcName = sprintf("MCD12Q1.A%s001.%s_gcs_reclsfy_upscaled.tif", yearStr, modisVersion);
    modisLcLayer = readgeoraster(fullfile(modisLcDir, modisLcName));
    lcTypeList = unique(modisLcLayer); lcTypeList(lcTypeList == 0) = [];
    for j = 1: channelN % 顺序: {'10', '18', '23', '36', '06', '07', '89'}.
        break
        channel = channelList{j};
        amsr2HVarName = sprintf('amsr2H%sYearArray', channel);
        amsr2VVarName = sprintf('amsr2V%sYearArray', channel);
        amsr2HArray = eval(amsr2HVarName);
        amsr2VArray = eval(amsr2VVarName);
        amsr2HIndexArray = ~ismember(amsr2HArray, amsr2Nodata);
        amsr2VIndexArray = ~ismember(amsr2VArray, amsr2Nodata);
        [amsr2HArray2, amsr2VArray2] = deal(zeros(size(amsr2HArray), 'int16'));
        for k = 1: length(lcTypeList)
            lcType = lcTypeList(k);
            lcIndexArray = repmat((modisLcLayer == lcType), 1, 1, size(amsr2HArray, 3));

            amsr2HLcIndexArray = lcIndexArray & amsr2HIndexArray;
            amsr2HLcArray = single(amsr2HArray); amsr2HLcArray(~amsr2HLcIndexArray) = nan;
            amsr2HStd = std(amsr2HLcArray, 0, 'all', 'omitnan');
            amsr2HMean = mean(amsr2HLcArray, 'all', 'omitnan');
            outlierIndexArray = abs(amsr2HLcArray - amsr2HMean) > 3 * amsr2HStd;
            amsr2HLcIndexArray2 = amsr2HLcIndexArray & ~outlierIndexArray;
            amsr2HArray2(amsr2HLcIndexArray2) = amsr2HArray(amsr2HLcIndexArray2);

            amsr2VLcIndexArray = lcIndexArray & amsr2VIndexArray;
            amsr2VLcArray = single(amsr2VArray); amsr2VLcArray(~amsr2VLcIndexArray) = nan;
            amsr2VStd = std(amsr2VLcArray, 0, 'all', 'omitnan');
            amsr2VMean = mean(amsr2VLcArray, 'all', 'omitnan');
            outlierIndexArray = abs(amsr2VLcArray - amsr2VMean) > 3 * amsr2VStd;
            amsr2VLcIndexArray2 = amsr2VLcIndexArray & ~outlierIndexArray;
            amsr2VArray2(amsr2VLcIndexArray2) = amsr2VArray(amsr2VLcIndexArray2);

            % 异常值去除前后各LC分区的AMSR2 BT值直方图分布.
            f1 = figure; histogram(amsr2VLcArray, Normalization='pdf');
            title([sprintf('AMSR2V %sGHz BT in Landcover %d Original', channel, lcType)]);
            f2 = figure; histogram(amsr2VArray(amsr2VLcIndexArray2), Normalization='pdf');
            title([sprintf('AMSR2V %sGHz BT in Landcover %d filtered', channel, lcType)]);

            f3 = figure; histogram(amsr2HLcArray, Normalization='pdf');
            title([sprintf('AMSR2H %sGHz BT in Landcover %d Original', channel, lcType)]);
            f4 = figure; histogram(amsr2HArray(amsr2HLcIndexArray2), Normalization='pdf');
            title([sprintf('AMSR2H %sGHz BT in Landcover %d filtered', channel, lcType)]);
        end
        close all
        assignin('base', amsr2HVarName, amsr2HArray2);
        assignin('base', amsr2VVarName, amsr2VArray2);
    end

    % B) 去除值大于30000的像元值, 以及非湖泊, 冰川区小于18000的像元值.
    waterBufferName = sprintf("MCD12Q1.A%s001.%s_gcs_reclsfy_upscaled_waterBuffer.tif", ...
        yearStr, modisVersion);
    waterBufferLayer = readgeoraster(fullfile(modisLcDir, waterBufferName));
    waterBufferIndexLayer = ismember(waterBufferLayer, waterLcCode);
    for j = 1: channelN  % 顺序: {'10', '18', '23', '36', '06', '07', '89'}.
%         break
        amsr2HVarName = sprintf('amsr2H%sYearArray',  channelList{j});
        amsr2VVarName = sprintf('amsr2V%sYearArray',  channelList{j});
        amsr2HArray = eval(amsr2HVarName);
        amsr2VArray = eval(amsr2VVarName);
        waterBufferIndexArray = repmat(waterBufferIndexLayer, 1, 1, size(amsr2HArray, 3));
        amsr2HOutlierArray = amsr2HArray > 30000 | (~waterBufferIndexArray & amsr2HArray < 18000);
        amsr2VOutlierArray = amsr2VArray > 31000 | (~waterBufferIndexArray & amsr2VArray < 18000);
        amsr2HArray(amsr2HOutlierArray) = amsr2Nodata(1);
        amsr2VArray(amsr2VOutlierArray) = amsr2Nodata(1);
        assignin('base', amsr2HVarName, amsr2HArray);
        assignin('base', amsr2VVarName, amsr2VArray);
    end

    % 获取MODIS LST和AMSR2 BT均有数据日期的文件路径和数据矩阵.
    [lstDateFilterList, modisDateIndex, amsr2DateIndex] = intersect(modisLstDateList,amsr2DateList);
    lstDateFilterN = length(lstDateFilterList);
    modisLstPathFilterList = cellfun(@fullfile, repmat({modisLstPrjYearDir}, lstDateFilterN, 1),...
        modisLstYearList(modisDateIndex), 'UniformOutput', false);
    modisQcPathFilterList = cellfun(@fullfile,repmat({modisLstPrjYearDir},lstDateFilterN,1),...
        modisQcYearList(modisDateIndex), 'UniformOutput', false);
    modisEmis31PathFilterList = cellfun(@fullfile,repmat({modisLstPrjYearDir},lstDateFilterN,1),...
        modisEmis31YearList(modisDateIndex), 'UniformOutput', false);
    modisEmis32PathFilterList = cellfun(@fullfile,repmat({modisLstPrjYearDir},lstDateFilterN,1),...
        modisEmis32YearList(modisDateIndex), 'UniformOutput', false);
    amsr2PathFilterMatrix = amsr2PathMatrix(amsr2DateIndex, :);
    for j = 1: channelN % 顺序: {'10', '18', '23', '36', '06', '07', '89'}.
        amsr2HVarName = sprintf('amsr2H%sYearArray', channelList{j});
        amsr2VVarName = sprintf('amsr2V%sYearArray', channelList{j});
        amsr2HArray = eval(amsr2HVarName);
        amsr2VArray = eval(amsr2VVarName);
        assignin('base', amsr2HVarName, amsr2HArray(:, :, amsr2DateIndex));
        assignin('base', amsr2VVarName, amsr2VArray(:, :, amsr2DateIndex));
    end
    
    % 创建存储掩膜后MODIS LST和AMSR2 BT数据的年路径.
    modisLstMaskTifYearDir = fullfile(modisLstMaskTifDir, modisLstYearFolder);
    if ~exist(modisLstMaskTifYearDir, 'dir')
        mkdir(modisLstMaskTifYearDir)
    end

    amsr2MaskTifYearDir = fullfile(amsr2MaskTifDir, sprintf('AMSR2_%sXXXX_TIF', yearStr));
    if ~exist(amsr2MaskTifYearDir, 'dir')
        mkdir(amsr2MaskTifYearDir)
    end

    % 创建存储掩膜后年度MODIS LST和AMSR2 BT的矩阵.
    if ~modisMatExist || ~amsr2MatExist
        modisLstMaskYearArray = zeros(amsr2RowN, amsr2ColN, lstDateFilterN, 'single') * nan;
        [amsr2HMaskYearArray, amsr2VMaskYearArray] = ...
            deal(zeros(amsr2RowN, amsr2ColN, channelN * lstDateFilterN, 'uint16'));
        startN = 1;
    end

    % 读取MODIS LST和AMSR2 BT数据, 并掩膜输出.
    for j = 1: lstDateFilterN
        % 掩膜后的MODIS LST文件输出路径.
        modisLstMaskFileName = split(modisLstPathFilterList{j}, '\');
        modisLstMaskFileName = modisLstMaskFileName{end};
        modisLstMaskFilePath = fullfile(modisLstMaskTifYearDir, modisLstMaskFileName);

        % 掩膜后的各通道AMSR2 BT文件输出路径.
        amsr2MaskYearDateDir = fullfile(amsr2MaskTifYearDir, ['AMSR2_', lstDateFilterList{j}]);
        if ~exist(amsr2MaskYearDateDir, 'dir')
            mkdir(amsr2MaskYearDateDir);
        end
        amsr2MaskFileNameList = split(amsr2PathFilterMatrix(j, :), '\');
        amsr2MaskFileNameList = amsr2MaskFileNameList(:, :, end);
        amsr2MaskFilePathList = fullfile(amsr2MaskYearDateDir, amsr2MaskFileNameList);

        % 输出的TIF数据是否存在的标记.
        amsr2MaskFileExistList = false(cpN, 1);
        for k = 1: cpN
            amsr2MaskFileExistList(k) = exist(amsr2MaskFilePathList{k}, 'file');
        end
        amsr2MaskFileN = sum(logical(amsr2MaskFileExistList));

        % 在输出的掩膜后TIF文件存在的前提下, 如果Mat文件也存在, 则直接进入下次循环, 否则将当前日期TIF数据存
        %   储在年度矩阵中. 如果掩膜后的TIF文件不存在, 则Mat文件肯定不存在, 此时进行掩膜处理, 并保存TIF文件
        %   和Mat文件.
        if exist(modisLstMaskFilePath, 'file') && (amsr2MaskFileN == cpN)
            if modisMatExist && amsr2MatExist
                continue
            else
                modisLstMaskYearArray(:, :, j) = readgeoraster(modisLstMaskFilePath);
                [amsr2HArray, amsr2VArray] = deal(zeros(amsr2RowN, amsr2ColN, channelN));
                for k = 1: channelN
                    amsr2HArray(:, :, k) = readgeoraster(amsr2MaskFilePathList{2*k-1});
                    amsr2VArray(:, :, k) = readgeoraster(amsr2MaskFilePathList{2*k});
                end
                amsr2HMaskYearArray(:, :, startN:startN+channelN-1) = amsr2HArray;
                amsr2VMaskYearArray(:, :, startN:startN+channelN-1) = amsr2VArray;
                startN = startN + channelN;
                continue
            end
        end

        % 读取MODIS LST, QC, Emis31, Emis32数据层.
        modisLstLayer = readgeoraster(modisLstPathFilterList{j});
        modisQcLayer = readgeoraster(modisQcPathFilterList{j});
        emis31Layer = readgeoraster(modisEmis31PathFilterList{j});
        emis32Layer = readgeoraster(modisEmis32PathFilterList{j});

        % 将这些数据裁剪到AMSR2影像定义的中国范围.
        modisLstLayer = modisLstLayer(modisLStartRow: modisLEndRow, modisLStartCol: modisLEndCol);
        modisQcLayer = modisQcLayer(modisLStartRow: modisLEndRow, modisLStartCol: modisLEndCol);
        emis31Layer = emis31Layer(modisLStartRow: modisLEndRow, modisLStartCol: modisLEndCol);
        emis32Layer = emis32Layer(modisLStartRow: modisLEndRow, modisLStartCol: modisLEndCol);

        % 将MODIS LST和发射率DN值转为实际值.
        modisLstLayer = double(modisLstLayer) * 0.02;
        % MYD11A1 V061 QC中, 原本代表质量好的0值不知为何被赋值为Nodata, Matlab将Nodata读成256, 因此QcBad
        %   中需要排除256. V006中不存在此问题.
        qcBadIndexLayer = ismember(modisQcLayer, qcBad1) | (modisQcLayer >= qcBad2 & ...
            modisQcLayer ~= 256);
        modisLstNodataIndexLayer = ismember(modisLstLayer, modisLstNodata * 0.02);
        modisLstLayer(qcBadIndexLayer | modisLstNodataIndexLayer) = nan;
        emis31Layer = double(emis31Layer) * 0.002 + 0.49;
        emis31Layer(emis31Layer == 255 * 0.002 + 0.49) = nan;
        emis32Layer = double(emis32Layer) * 0.002 + 0.49;
        emis32Layer(emis32Layer == 255 * 0.002 + 0.49) = nan;

        % MODIS LST影像升尺度.
        modisBbeLayer = 0.273 + 1.778 * emis31Layer - 1.807 * emis31Layer .* ...
            emis32Layer - 1.037 * emis32Layer + 1.774 * emis32Layer .^ 2;
        numeratorLayer = modisAreaLayer .* modisBbeLayer .* modisLstLayer .^ 4 .* ...
            secd(srtmSlpLayer) ./ cosGamaLayer;
        numeratorLayer(numeratorLayer < 0) = nan;
        denominatorLayer = modisBbeLayer .* modisAreaLayer .* secd(srtmSlpLayer);
        [numeratorSumLayer, denominatorSumLayer] = deal(zeros(amsr2RowN, amsr2ColN) * nan);
        for ii = 1: amsr2RowN
            for jj = 1: amsr2ColN
                % 定位滑动块的位置.
                lstBlockTopRow = lstBlockBdy(1) + lstBlockSize(1) * (ii - 1);
                lstBlockBottomRow = lstBlockBdy(2) + lstBlockSize(1) * (ii - 1);
                lstBlockLeftCol = lstBlockBdy(3) + lstBlockSize(2) * (jj - 1);
                lstBlockRightCol = lstBlockBdy(4) + lstBlockSize(2) * (jj - 1);
                if lstBlockRightCol > modisLstColN || lstBlockBottomRow > modisLstRowN
                    continue
                end

                % 块计算.
                numeratorBlock = numeratorLayer(lstBlockTopRow: lstBlockBottomRow, ...
                    lstBlockLeftCol: lstBlockRightCol);
                denominatorBlock = denominatorLayer(lstBlockTopRow: lstBlockBottomRow, ...
                    lstBlockLeftCol: lstBlockRightCol);
                blockAvailableIndex = ~isnan(numeratorBlock) & ~isnan(denominatorBlock);
                availablePixelRatio = sum(blockAvailableIndex(:)) / lstBlockN;
                if availablePixelRatio < modisPixelPct
                    continue
                end
                numeratorBlock(~blockAvailableIndex) = nan;
                denominatorBlock(~blockAvailableIndex) = nan;
                numeratorSumLayer(ii, jj) = sum(numeratorBlock(:), 'omitnan');
                denominatorSumLayer(ii, jj) = sum(denominatorBlock(:), 'omitnan');
            end
        end
        N = ones(amsr2RowN, amsr2ColN) * 4;
        upscaledModisLstLayer = nthroot(numeratorSumLayer ./ denominatorSumLayer, N);

        % AMSR2 BT质量控制, 排除PR值大于1和轨道间隙像元.
        [amsr2HArray, amsr2VArray] = deal(zeros(amsr2RowN, amsr2ColN, channelN, 'uint16'));
        for k = 1: channelN % 顺序: {'10', '18', '23', '36', '06', '07', '89'}.
            amsr2HArray(:, :, k) = eval(sprintf('amsr2H%sYearArray(:, :, j)', channelList{k}));
            amsr2VArray(:, :, k) = eval(sprintf('amsr2V%sYearArray(:, :, j)', channelList{k}));
        end
        amsr2PrArray = single(amsr2HArray) ./ single(amsr2VArray);
        amsr2PrIndexLayer = logical(sum(amsr2PrArray > 1, 3));
        amsr2NodataIndexArray = ismember(amsr2HArray,amsr2Nodata)|ismember(amsr2VArray,amsr2Nodata);
        amsr2NodataIndexLayer = logical(sum(amsr2NodataIndexArray, 3));
        amsr2NanIndexLayer = amsr2PrIndexLayer | amsr2NodataIndexLayer;
        amsr2NanIndexArray = repmat(amsr2NanIndexLayer, 1, 1, channelN);
        amsr2HArray(amsr2NanIndexArray) = amsr2Nodata(1);
        amsr2VArray(amsr2NanIndexArray) = amsr2Nodata(1);

        % 获取中国范围内AMSR2 BT数据和升尺度后MODIS LST数据中的共有可用像元.
        modisNanIndexLayer = isnan(upscaledModisLstLayer);
        nanIndexLayer = modisNanIndexLayer | amsr2NanIndexLayer | ~cnMaskLayer;
        nanIndexArray = repmat(nanIndexLayer, 1, 1, channelN);
        upscaledModisLstLayer(nanIndexLayer) = nan;
        amsr2HArray(nanIndexArray) = amsr2Nodata(1);
        amsr2VArray(nanIndexArray) = amsr2Nodata(1);

        % 输出质量控制且升尺度后的MODIS LST和AMSR2 BT数据.
        fprintf('输出%s %s的MODIS LST, AMSR2 BT掩膜数据.\n', lstDateFilterList{j}, dayNight)
        geotiffwrite(modisLstMaskFilePath, single(upscaledModisLstLayer), amsr2Ref, ...
            TiffTags=struct('Compression','LZW'));
        for k = 1: channelN
            geotiffwrite(amsr2MaskFilePathList{2 * k - 1}, amsr2HArray(:, :, k), amsr2Ref, ...
                TiffTags=struct('Compression','LZW'));
            geotiffwrite(amsr2MaskFilePathList{2 * k}, amsr2VArray(:, :, k), amsr2Ref, ...
                TiffTags=struct('Compression','LZW'));
        end

        % 年度掩膜MODIS LST和AMSR2 BT矩阵.
        if ~modisMatExist || ~amsr2MatExist
            modisLstMaskYearArray(:, :, j) = upscaledModisLstLayer;
            amsr2HMaskYearArray(:, :, startN: startN + channelN - 1) = amsr2HArray;
            amsr2VMaskYearArray(:, :, startN: startN + channelN - 1) = amsr2VArray;
            startN = startN + channelN;
        end
    end

    % 存储掩膜后年度MODIS LST和AMSR2 BT的矩阵的mat文件.
    if ~modisMatExist || ~amsr2MatExist
        fprintf('保存%s年的MODIS LST, AMSR2 BT Mat文件.\n', yearStr)
        for j = 1: channelN
            channelIndexVector = j: channelN: channelN * lstDateFilterN;
            amsr2HVarName = sprintf('amsr2H%sMaskYearArray', channelList{j});
            amsr2VVarName = sprintf('amsr2V%sMaskYearArray', channelList{j});
            assignin('base', amsr2HVarName, amsr2HMaskYearArray(:, :, channelIndexVector));
            assignin('base', amsr2VVarName, amsr2VMaskYearArray(:, :, channelIndexVector));
        end
        clear amsr2HMaskYearArray amsr2VMaskYearArray
        save(modisLstMaskMatPath, 'amsr2Ref', 'modisLstMaskYearArray', 'lstDateFilterList');
        save(amsr2MaskMatPath, 'amsr2Ref', 'amsr2H*MaskYearArray', 'amsr2V*MaskYearArray', ...
            'lstDateFilterList');
    end
end

system('shutdown -s -t 60')