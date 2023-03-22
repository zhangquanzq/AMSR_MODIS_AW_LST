%% 用于AMSRE/2 LST降尺度试验的数据筛选和准备.

%% 标记和预设参数.
% 指定区域的标记. 1表示内蒙古, 2表示晋南豫西, 3表示云贵高原, 4表示北大河流域, 5表示那曲.
flg1 = 2;
% 指定昼夜的标记. 1表示白天, 2表示夜晚.
flg2 = 1;
% 指定微波传感器的标记. 1表示AMSRE, 2表示AMSR2.
flg3 = 2;

% 研究区, 昼夜.
region = {'NMG', 'JNYX', 'YGGY', 'BRB', 'Naqu'};
region = region{flg1};

dayNight = {'Day', 'Night'};
dayNight = dayNight{flg2};

% 传感器, 年份列表.
sensor = {'AMSRE', 'AMSR2'};
sensor = sensor{flg3};

yearList = {2010, 2013: 2020};
yearList = yearList{flg3};

cellsizeList = {{'0.01', '0.03', '0.08', '0.25'}, {'0.01', '0.03', '0.1'}};
cellsizeList = cellsizeList{flg3};
cellsizeN = length(cellsizeList);

% 地形参数, 云覆盖比例, 地表温度质量控制.
topoParams = {'Elev', 'Slp'};
cloudPctEdges = [0, 0.1];  % 针对降尺度试验.
% cloudPctEdges = [0.1, 0.9];  % 针对融合试验.
% modisLstQcCode = [0, 17];  % 只保留0和17的话, 满足云比例条件的数据量大幅减小.
qcBad1 = [2, 3]; qcBad2 = 192;  % 保留除云覆盖和无效值之外的所有LST像元.
drStr = sprintf('%s_%s', dayNight, region);

% AMSRE/2和MODIS地表温度数据Nodata.
amsrLstNodata = [nan, 0]; amsrLstNodata = amsrLstNodata(flg3);
modisLstNodata = 0;

%% 路径.
% 根目录.
rootDir = 'F:\AMSR_MODIS_AW_LST';
dataDir1 = fullfile(rootDir, 'AMSR_LST_Retrieval\Data');
dataDir2 = fullfile(rootDir, 'AMSR_LST_Downscaling\Data');
addpath(fullfile(rootDir, 'Code\functions'));

% 输入数据路径.
amsrLstDir = fullfile(dataDir1, sprintf('%s_4_LSTCN_TIF', sensor));
modisLstDir = fullfile(dataDir1, 'MYD11A1_2_PrjCN_TIF');

ndviDir = fullfile(dataDir2, 'MYD13A2_2_PrjCN_TIF');

regionExtentPath = fullfile(dataDir2, 'Feature', sprintf('%s_Extent.shp', region));
regionBufferPath = fullfile(dataDir2, 'Feature', sprintf('%s_Buffer.shp', region));

snapAmsrePath = fullfile(dataDir1, 'AMSRE_2_CN_TIF\AMSRE_2004XXXX\AMSRE_20040102',...
    'AMSRE_D25_20040102A_v03_06H.tif');
snapAmsr2Path = fullfile(dataDir1, 'AMSR2_2_CN_TIF\L3.TB6GHz_10\2012\07',...
    'GW1AM2_20120703_01D_EQMA_L3SGT06HA2220220_BtH.tif');
snapRasterPath = {snapAmsrePath, snapAmsr2Path}; snapRasterPath = snapRasterPath{flg1};

% 输出数据路径.
regionDir = fullfile(dataDir2, sprintf('Region_%s_%s', region, cellsizeList{end}));
if ~exist(regionDir, 'dir')
    mkdir(regionDir)
end
srtmRegionDir = fullfile(regionDir, sprintf('SRTM_%s', region));
if ~exist(srtmRegionDir, 'dir')
    mkdir(srtmRegionDir)
end

%% SRTM数据准备. 裁剪研究区的SRTM高程, 坡度数据, 并生成分辨率序列.
for i = 1: length(topoParams)
    topo = topoParams{i};

    % 裁剪研究区的SRTM数据, 包括原始范围和缓冲区范围.
    srtmTopoPath = fullfile(dataDir1, 'SRTM', sprintf('SRTM_0d01_CN_%s.tif', topo));
    srtmTopoExtentPath = fullfile(srtmRegionDir, sprintf('SRTM_0.01_%s_%s.tif', region, topo));
    if ~exist(srtmTopoExtentPath, 'file')
        clipRaster(srtmTopoPath, regionExtentPath, srtmTopoExtentPath);
    end
    srtmTopoBufferPath = fullfile(srtmRegionDir, sprintf('SRTM_0.01_buffer_%s.tif', topo));
    if ~exist(srtmTopoBufferPath, 'file')
        clipRaster(srtmTopoPath, regionBufferPath, srtmTopoBufferPath);
    end

    % 重采样分辨率序列的SRTM数据.
    for j = 1: cellsizeN
        cellsize = cellsizeList{j};

        srtmTopoResample1Name = sprintf('SRTM_%s_%s_%s.tif', cellsize, region, topo);
        srtmTopoResample1Path = fullfile(srtmRegionDir, srtmTopoResample1Name);
        if exist(srtmTopoResample1Path, 'file')
            continue
        end

        srtmTopoResample2Name = sprintf('SRTM_%s_buffer_%s.tif', cellsize, topo);
        srtmTopoResample2Path = fullfile(srtmRegionDir, srtmTopoResample2Name);
        if exist(srtmTopoResample2Path, 'file')
            delete(srtmTopoResample2Path)
        end

        resampleRaster(srtmTopoBufferPath, srtmTopoResample2Path, cellsize, snapAmsrePath);
        clipRaster(srtmTopoResample2Path, regionExtentPath, srtmTopoResample1Path);
        delete(srtmTopoResample2Path)
    end
    delete(srtmTopoBufferPath)
end

%% 分年份筛选可用数据.
for i = 1: length(yearList)
    yearNum = yearList(i);
%     yearNum = 2013;

    % 创建存放研究区昼或夜数据的年度文件夹.
    yearDir = fullfile(regionDir, sprintf('%s_%d_%s', region, yearNum, dayNight));
    if ~exist(yearDir, 'dir')
        mkdir(yearDir)
    end

    % 统计数据质量的CSV文件路径.
    staRegionCsvName = sprintf('DateSelect_%s_%d_%s.csv', region, yearNum, dayNight);
    staRegionCsvPath = fullfile(regionDir, staRegionCsvName);

    % 读取数据列表-----------------------------------------------------------------------------------
    % 读取MODIS地表温度数据文件名和日期列表.
    modisLstYearDir = fullfile(modisLstDir, sprintf('MYD11A1_%dXXX_TIF', yearNum));

    modisLstStr = sprintf('MYD11A1*LST_%s.tif', dayNight);
    modisLstNameList = {dir(fullfile(modisLstYearDir, modisLstStr)).name}';
    modisDateList = string(cellfun(@(x) yday2ymd(x(10:16)),modisLstNameList,UniformOutput=false));

    modisQcStr = sprintf('MYD11A1*QC_%s.tif', dayNight);
    modisQcNameList = {dir(fullfile(modisLstYearDir, modisQcStr)).name}';
    modisQcDateList = string(cellfun(@(x) yday2ymd(x(10:16)),modisQcNameList,UniformOutput=false));

    if (length(modisLstNameList) ~= length(modisQcNameList)) || sum(modisDateList ~= ...
            modisQcDateList) > 0
        error('MODIS LST数据与QC数据不匹配, 请检查.')
    end
    modisLstPathList = fullfile(modisLstYearDir, modisLstNameList);
    modisQcPathList = fullfile(modisLstYearDir, modisQcNameList);

    % 读取AMSRE/2地表温度数据文件名和日期列表.
    amsrLstYearDir = fullfile(amsrLstDir, sprintf('%s_LST_%dXXXX_TIF', sensor, yearNum));
    amsrLstNameList = {dir(fullfile(amsrLstYearDir, sprintf('AMSR*%s*.tif', dayNight))).name}';
    sIndex = strfind(amsrLstNameList{1}, num2str(yearNum)); eIndex = sIndex + 7;
    amsrDateList = string(cellfun(@(x) x(sIndex: eIndex), amsrLstNameList, UniformOutput=false));
    amsrLstPathList = fullfile(amsrLstYearDir, amsrLstNameList);

    % 读取MODIS NDVI数据文件名和日期列表.
    ndviYearDir = fullfile(ndviDir, sprintf('MYD13A2_%dXXX', yearNum));
    ndviLastYearDir = fullfile(ndviDir, sprintf('MYD13A2_%dXXX', yearNum - 1));
    ndviNextYearDir = fullfile(ndviDir, sprintf('MYD13A2_%dXXX', yearNum + 1));

    ndvi361Path = fullfile(ndviLastYearDir, sprintf('MYD13A2.A%d361.061_NDVI.tif', yearNum - 1));
    ndvi009Path = fullfile(ndviNextYearDir, sprintf('MYD13A2.A%d009.061_NDVI.tif', yearNum + 1));
    ndviQa361Path = fullfile(ndviLastYearDir, sprintf('MYD13A2.A%d361.061_QA.tif', yearNum - 1));
    ndviQa009Path = fullfile(ndviNextYearDir, sprintf('MYD13A2.A%d009.061_QA.tif', yearNum + 1));
    if ~exist(ndvi361Path, 'file') || ~exist(ndvi009Path, 'file') || ...
            ~exist(ndviQa361Path, 'file') || ~exist(ndviQa009Path, 'file')
        error('上一年最后一个或下一年第一个MODIS NDVI数据与QA数据不存在, 请检查.')
    end

    ndviNameList = {dir(fullfile(ndviYearDir, 'MYD13A2*NDVI.tif')).name}';
    ndviPathList = fullfile(ndviYearDir, ndviNameList);
    ndviFullPathList = [ndvi361Path; ndviPathList; ndvi009Path];

    ndviQaNameList = {dir(fullfile(ndviYearDir, 'MYD13A2*QA.tif')).name}';
    ndviQaPathList = fullfile(ndviYearDir, ndviQaNameList);
    ndviQaFullPathList = [ndviQa361Path; ndviQaPathList; ndviQa009Path];

    [~, ndviNameList, ~] = cellfun(@(x) fileparts(x), ndviFullPathList, UniformOutput=false);
    ndviDateList = string(cellfun(@(x) yday2ymd(x(10:16)), ndviNameList, UniformOutput=false));

    [~, ndviQaNameList, ~] = cellfun(@(x) fileparts(x), ndviQaFullPathList, UniformOutput=false);
    ndviQaDateList = string(cellfun(@(x) yday2ymd(x(10:16)), ndviQaNameList, UniformOutput=false));

    if (length(ndviFullPathList) ~= length(ndviQaFullPathList)) || sum(ndviDateList ~= ...
            ndviQaDateList) > 0
        error('MODIS NDVI数据与QA数据不匹配, 请检查.')
    end

    % 筛选可用日期的数据-----------------------------------------------------------------------------
    % 创建年度日期列表.
    yearDateRange = datetime(sprintf('%d-01-01', yearNum)): datetime(sprintf('%d-12-31', yearNum));
    yearDateList = string(yearDateRange', 'yyyyMMdd');

    % 数据日期筛选.
    writelines("YearDate,PixelPercent,R", staRegionCsvPath);
    fprintf('筛选%s区%d年%s的数据.\n', region, yearNum, dayNight);
    for j = 1: length(yearDateList)
        yearDate = yearDateList(j);
        uselessDataPrompt = sprintf('%s 数据不满足要求.\n', yearDate);

        % 获取均有数据的AMSRE/2和MODIS地表温度文件路径, 跳过至少有其中一个没有数据的日期.
        [dateIndex1, dateLocate1] = ismember(yearDate, amsrDateList);
        [dateIndex2, dateLocate2] = ismember(yearDate, modisDateList);
        if dateIndex1 == 1 && dateIndex2 == 1
            amsrLstPath = amsrLstPathList{dateLocate1};
            modisLstPath = modisLstPathList{dateLocate2};
            modisQcPath = modisQcPathList{dateLocate2};
        else
            fprintf(uselessDataPrompt);
            continue
        end

        % 创建AMSRE/2和MODIS都有数据的日期文件夹.
        yearDateDir = fullfile(yearDir, sprintf('%s_%s_%s', region, yearDate, dayNight));
        if ~exist(yearDateDir, 'file')
            mkdir(yearDateDir)
        end

        % 裁剪研究区范围的AMSRE/2数据.
        [~, amsrLst, ext] = fileparts(amsrLstPath);
        amsrLstRegionName = replace([amsrLst, ext], dayNight, drStr);
        amsrLstRegionPath = fullfile(yearDateDir, amsrLstRegionName);
        if ~exist(amsrLstRegionPath, 'file')
            clipRaster(amsrLstPath, regionExtentPath, amsrLstRegionPath);
        end

        % 如果AMSRE/2地表温度影像中存在Nodata或0, 则舍弃当日的数据.
        amsrLstLayer = readgeoraster(amsrLstRegionPath);
        if sum(ismember(amsrLstLayer, amsrLstNodata), 'all') > 0
            rmdir(yearDateDir, 's')
            fprintf(uselessDataPrompt);
            continue
        end

        % 裁剪研究区MODIS数据, 计算云覆盖面积, 确定数据可用性. 若不可用, 删除当天AMSR2和MODIS数据.
        [~, modisLst, ext] = fileparts(modisLstPath);
        modisLstRegionName = replace([modisLst, ext], dayNight, [drStr, '_0.01']);
        modisLstRegionPath = fullfile(yearDateDir, modisLstRegionName);
        if ~exist(modisLstRegionPath, 'file')
            modisLstRegion2Path = replace(modisLstRegionPath, '.tif', '_2.tif');
            clipRaster(modisLstPath, regionExtentPath, modisLstRegion2Path);
            [modisLstLayer, modisLstRef] = readgeoraster(modisLstRegion2Path);
            geotiffwrite(modisLstRegionPath, single(modisLstLayer) * 0.02, modisLstRef)
            delete(modisLstRegion2Path)
        end

        [~, modisQc, ext] = fileparts(modisQcPath);
        modisQcRegionName = replace([modisQc, ext], dayNight, [drStr, '_0.01']);
        modisQcRegionPath = fullfile(yearDateDir, modisQcRegionName);
        if ~exist(modisQcRegionPath, 'file')
            clipRaster(modisQcPath, regionExtentPath, modisQcRegionPath)
        end

        [modisLstLayer, modisLstRef] = readgeoraster(modisLstRegionPath);
        modisLstLayer(ismember(modisLstLayer, modisLstNodata)) = 0;

        modisQcLayer = readgeoraster(modisQcRegionPath);
        qcBadIndexLayer = ismember(modisQcLayer, qcBad1) | (modisQcLayer >= qcBad2 & ...
            modisQcLayer ~= 256);

        modisLstLayer(qcBadIndexLayer) = 0;
        modisNodateLayer = (modisLstLayer == 0);
        modisNodataPct = sum(modisNodateLayer, 'all') / numel(modisNodateLayer);
        if modisNodataPct < cloudPctEdges(1) || modisNodataPct > cloudPctEdges(2)
            rmdir(yearDateDir, 's')
            fprintf(uselessDataPrompt);
            continue
        end

        modisLstLayer(modisLstLayer == 0) = nan;
        geotiffwrite(modisLstRegionPath, modisLstLayer, modisLstRef, CoordRefSysCode=4326);
        delete(modisQcRegionPath);

        % 创建存放重采样过程数据的临时文件夹.
        tempDir = fullfile(dataDir2, sprintf('Temp_%s_%s_%s', region, yearDate, dayNight));
        if ~exist(tempDir, 'dir')
            mkdir(tempDir)
        end

        % 若MODIS影像的Nodata像元数在指定范围内, 重采样生成MODIS LST分辨率序列.
        modisLstValidList = false(cellsizeN, 1);
        for k = 1: cellsizeN
            modisPath = replace(modisLstRegionPath, '0.01', cellsizeList(k));
            modisLstValidList(k) = exist(modisPath, 'file');
        end
        if ismember(false, modisLstValidList)
            modisLstBufferName = replace([modisLst, ext], dayNight, [dayNight, '_Buffer_0.01']);
            modisLstBufferPath = fullfile(tempDir, modisLstBufferName);
            clipRaster(modisLstPath, regionBufferPath, modisLstBufferPath); % 暂没加*0.02和SetNull.

            modisQcBufferName = replace([modisQc, ext], dayNight, [dayNight, '_Buffer_0.01']);
            modisQcBufferPath = fullfile(tempDir, modisQcBufferName);
            clipRaster(modisQcPath, regionBufferPath, modisQcBufferPath);

            % 保存质量控制后的MODIS LST.
            modisQcLayer = readgeoraster(modisQcBufferPath);
            qcBadIndexLayer = ismember(modisQcLayer, qcBad1) | (modisQcLayer >= qcBad2 & ...
                modisQcLayer ~= 256);

            [modisLstLayer, modisLstRef] = readgeoraster(modisLstBufferPath);
            modisLstLayer = single(modisLstLayer) * 0.02;
            modisLstLayer(qcBadIndexLayer) = nan;
            geotiffwrite(modisLstBufferPath, modisLstLayer, modisLstRef, CoordRefSysCode=4326);
            delete(modisQcBufferPath);

            % 重采样.
            resamplePixelSeries(modisLstBufferPath, regionExtentPath, yearDateDir, cellsizeList, ...
                region, 'initial', amsrLstRegionPath)
            delete(modisLstBufferPath);
        end

        % 准备NDVI数据.
        % 获取离MODIS LST最近的NDVI数据日期.
        yearDate2 = datetime(yearDate, InputFormat='yyyyMMdd');
        ndviDateDiff = abs(yearDate2 - datetime(ndviDateList, InputFormat='yyyyMMdd'));

        % NDVI数据裁剪与质量控制. 裁剪使用研究区缓冲区范围, 避免重采样时边界像元插值问题.
        ndviPath = ndviFullPathList{ndviDateDiff == min(ndviDateDiff)};
        [~, ndvi, ndviExt] = fileparts(ndviPath);
        ndviRegionName = replace([ndvi, ndviExt], 'NDVI', sprintf('NDVI_%s_0.01', region));
        ndviRegionPath = fullfile(yearDateDir, ndviRegionName);

        ndviRegionValidList = false(cellsizeN, 1);
        for k = 1: cellsizeN
            ndviRegionValidList(k) = exist(replace(ndviRegionPath, '0.01',cellsizeList(k)),'file');
        end
        if ismember(false, ndviRegionValidList)
            ndviBufferName = replace([ndvi, ndviExt], 'NDVI', 'NDVI_Buffer_0.01');
            ndviBufferPath = fullfile(tempDir, ndviBufferName);
            if ~exist(ndviBufferPath, 'file')
                % 裁剪NDVI数据.
                ndviClipPath = fullfile(tempDir, replace([ndvi, ndviExt],'NDVI','NDVI_Clip_0.01'));
                clipRaster(ndviPath, regionBufferPath, ndviClipPath);

                % 裁剪QA数据.
                ndviQaPath = ndviQaFullPathList(find(ndviDateDiff == min(ndviDateDiff), 1));
                [~, qaName, qaExt] = fileparts(ndviQaPath);
                ndviQaClipPath = fullfile(tempDir, replace([qaName, qaExt], 'QA', 'QA_Clip_0.01'));
                clipRaster(ndviQaPath, regionBufferPath, ndviQaClipPath);

                % 对NDVI进行质量控制, 并导出控制后的NDVI数据.
                [ndviQaLayer, ndviQaRef] = readgeoraster(ndviQaClipPath);
                [qaRowN, qaColN] = size(ndviQaLayer);
                ndviQaBinVector = dec2bin(ndviQaLayer, 16);
                ndviQaBinN = length(ndviQaBinVector);
                [ndviQaBinCutArray1, ndviQaBinCutArray2] = deal(strings(ndviQaBinN, 1));
                for m = 1: ndviQaBinN
                    ndviQaBin = ndviQaBinVector(m, :);
                    ndviQaBinCutArray1(m) = ndviQaBin(end - 1: end);
                    ndviQaBinCutArray2(m) = ndviQaBin(end - 5: end - 4);
                end
                ndviQaBinCutArray1 = reshape(ndviQaBinCutArray1, qaRowN, qaColN);
                ndviQaBinCutArray2 = reshape(ndviQaBinCutArray2, qaRowN, qaColN);
                ndviIndexLayer = ((ndviQaBinCutArray1 == '00') | (ndviQaBinCutArray1 == '01') | ...
                    (ndviQaBinCutArray2 ~= '11'));

                ndviClipLayer = readgeoraster(ndviClipPath) .* int16(ndviIndexLayer);
                geotiffwrite(ndviBufferPath, ndviClipLayer, ndviQaRef, CoordRefSysCode=4326);

                % 删除质量控制前的NDVI和QA Clip文件.
                delete(ndviClipPath);
                delete(ndviQaClipPath);
            end

            clipRaster(ndviBufferPath, regionExtentPath, ndviRegionPath);
            resamplePixelSeries(ndviBufferPath, regionExtentPath, yearDateDir, cellsizeList, ...
                region, 'initial', amsrLstRegionPath, nan)
            delete(ndviBufferPath)
        end
        rmdir(tempDir, 's')

        % 获取AMSR和MODIS地表温度数据的重叠区域, 并计算相关系数.
        modisLstLowestPath = replace(modisLstRegionPath, '0.01', cellsizeList(end));
        [modisLstLayer, amsrLstLayer] = intersectRaster(modisLstLowestPath, amsrLstRegionPath);

        validIndexLayer = ~isnan(modisLstLayer) & ~isnan(amsrLstLayer);
        modisLstVector = modisLstLayer(validIndexLayer);
        amsrLstVector = amsrLstLayer(validIndexLayer);
        cc = corrcoef(modisLstVector, amsrLstVector); cc = cc(2);
        fprintf("%s, %s, %s, Area: %.3f, R: %.3f.\n", region, yearDate, dayNight, ...
            1-modisNodataPct, cc);

        % 将统计数据写入CSV文件.
        record = sprintf("%s,%.3f,%.3f", yearDate, 1 - modisNodataPct, cc);
        writelines(record, staRegionCsvPath, WriteMode='append');
    end
end

% system('shutdown -s -t 60');