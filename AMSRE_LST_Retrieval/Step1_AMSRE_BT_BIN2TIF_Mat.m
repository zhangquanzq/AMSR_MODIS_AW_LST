%% 将AMSRE的二进制文件各通道亮温数据根据指定空间范围输出为TIF格式.
% 虽然此脚本输出的影像范围包含了全球, 但实际并不使用.

%% 标记和预设参数.
% 指定输出范围的标记. 1表示全球, 2表示中国.
flg1 = 2;
% 指定白天和晚上的标记. 1表示白天(升轨), 2表示晚上(降轨).
flg2 = 2;

% 原始AMSRE BT数据空间范围.
lonLimWorld = [-180, 180]; latLimWorld = [-90, 90]; sizeWorld = [720, 1440];

% 输出AMSR2影像的经纬度范围和空间参考.
latLim = {latLimWorld, [17, 55]};  % {[World], [CN]}
lonLim = {lonLimWorld, [70, 136]};  % {[World], [CN]}
latLim = latLim{flg1}; lonLim = lonLim{flg1}; cellsizeX = 0.25; cellsizeY = 0.25;

startRow = 1 + (latLimWorld(2) - latLim(2)) / cellsizeY;
endRow = sizeWorld(1) - (latLim(1) - latLimWorld(1)) / cellsizeY;
startCol = 1 + (lonLim(1) - lonLimWorld(1)) / cellsizeX;
endCol = sizeWorld(2) - (lonLimWorld(2) - lonLim(2)) / cellsizeX;
amsreRowN = endRow - startRow + 1; amsreColN = endCol - startCol + 1;
amsreRef = georefcells(latLim, lonLim, [amsreRowN, amsreColN], ColumnsStartFrom='north');

% AMSRE数据的通道与极化.
channelList = ["10", "18", "23", "36", "6", "89"];
polarize = ["H", "V"];
[cMatrix, pMatrix] = meshgrid(channelList, polarize);
cpList = cellstr(reshape(cMatrix + pMatrix, [], 1));
cpN = length(cpList);

% AMSR2数据的范围, 轨道和昼夜.
region = {'World', 'CN'};
region = region{flg1};

orbit = {'A', 'D'};
orbit = orbit{flg2};

daynight = {'Day', 'Night'};
daynight = daynight{flg2};

% 有数据的时间区间：2003/01/01-2011/09/27.
startDate = [2003, 01, 01]; endDate = [2011, 09, 27];
dateAllList = cellstr(datetime(startDate): datetime(endDate), 'yyyyMMdd')';
yearNumList = startDate(1): endDate(1);

%% 路径.
% 根目录.
rootDir = 'J:\AMSRE_MODIS_AW_LST';
dataDir = fullfile(rootDir, 'AMSRE_LST_Retrieval\Data');
addpath(fullfile(rootDir, 'Code\Functions'));

% 输入数据路径.
amsreBinDir = fullfile(dataDir, 'AMSRE_1_BIN');

% 输出数据路径.
amsreTifDir = fullfile(dataDir, sprintf('AMSRE_2_%s_TIF', region));
if ~exist(amsreTifDir, 'dir')
    mkdir(amsreTifDir);
end

amsreMatDir = fullfile(dataDir, sprintf('AMSRE_2_%s_Matlab', region));
if ~exist(amsreMatDir, 'dir')
    mkdir(amsreMatDir);
end

%% 将AMSRE 二进制格式转为TIF格式.
disp('将AMSR2 H5格式转为TIF格式.');
amsreYearFolderList = {dir(fullfile(amsreBinDir, 'AMSRE*')).name}';
for i = 1: length(amsreYearFolderList)
%     break
    amsreYearFolder = amsreYearFolderList{i};
    amsreBinYearPath = fullfile(amsreBinDir, amsreYearFolder);
    amsreBinList = {dir(fullfile(amsreBinYearPath, 'ID2r1*')).name}';

    % 创建存储指定空间范围AMSR BT文件的文件夹.
    amsreRegionYearDir = fullfile(amsreTifDir, replace(amsreYearFolder, 'XXX', 'XXXX'));
    if ~exist(amsreRegionYearDir, 'dir')
        mkdir(amsreRegionYearDir);
    end

    % 读取AMSRE的二进制文件, 并将指定空间范围的数据输出为TIF格式.
    for j = 1: length(amsreBinList)
        amsreBin = amsreBinList{j};
        amsreDate = yday2ymd(amsreBin(17:23));

        amsreRegionDateDir = fullfile(amsreRegionYearDir, sprintf('AMSRE_%s', amsreDate));
        if ~exist(amsreRegionDateDir, 'dir')
            mkdir(amsreRegionDateDir)
            fprintf('输出%s %s的AMSRE BT数据.\n', amsreDate, region)
        end

        tailName = replace(amsreBin(24:end), '.', '_');
        amsreTif = sprintf('AMSRE_D25_%s%s.tif', amsreDate, tailName);
        amsreTifPath = fullfile(amsreRegionDateDir, amsreTif);
        if exist(amsreTifPath, 'file')
            continue
        end

        fileID = fopen(fullfile(amsreBinYearPath, amsreBin));
        amsreArray = uint16(fread(fileID, [1440, 720], 'uint16'))';
        amsreArray = amsreArray(startRow: endRow, startCol: endCol);
        geotiffwrite(amsreTifPath, amsreArray, amsreRef, CoordRefSysCode=4326);
        fclose(fileID);
    end
end

%% 将AMSR2 BT数据从TIF格式转为Mat格式.
fprintf('获取%s所有年份AMSRE TIF文件的路径列表.\n', region);
amsreDailyPathAllList = cell(length(dateAllList) * cpN, 1); % 每日6个波段, 2个极化.
amsreRegionYearFolderList = {dir(fullfile(amsreTifDir, 'AMSRE*')).name}';
startIndex = 1;
for i = 1: length(amsreRegionYearFolderList)    
    amsreYearDir = fullfile(amsreTifDir, amsreRegionYearFolderList{i});
    amsreDailyFolderList = {dir(fullfile(amsreYearDir, 'AMSRE*')).name}';
    for j = 1: length(amsreDailyFolderList)
        amsreDailyDir = fullfile(amsreYearDir, amsreDailyFolderList{j});
        amsreDailyList = {dir(fullfile(amsreDailyDir, sprintf('AMSRE*%s_v*.tif', orbit))).name}';
        amsreDailyList(contains(amsreDailyList, 'TIM')) = [];
        amsreDailyPath = fullfile(amsreDailyDir, amsreDailyList);
        endIndex = startIndex + length(amsreDailyList) - 1;
        amsreDailyPathAllList(startIndex: endIndex) = amsreDailyPath;
        startIndex = endIndex + 1;
    end
end
amsreDailyPathAllList(cellfun(@isempty, amsreDailyPathAllList)) = [];

% 读取AMSRE BT数据, 并存储为Mat格式.
% [~, amsreRef] = readgeoraster(amsreDailyPathAllList{1});
% amsreRowN = amsreRef.RasterSize(1); amsreColN = amsreRef.RasterSize(2);
for i = 1: length(yearNumList)
    yearStr = num2str(yearNumList(i));

    % 检查是否已存在当年的AMSRE BT mat格式文件.
    amsreYearMatName = sprintf('AMSRE_BT_%s_%s_%s.mat', region, yearStr, daynight);
    amsreYearMatPath = fullfile(amsreMatDir, amsreYearMatName);
    if exist(amsreYearMatPath, 'file')
        continue
    end

    % 索引当年的AMSRE亮温数据存储路径, 并分极化方式与波段存储为mat格式文件.
    yearIndex1 = contains(amsreDailyPathAllList, [yearStr, 'XXXX']);
    yearIndex2 = strcmp(extractBefore(dateAllList, 5), yearStr);
    % 极化排序：[H, V]，通道排序：[10, 18, 23, 36, 6, 89]
    amsreDailyPathYearList = amsreDailyPathAllList(yearIndex1);
    dateYearList = dateAllList(yearIndex2);
    dateYearN = length(dateYearList);

    % 创建存储各年度的每日AMSRE BT数据矩阵.
    for j = 1: cpN
        assignin('base', sprintf('amsre%sYearArray', cpList{j}), ...
            zeros(amsreRowN, amsreColN, dateYearN, 'uint16'))
    end
    % 创建存储每日AMSRE BT数据路径字符串的矩阵.
    amsrePathMatrix = cell(dateYearN + 1, cpN);
    amsrePathMatrix(1, :) = cpList'; % 第一行存极化通道名称, 其他行存各日期文件路径.

    emptyRowIndex = false(dateYearN + 1, 1);
    for j = 1: dateYearN
        dateYear = dateYearList{j};
        dateYearIndex = contains(amsreDailyPathYearList, dateYear);
        amsreDailyPathList = amsreDailyPathYearList(dateYearIndex);
        amsreDailyPathN = length(amsreDailyPathList);
        if amsreDailyPathN ~= cpN  % cpListN == 12
            emptyRowIndex(j + 1) = true;
            fprintf('%s %s的AMSRE BT数据有缺失.\n', dateYear, daynight);
            continue
        end

        fprintf('读取%s %s %s的AMSRE BT数据.\n', region, dateYear, daynight);
        amsreDailyArray = zeros(amsreRowN, amsreColN, amsreDailyPathN, 'uint16');
        for k = 1: amsreDailyPathN
            amsreDailyArray(:, :, k) = readgeoraster(amsreDailyPathList{k});
            amsrePathMatrix(j+1, k) = amsreDailyPathList(k);
        end
        for k = 1: cpN
            evalin('base', sprintf('amsre%sYearArray(:,:,j) = amsreDailyArray(:,:,k);', cpList{k}))
        end
    end

    % 删除数据缺失日期行.
    amsrePathMatrix(emptyRowIndex, :) = [];
    emptyRowIndex = emptyRowIndex(2:end);
    dateYearList(emptyRowIndex) = [];
    for j = 1: cpN
        evalin('base', sprintf('amsre%sYearArray(:, :, emptyRowIndex) = [];', cpList{j}))
    end

    fprintf('保存%s %s年%s的AMSRE BT数据.\n', region, yearStr, daynight);
    save(amsreYearMatPath, 'amsreRef', 'dateYearList', 'amsre*YearArray', 'amsrePathMatrix');
end
