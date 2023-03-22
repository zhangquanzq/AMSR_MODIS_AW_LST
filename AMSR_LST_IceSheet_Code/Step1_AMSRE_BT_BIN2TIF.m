%% 将AMSRE的二进制文件各通道亮温数据根据指定空间范围输出为TIF格式.
% 虽然此脚本输出的影像范围包含了全球, 但实际并不使用.

%% 标记和预设参数.
% 指定输出范围的标记. 1表示全球, 2表示南极, 3表示格陵兰.
flg1 = 3;

% AMSRE数据范围.
region = {'World', 'Antarctic', 'Greenland'};
region = region{flg1};

% 原始AMSRE BT数据空间范围.
lonLimWorld = [-180, 180]; latLimWorld = [-90, 90]; sizeWorld = [720, 1440];

% 输出AMSR2影像的经纬度范围和空间参考.
latLim = {latLimWorld, [-90, -50], [58, 84]};  % {[World], [Antarctic], [Greenland]}
lonLim = {lonLimWorld, [-180, 180], [-74, -9]};  % {[World], [Antarctic], [Greenland]}
latLim = latLim{flg1}; lonLim = lonLim{flg1}; cellsizeX = 0.25; cellsizeY = 0.25;

startRow = 1 + (latLimWorld(2) - latLim(2)) / cellsizeY;
endRow = sizeWorld(1) - (latLim(1) - latLimWorld(1)) / cellsizeY;
startCol = 1 + (lonLim(1) - lonLimWorld(1)) / cellsizeX;
endCol = sizeWorld(2) - (lonLimWorld(2) - lonLim(2)) / cellsizeX;
arraySize = [endRow - startRow + 1, endCol - startCol + 1];

amsreRef = georefcells(latLim, lonLim, arraySize, 'ColumnsStartFrom', 'north');

% 年份范围: 2003到2011.

%% 路径.
% 根目录.
rootDir = 'E:\AMSRE_LST_IceSheet';
dataDir = fullfile(rootDir, 'Data');
addpath(fullfile(rootDir, 'Code\Functions'));

% 输入数据路径.
amsreBinDir = fullfile(dataDir, 'AMSRE_1_BT_World_Bin');

% 输出数据路径.
amsreRegionDir = fullfile(dataDir, sprintf('AMSRE_2_BT_%s_TIF', region));
if ~exist(amsreRegionDir, 'dir')
    mkdir(amsreRegionDir);
end

%% 数据格式转换.
amsreBinYearFolderList = dir(fullfile(amsreBinDir, 'AMSRE*'));
amsreBinYearFolderList = {amsreBinYearFolderList.name}';
for i = 1: length(amsreBinYearFolderList)
    amsreBinYearFolder = amsreBinYearFolderList{i};
    amsreBinYearPath = fullfile(amsreBinDir, amsreBinYearFolder);
    amsreBinNameList = dir(fullfile(amsreBinYearPath, 'ID2r1*'));
    amsreBinNameList = {amsreBinNameList.name}';

    % 创建存储指定空间范围AMSR BT文件的文件夹.
    amsreRegionYearDir = fullfile(amsreRegionDir, replace(amsreBinYearFolder, 'XXX', 'XXXX'));
    if ~exist(amsreRegionYearDir, 'dir')
        mkdir(amsreRegionYearDir);
    end

    % 读取AMSRE的二进制文件, 并将指定空间范围的数据输出为TIF格式.
    for j = 1: length(amsreBinNameList)
        amsreBinName = amsreBinNameList{j};
        amsreDate = yday2ymd(amsreBinName(17:23));

        amsreRegionDateDir = fullfile(amsreRegionYearDir, sprintf('AMSRE_%s', amsreDate));
        if ~exist(amsreRegionDateDir, 'dir')
            fprintf('输出%s %s的AMSRE BT数据.\n', amsreDate, region)
            mkdir(amsreRegionDateDir)
        end

        tailName = replace(amsreBinName(24:end), '.', '_');
        amsreTifName = sprintf('AMSRE_D25_%s%s.tif', amsreDate, tailName);
        amsreTifPath = fullfile(amsreRegionDateDir, amsreTifName);
        if exist(amsreTifPath, 'file')
            continue
        end

        fileID = fopen(fullfile(amsreBinYearPath, amsreBinName));
        amsreArray = uint16(fread(fileID, [1440, 720], 'uint16'))';
        amsreArray = amsreArray(startRow:endRow, startCol:endCol);
        geotiffwrite(amsreTifPath, amsreArray, amsreRef, 'CoordRefSysCode', 4326);
        fclose(fileID);
    end
end
