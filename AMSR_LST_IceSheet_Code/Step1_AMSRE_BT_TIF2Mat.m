%% 将投影转换后的AMSRE BT数据从TIF格式转为Mat格式.
% 虽然此脚本输出的影像范围包含了全球, 但实际并不使用.

%% 标记和预设参数.
% 指定输出范围的标记. 1表示全球, 2表示南极, 3表示格陵兰.
flg1 = 3;
% 指定白天和晚上的标记. 1表示白天(升轨), 2表示晚上(降轨).
flg2 = 1;

% AMSR2数据的范围, 轨道和昼夜.
region = {'World', 'Antarctic', 'Greenland'};
region = region{flg1};

orbit = {'A', 'D'};
orbit = orbit{flg2};

daynight = {'Day', 'Night'};
daynight = daynight{flg2};

% AMSR2数据的通道与极化.
polarize = ["H", "V"];
channelList = ["10", "18", "23", "36", "06", "89"];
[pMatrix, cMatrix] = meshgrid(polarize, channelList);
cpList = cellstr(reshape((pMatrix + cMatrix)', 1, []));
cpN = length(cpList);

% 有数据的时间区间：2003/01/01-2011/09/27.
startDate = [2003, 01, 01]; endDate = [2011, 09, 27];
dateAllList = cellstr(datetime(startDate): datetime(endDate), 'yyyyMMdd')';
yearNumList = startDate(1): endDate(1);

%% 路径.
% 根目录.
dataDir = 'H:\AMSR_LST_IceSheet\Data';

% 输入数据路径.
amsreRegionDir = fullfile(dataDir, sprintf('AMSRE_2_BT_Prj%s_TIF', region));

% 输出数据路径.
amsreMatDir = fullfile(dataDir, sprintf('AMSRE_2_BT_Prj%s_Matlab', region));
if ~exist(amsreMatDir, 'dir')
    mkdir(amsreMatDir);
end

%% 数据处理过程.
fprintf('获取%s所有年份AMSRE TIF文件的路径列表.\n', region);
amsreDailyPathAllList = cell(length(dateAllList) * cpN, 1); % 每日6个波段, 2个极化.
amsreRegionYearFolderList = {dir(fullfile(amsreRegionDir, 'AMSRE*')).name}';
startIndex = 1;
for i = 1: length(amsreRegionYearFolderList)    
    amsreYearDir = fullfile(amsreRegionDir, amsreRegionYearFolderList{i});
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
[~, amsreRef] = readgeoraster(amsreDailyPathAllList{1});
amsreRowN = amsreRef.RasterSize(1); amsreColN = amsreRef.RasterSize(2);
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
    % 极化排序：[H, V]，通道排序：[10, 18, 23, 36, 6, 7, 89]
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
            fprintf('%s %s的AMSRE BT数据有缺失, 请检查.\n', dateYear, daynight);
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

% system('shutdown -s -t 60')


