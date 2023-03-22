%% 将投影转换后的AMSR2 BT数据从TIF格式转为Mat格式.
% 虽然此脚本输出的影像范围包含了全球, 但实际并不使用.

%% 标记和预设参数.
% 指定输出范围的标记. 1表示全球, 2表示南极, 3表示格陵兰.
flg1 = 3;
% 指定白天和晚上的标记. 1表示白天(升轨), 2表示晚上(降轨).
flg2 = 2;

% AMSR2数据的范围, 轨道和昼夜.
region = {'World', 'Antarctic', 'Greenland'};
region = region{flg1};

orbit = {'A', 'D'};
orbit = orbit{flg2};

daynight = {'Day', 'Night'};
daynight = daynight{flg2};

% AMSR2数据的通道与极化.
polarize = ["H", "V"];
channelList = ["10", "18", "23", "36", "06", "07", "89"];
[pMatrix, cMatrix] = meshgrid(polarize, channelList);
cpList = cellstr(reshape((pMatrix + cMatrix)', 1, []));
cpN = length(cpList);

% 有数据的时间区间：2012/07/02-2020/12/31.
startDate = [2012, 07, 02]; endDate = [2020, 12, 31];
dateAllList = cellstr(datetime(startDate): datetime(endDate), 'yyyyMMdd')';
yearNumList = startDate(1): endDate(1);

%% 路径.
% 根目录.
dataDir = 'F:\AMSR_LST_IceSheet\Data';

% 输入数据路径.
amsr2RegionDir = fullfile(dataDir, sprintf('AMSR2_2_BT_Prj%s_TIF', region));

% 输出数据路径.
amsr2MatDir = fullfile(dataDir, sprintf('AMSR2_2_BT_Prj%s_Matlab', region));
if ~exist(amsr2MatDir, 'dir')
    mkdir(amsr2MatDir);
end

%% 将AMSR2 BT数据从TIF格式转为Mat格式.
fprintf('获取%s所有年份AMSR2 TIF文件的路径列表.\n', region);
amsr2DailyPathAllList = cell(length(dateAllList) * cpN, 1); % 每日7个波段, 2各极化.
amsr2RegionYearFolderList = {dir(fullfile(amsr2RegionDir, 'AMSR2*TIF')).name}';
startIndex = 1;
for i = 1: length(amsr2RegionYearFolderList)    
    amsr2YearDir = fullfile(amsr2RegionDir, amsr2RegionYearFolderList{i});
    amsr2DailyFolderList = {dir(fullfile(amsr2YearDir, 'AMSR2*')).name}';
    for j = 1: length(amsr2DailyFolderList)
        amsr2DailyDir = fullfile(amsr2YearDir, amsr2DailyFolderList{j});
        amsr2DailyList = {dir(fullfile(amsr2DailyDir, sprintf('*EQM%s*.tif', orbit))).name}';
        amsr2DailyPath = fullfile(amsr2DailyDir, amsr2DailyList);
        endIndex = startIndex + length(amsr2DailyList) - 1;
        amsr2DailyPathAllList(startIndex: endIndex) = amsr2DailyPath;
        startIndex = endIndex + 1;
    end
end
amsr2DailyPathAllList(cellfun(@isempty, amsr2DailyPathAllList)) = [];

% 读取AMSR2 BT数据, 并存储为Mat格式.
[~, amsr2Ref] = readgeoraster(amsr2DailyPathAllList{1});
amsr2RowN = amsr2Ref.RasterSize(1); amsr2ColN = amsr2Ref.RasterSize(2);
for i = 1: length(yearNumList)
    yearStr = num2str(yearNumList(i));

    % 检查是否已存在当年的AMSR2 BT mat格式文件.
    amsr2YearMatName = sprintf('AMSR2_BT_%s_%s_%s.mat', region, yearStr, daynight);
    amsr2YearMatPath = fullfile(amsr2MatDir, amsr2YearMatName);
    if exist(amsr2YearMatPath, 'file')
        continue
    end

    % 索引当年的AMSR2亮温数据存储路径, 并分极化方式与波段存储为mat格式文件.
    yearIndex1 = contains(amsr2DailyPathAllList, [yearStr, 'XXXX']);
    yearIndex2 = strcmp(extractBefore(dateAllList, 5), yearStr);
    % 极化排序：[H, V]，通道排序：[10, 18, 23, 36, 6, 7, 89]
    amsr2DailyPathYearList = amsr2DailyPathAllList(yearIndex1);
    dateYearList = dateAllList(yearIndex2);
    dateYearN = length(dateYearList);

    % 创建存储各年度的每日AMSR2 BT数据矩阵.
    for j = 1: cpN
        assignin('base', sprintf('amsr2%sYearArray', cpList{j}), ...
            zeros(amsr2RowN, amsr2ColN, dateYearN, 'uint16'))
    end
    % 创建存储每日AMSR2 BT数据路径字符串的矩阵.
    amsr2PathMatrix = cell(dateYearN + 1, cpN);
    amsr2PathMatrix(1, :) = cpList'; % 第一行存极化通道名称, 其他行存各日期文件路径.

    emptyRowIndex = false(dateYearN + 1, 1);
    for j = 1: dateYearN
        dateYear = dateYearList{j};
        dateYearIndex = contains(amsr2DailyPathYearList, dateYear);
        amsr2DailyPathList = amsr2DailyPathYearList(dateYearIndex);
        amsr2DailyPathN = length(amsr2DailyPathList);
        if amsr2DailyPathN ~= cpN  % cpListN == 14
            emptyRowIndex(j + 1) = true;
            fprintf('%s %s的AMSR2 BT数据有缺失, 请检查.\n', dateYear, daynight);
            continue
        end

        fprintf('读取%s %s %s的AMSR2 BT数据.\n', region, dateYear, daynight);
        amsr2DailyArray = zeros(amsr2RowN, amsr2ColN, amsr2DailyPathN, 'uint16');
        for k = 1: amsr2DailyPathN
            amsr2DailyArray(:, :, k) = readgeoraster(amsr2DailyPathList{k});
            amsr2PathMatrix(j+1, k) = amsr2DailyPathList(k);
        end
        for k = 1: cpN
            evalin('base', sprintf('amsr2%sYearArray(:,:,j) = amsr2DailyArray(:,:,k);', cpList{k}))
        end
    end

    % 删除数据缺失日期行.
    amsr2PathMatrix(emptyRowIndex, :) = [];
    emptyRowIndex = emptyRowIndex(2:end);
    dateYearList(emptyRowIndex) = [];
    for j = 1: cpN
        evalin('base', sprintf('amsr2%sYearArray(:, :, emptyRowIndex) = [];', cpList{j}))
    end

    fprintf('保存%s %s年%s的AMSR2 BT数据.\n', region, yearStr, daynight);
    save(amsr2YearMatPath, 'amsr2Ref', 'dateYearList', 'amsr2*YearArray', 'amsr2PathMatrix');
end

% system('shutdown -s -t 60')


