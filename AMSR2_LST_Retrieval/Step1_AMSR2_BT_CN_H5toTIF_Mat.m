%% 将AMSR2 H5格式的各通道亮温数据根据指定空间范围输出为TIF格式, 并保存为Mat格式.
% 虽然此程序可以输出整个全球范围的影像, 但实际并没有用到.

%% 标记和预设参数.
% 指定输出范围的标记. 1表示全球, 2表示中国(包含高亚洲地区).
flg1 = 2;
% 指定白天和晚上的标记. 1表示白天(升轨), 2表示晚上(降轨).
flg2 = 2;

% 输出AMSR2影像的经纬度范围和空间参考.
latLim = {[-90, 90], [17, 55]}; lonLim = {[0, 360], [66, 136]};  % {[World], [CN]}
latLim = latLim{flg1}; lonLim = lonLim{flg1}; cellsize = 0.1;

startLon = (lonLim(1) - 0) / cellsize + 1; endLon = (lonLim(2) - 0) / cellsize;
startLat = (90 - latLim(2)) / cellsize + 1; endLat = (90 - latLim(1)) / cellsize;
amsr2Ref = georefcells(latLim, lonLim, cellsize, cellsize, 'ColumnsStartFrom', 'north');

% AMSR2数据的通道与极化.
channelList = ["10", "18", "23", "36", "6", "7", "89"];
polarize = ["H", "V"];
[cMatrix, pMatrix] = meshgrid(channelList, polarize);
cpList = cellstr(reshape(cMatrix + pMatrix, [], 1));
cpN = length(cpList);

% AMSR2数据的范围, 轨道和昼夜.
extent = {'World', 'CN'};
extent = extent{flg1};

orbit = {'A', 'D'};
orbit = orbit{flg2};

daynight = {'Day', 'Night'};
daynight = daynight{flg2};

% 有数据的年份月份列表(时间区间：2012/07/02-2021/12/31).
startDate = [2012, 07, 02]; endDate = [2021, 12, 31];
dateAllList = cellstr(datetime(startDate) : datetime(endDate), 'yyyyMMdd')';
yearNumList = startDate(1) : endDate(1);

%% 目录.
% 根目录.
dataDir = 'E:\AMSR2_MODIS_AW_LST\AMSR2_LST_Retrieval\Data';

% 输入数据路径.
amsr2H5Dir = fullfile(dataDir, 'AMSR2_1_H5');

% 输出路径, AMSR2 BT的TIF, Mat格式数据.
amsr2TifDir = fullfile(dataDir, sprintf('AMSR2_2_%s_TIF', extent));
if ~exist(amsr2TifDir, 'dir')
    mkdir(amsr2TifDir)
end

amsr2MatDir = fullfile(dataDir, sprintf('AMSR2_2_%s_Matlab', extent));
if ~exist(amsr2MatDir, 'dir')
    mkdir(amsr2MatDir);
end

%% 将AMSR2 H5格式转为TIF格式.
% 读取AMSR2 H5波段文件夹列表.
disp('将AMSR2 H5格式转为TIF格式.');
amsr2BandFolderList = dir(fullfile(amsr2H5Dir, 'L3*'));
amsr2BandFolderList = {amsr2BandFolderList.name}';
for i = 1: length(amsr2BandFolderList)
    amsr2BandFolder = amsr2BandFolderList{i};

    % 创建AMSR2 TIF波段文件夹.
    amsr2TifBandDir = fullfile(amsr2TifDir, amsr2BandFolder);
    if ~exist(amsr2TifBandDir, 'dir')
        mkdir(amsr2TifBandDir)
    end

    % 读取AMSR2 H5年份文件夹列表.
    amsr2H5BandDir = fullfile(amsr2H5Dir, amsr2BandFolder);
    amsr2YearFolderList = dir(amsr2H5BandDir);
    amsr2YearFolderList = {amsr2YearFolderList(3:end).name}';
    for j = 1: length(amsr2YearFolderList)
        amsr2YearFolder = amsr2YearFolderList{j};

        % 创建AMSR2 TIF年份文件夹.
        amsr2TifYearDir = fullfile(amsr2TifBandDir, amsr2YearFolder);
        if ~exist(amsr2TifYearDir, 'dir')
            mkdir(amsr2TifYearDir)
        end

        % 读取AMSR2 H5月份文件夹列表.
        amsr2H5YearDir = fullfile(amsr2H5BandDir, amsr2YearFolder);
        amsr2MonthFolderList = dir(amsr2H5YearDir);
        amsr2MonthFolderList = {amsr2MonthFolderList(3:end).name}';
        for k = 1: length(amsr2MonthFolderList)
            amsr2MonthFolder = amsr2MonthFolderList{k};

            % 创建AMSR2 TIF月份文件夹.
            amsr2TifMonthDir = fullfile(amsr2TifYearDir, amsr2MonthFolder);
            if ~exist(amsr2TifMonthDir, 'dir')
                mkdir(amsr2TifMonthDir)
            end
            fprintf('转换%s年%s月%s AMSR2 BT %s数据的格式.\n', amsr2YearFolder, amsr2MonthFolder, ...
                daynight, amsr2BandFolder(6:10))

            % 读取AMSR2 H5每日文件列表.
            amsr2H5MonthDir = fullfile(amsr2H5YearDir, amsr2MonthFolder);
            amsr2H5DailyList = dir(fullfile(amsr2H5MonthDir, '*.h5'));
            amsr2H5DailyList = {amsr2H5DailyList.name}';
            for m = 1: length(amsr2H5DailyList)
                amsr2H5DailyName = amsr2H5DailyList{m};
                amsr2H5DailyPath = fullfile(amsr2H5MonthDir, amsr2H5DailyName);

                % 保存H5格式中的极化亮温数据为TIF格式.
                for n = 1: length(polarize)
                    replacedStr = sprintf('_Bt%s.tif', polarize(n));
                    amsr2BtTifName = replace(amsr2H5DailyName, '.h5', replacedStr);
                    amsr2BtTifPath = fullfile(amsr2TifMonthDir, amsr2BtTifName);
                    if ~exist(amsr2BtTifPath, 'file')
                        layerName = sprintf('/Brightness Temperature (%s)', polarize(n));
                        try
                            btArray = h5read(amsr2H5DailyPath, layerName)';
                            btArray = btArray(startLat: endLat, startLon: endLon);
                            geotiffwrite(amsr2BtTifPath, btArray, amsr2Ref, CoordRefSysCode=4326);
                        catch
                            disp(['有问题数据：', amsr2BtTifName]);
                        end
                    end
                end
            end
        end
    end
end

%% 将AMSR2 BT数据从TIF格式转为Mat格式.
% 获取所有年份AMSR2 TIF文件的路径列表.
disp('获取所有年份AMSR2 TIF文件的路径列表.');
amsr2DailyPathAllList = cell(length(dateAllList) * cpN, 1); % 每日7个波段, 2各极化.
amsr2TifDir = fullfile(dataDir, 'AMSR2_2_CN_TIF');
amsr2ChannelList = dir(fullfile(amsr2TifDir, 'L3*'));
amsr2ChannelList = {amsr2ChannelList.name}';
startIndex = 1;
for i = 1: length(amsr2ChannelList)
    amsr2ChannelDir = fullfile(amsr2TifDir, amsr2ChannelList{i});
    amsr2YearList = dir(fullfile(amsr2ChannelDir));
    amsr2YearList = {amsr2YearList(3:end).name}'; % 排除 '.', '..'两个文件夹.
    for j = 1: length(amsr2YearList)
        amsr2YearDir = fullfile(amsr2ChannelDir, amsr2YearList{j});
        amsr2MonthList = dir(amsr2YearDir);
        amsr2MonthList = {amsr2MonthList(3:end).name}'; % 排除 '.', '..'两个文件夹.
        for k = 1: length(amsr2MonthList)
            amsr2MonthDir = fullfile(amsr2YearDir, amsr2MonthList{k});
            amsr2DailyList = dir(fullfile(amsr2MonthDir, sprintf('*01D_EQM%s*.tif', orbit)));
            amsr2DailyList = {amsr2DailyList.name}';
            amsr2DailyPath = fullfile(amsr2MonthDir, amsr2DailyList);
            endIndex = startIndex + length(amsr2DailyList) - 1;
            amsr2DailyPathAllList(startIndex: endIndex) = amsr2DailyPath;
            startIndex = endIndex + 1;
        end
    end
end
amsr2DailyPathAllList(cellfun(@isempty, amsr2DailyPathAllList)) = [];

% 读取AMSR2 BT数据, 并存储为Mat格式.
disp('按年份, 通道, 极化和昼夜读取AMSR2亮温数据, 并存储为mat格式.');
amsr2RowN = amsr2Ref.RasterSize(1); amsr2ColN = amsr2Ref.RasterSize(2);
for i = 1: length(yearNumList)
    yearStr = num2str(yearNumList(i));

    % 检查是否已存在当年的AMSR2 BT mat格式文件。
    amsr2YearMatName = sprintf('AMSR2_BT_%s_%s.mat', yearStr, daynight);
    amsr2YearMatPath = fullfile(amsr2MatDir, amsr2YearMatName);
    if exist(amsr2YearMatPath, 'file')
        continue
    end

    % 索引当年的AMSR2亮温数据存储路径, 并分极化方式与波段存储为mat格式文件.
    yearIndex1 = contains(amsr2DailyPathAllList, [yearStr, '\']);
    yearIndex2 = strcmp(extractBefore(dateAllList, 5), yearStr);
    % 极化排序：[H, V]，通道排序：[10, 18, 23, 36, 6, 7, 89]
    amsr2DailyPathYearList = amsr2DailyPathAllList(yearIndex1);
    dateYearList = dateAllList(yearIndex2);
    dateYearN = length(dateYearList);
    [amsr2H06YearArray, amsr2V06YearArray, amsr2H07YearArray, amsr2V07YearArray, ...
        amsr2H10YearArray, amsr2V10YearArray, amsr2H18YearArray, amsr2V18YearArray, ...
        amsr2H23YearArray, amsr2V23YearArray, amsr2H36YearArray, amsr2V36YearArray, ...
        amsr2H89YearArray, amsr2V89YearArray] = ...
        deal(zeros(amsr2RowN, amsr2ColN, dateYearN, 'uint16'));
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
            fprintf('  %s %s的AMSR2亮温数据有缺失, 请检查.\n', dateYear, daynight);
            continue
        end

        fprintf('  读取%s %s的AMSR2亮温数据.\n', dateYear, daynight);
        amsr2DailyArray = zeros(amsr2RowN, amsr2ColN, amsr2DailyPathN, 'uint16');
        for k = 1: amsr2DailyPathN
            amsr2DailyArray(:, :, k) = readgeoraster(amsr2DailyPathList{k});
            amsr2PathMatrix(j+1, k) = amsr2DailyPathList(k);
        end
        amsr2H10YearArray(:, :, j) = amsr2DailyArray(:, :, 1);
        amsr2V10YearArray(:, :, j) = amsr2DailyArray(:, :, 2);
        amsr2H18YearArray(:, :, j) = amsr2DailyArray(:, :, 3);
        amsr2V18YearArray(:, :, j) = amsr2DailyArray(:, :, 4);
        amsr2H23YearArray(:, :, j) = amsr2DailyArray(:, :, 5);
        amsr2V23YearArray(:, :, j) = amsr2DailyArray(:, :, 6);
        amsr2H36YearArray(:, :, j) = amsr2DailyArray(:, :, 7);
        amsr2V36YearArray(:, :, j) = amsr2DailyArray(:, :, 8);
        amsr2H06YearArray(:, :, j) = amsr2DailyArray(:, :, 9);
        amsr2V06YearArray(:, :, j) = amsr2DailyArray(:, :, 10);
        amsr2H07YearArray(:, :, j) = amsr2DailyArray(:, :, 11);
        amsr2V07YearArray(:, :, j) = amsr2DailyArray(:, :, 12);
        amsr2H89YearArray(:, :, j) = amsr2DailyArray(:, :, 13);
        amsr2V89YearArray(:, :, j) = amsr2DailyArray(:, :, 14);
    end

    % 删除数据缺失日期行.
    amsr2PathMatrix(emptyRowIndex, :) = [];
    emptyRowIndex = emptyRowIndex(2:end); dateYearList(emptyRowIndex) = [];
    amsr2H10YearArray(:, :, emptyRowIndex) = []; amsr2V10YearArray(:, :, emptyRowIndex) = [];
    amsr2H18YearArray(:, :, emptyRowIndex) = []; amsr2V18YearArray(:, :, emptyRowIndex) = [];
    amsr2H23YearArray(:, :, emptyRowIndex) = []; amsr2V23YearArray(:, :, emptyRowIndex) = [];
    amsr2H36YearArray(:, :, emptyRowIndex) = []; amsr2V36YearArray(:, :, emptyRowIndex) = [];
    amsr2H06YearArray(:, :, emptyRowIndex) = []; amsr2V06YearArray(:, :, emptyRowIndex) = [];
    amsr2H07YearArray(:, :, emptyRowIndex) = []; amsr2V07YearArray(:, :, emptyRowIndex) = [];
    amsr2H89YearArray(:, :, emptyRowIndex) = []; amsr2V89YearArray(:, :, emptyRowIndex) = [];

    fprintf('  保存%s年%s的AMSR2亮温数据.\n', yearStr, daynight);
    save(amsr2YearMatPath, 'amsr2Ref', 'dateYearList', 'amsr2*YearArray', 'amsr2PathMatrix');
end
