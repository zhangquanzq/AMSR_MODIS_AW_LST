%% 将AMSR2 H5格式的各通道亮温数据根据指定空间范围输出为TIF格式.
% 虽然此脚本输出的影像范围包含了全球, 但实际并不使用.

%% 标记和预设参数.
% 指定输出范围的标记. 1表示全球, 2表示南极, 3表示格陵兰.
flg1 = 3;

% AMSR2数据范围.
region = {'World', 'Antarctic', 'Greenland'};
region = region{flg1};

% 输出AMSR2影像的经纬度范围和空间参考.
latLim = {[-90, 90], [-90, -50], [58, 84]};  % {[World], [Antarctic], [Greenland]}
lonLim = {[0, 360], [0, 360], [285, 352]};  % {[World], [Antarctic], [Greenland]}
latLim = latLim{flg1}; lonLim = lonLim{flg1}; cellsize = 0.1;

startLon = (lonLim(1) - 0) / cellsize + 1; endLon = (lonLim(2) - 0) / cellsize;
startLat = (90 - latLim(2)) / cellsize + 1; endLat = (90 - latLim(1)) / cellsize;

amsr2Ref = georefcells(latLim, lonLim, cellsize, cellsize, 'ColumnsStartFrom', 'north');

% AMSR2数据的通道与极化.
% channelList = ["10", "18", "23", "36", "6", "7", "89"];
polarize = ["H", "V"];
% [cMatrix, pMatrix] = meshgrid(channelList, polarize);
% cpList = cellstr(reshape(cMatrix + pMatrix, [], 1));
% cpN = length(cpList);

% 有数据的时间区间：2012/07/02-2020/12/31.
% startDate = [2012, 07, 02]; endDate = [2020, 12, 31];
% dateAllList = cellstr(datetime(startDate) : datetime(endDate), 'yyyyMMdd')';
% yearNumList = startDate(1) : endDate(1);

%% 路径.
% 根目录.
dataDir = 'F:\AMSR_LST_IceSheet\Data';

% 输入数据路径.
amsr2H5Dir = fullfile(dataDir, 'AMSR2_1_BT_H5');

% 输出数据路径.
amsr2RegionDir = fullfile(dataDir, sprintf('AMSR2_2_BT_%s_TIF', region));
if ~exist(amsr2RegionDir, 'dir')
    mkdir(amsr2RegionDir)
end

% amsr2MatDir = fullfile(dataDir, sprintf('AMSR2_2_BT_%s_Matlab', region));
% if ~exist(amsr2MatDir, 'dir')
%     mkdir(amsr2MatDir);
% end

%% 将AMSR2 H5格式转为TIF格式.
amsr2BandFolderList = dir(fullfile(amsr2H5Dir, 'L3*'));
amsr2BandFolderList = {amsr2BandFolderList.name}';
for i = 1: length(amsr2BandFolderList)
    amsr2BandFolder = amsr2BandFolderList{i};

    % 创建AMSR2 TIF波段文件夹.
    amsr2TifBandDir = fullfile(amsr2RegionDir, amsr2BandFolder);
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
            fprintf('转换%s年%s月AMSR2 BT %s数据的格式.\n', amsr2YearFolder, amsr2MonthFolder, ...
                amsr2BandFolder(6:10))

            % 读取AMSR2 H5每日文件列表.
            amsr2H5MonthDir = fullfile(amsr2H5YearDir, amsr2MonthFolder);
            amsr2H5List = dir(fullfile(amsr2H5MonthDir, '*.h5'));
            amsr2H5List = {amsr2H5List.name}';
            for m = 1: length(amsr2H5List)
                amsr2H5Name = amsr2H5List{m};
                amsr2H5Path = fullfile(amsr2H5MonthDir, amsr2H5Name);

                % 保存H5格式中的极化亮温数据为TIF格式.
                for n = 1: length(polarize)
                    replacedStr = sprintf('_Bt%s.tif', polarize(n));
                    amsr2TifName = replace(amsr2H5Name, '.h5', replacedStr);
                    amsr2TifPath = fullfile(amsr2TifMonthDir, amsr2TifName);
                    if ~exist(amsr2TifPath, 'file')
                        layerName = sprintf('/Brightness Temperature (%s)', polarize(n));
                        try
                            btArray = h5read(amsr2H5Path, layerName)';
                            btArray = btArray(startLat : endLat, startLon : endLon);
                            geotiffwrite(amsr2TifPath, btArray, amsr2Ref, CoordRefSysCode=4326);
                        catch
                            disp(['有问题数据：', amsr2TifName]);
                        end
                    end
                end
            end
        end
    end
end

%% 将AMSR2 BT数据从TIF格式转为Mat格式.
% disp('获取所有年份AMSR2 TIF文件的路径列表.');
% amsr2DailyPathAllList = cell(length(dateAllList) * cpN, 1); % 每日7个波段, 2各极化.
% amsr2TifDir = fullfile(dataDir, sprintf('AMSR2_2_BT_%s_TIF', region));
% amsr2ChannelList = dir(fullfile(amsr2TifDir, 'L3*'));
% amsr2ChannelList = {amsr2ChannelList.name}';
% startIndex = 1;
% for i = 1: length(amsr2ChannelList)
%     amsr2ChannelDir = fullfile(amsr2TifDir, amsr2ChannelList{i});
%     amsr2YearList = dir(fullfile(amsr2ChannelDir));
%     amsr2YearList = {amsr2YearList(3:end).name}'; % 排除 '.', '..'两个文件夹.
%     for j = 1: length(amsr2YearList)
%         amsr2YearDir = fullfile(amsr2ChannelDir, amsr2YearList{j});
%         amsr2MonthList = dir(amsr2YearDir);
%         amsr2MonthList = {amsr2MonthList(3:end).name}'; % 排除 '.', '..'两个文件夹.
%         for k = 1: length(amsr2MonthList)
%             amsr2MonthDir = fullfile(amsr2YearDir, amsr2MonthList{k});
%             amsr2DailyList = dir(fullfile(amsr2MonthDir, sprintf('*01D_EQM%s*.tif', orbit)));
%             amsr2DailyList = {amsr2DailyList.name}';
%             amsr2DailyPath = fullfile(amsr2MonthDir, amsr2DailyList);
%             endIndex = startIndex + length(amsr2DailyList) - 1;
%             amsr2DailyPathAllList(startIndex: endIndex) = amsr2DailyPath;
%             startIndex = endIndex + 1;
%         end
%     end
% end
% amsr2DailyPathAllList(cellfun(@isempty, amsr2DailyPathAllList)) = [];

% % 读取AMSR2 BT数据, 并存储为Mat格式.
% amsr2RowN = amsr2Ref.RasterSize(1); amsr2ColN = amsr2Ref.RasterSize(2);
% for i = 1: length(yearNumList)
%     yearStr = num2str(yearNumList(i));
% 
%     % 检查是否已存在当年的AMSR2 BT mat格式文件。
%     amsr2YearMatName = sprintf('AMSR2_BT_%s_%s_%s.mat', region, yearStr, daynight);
%     amsr2YearMatPath = fullfile(amsr2MatDir, amsr2YearMatName);
%     if exist(amsr2YearMatPath, 'file')
%         continue
%     end
% 
%     % 索引当年的AMSR2亮温数据存储路径, 并分极化方式与波段存储为mat格式文件.
%     yearIndex1 = contains(amsr2DailyPathAllList, [yearStr, '\']);
%     yearIndex2 = strcmp(extractBefore(dateAllList, 5), yearStr);
%     % 极化排序：[H, V]，通道排序：[10, 18, 23, 36, 6, 7, 89]
%     amsr2DailyPathYearList = amsr2DailyPathAllList(yearIndex1);
%     dateYearList = dateAllList(yearIndex2);
%     dateYearN = length(dateYearList);
% 
%     for j = 1: length(channelList)
%         for k = 1: length(polarize)
% %             eval(sprintf('amsr2%s%sYearArray', polarize{k}, channelList{j}))
%             assignin('base',sprintf('amsr2%s%sYearArray', polarize{k}, channelList{j}), zeros(amsr2RowN, amsr2ColN, dateYearN, 'uint16'))
%         end
% 
%     end
% 
% 
% end














