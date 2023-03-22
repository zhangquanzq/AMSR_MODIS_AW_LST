%% AMSR2地表温度反演模型评价.

%% 功能标记与预设参数.
% 指定白天和晚上的标记. 1表示白天(升轨), 2表示晚上(降轨).
flg1 = 1;

% 昼夜标记.
daynight = {'Day', 'Night'};
daynight = daynight{flg1};

% 数据年份列表(时间区间2012/07/02-2019/12/31).
yearList = 2012: 2020;

% 各月份的名称.
monthList = {'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'};
monthN = length(monthList);

% 各统计变量名称.
staVarNameList = {'monthSampleRatioArray', 'fixedR2MonthArray', 'fixedRmseMonthArray'};

% 各统计图名称.
staNameList = {'PTS', 'R2', 'RMSE'};

% 直方图各统计变量的分级.
edgesList = {[0, 1, 3:3:15], 0:0.2:1, 0:6};

% 直方图各统计变量图表属性.
yLimList = {[0 40], [0 60], [0 60]};  % Y轴范围.
staTextY = [36, 56, 56];  % 精度评价指标Y轴位置.
monthTextY = [-3, -6, -5];  % 月份标注Y轴位置.
staNameX = [96, 84, 96];  % 统计指标名称X轴位置.

% 空间分布图各变量图表属性.
meanValueRange = {[0 15], [0 1], [0 5]};  % 均值取值范围.
stdValueRange = {[0 15], [0 0.2], [0 1]};  % 标准差取值范围.
meanValueTicks = {0:3:15, 0:0.2:1, 0:5};
stdValueTicks = {0:3:15, 0:0.05:0.2, 0:0.2:1};

%% 路径.
% 根目录.
rootDir = 'F:\AMSR2_MODIS_AW_LST\AMSR2_LST_Retrieval';
dataDir = fullfile(rootDir, 'Data');
figDir = fullfile(rootDir, 'Figures');

% 输入数据路径.
regressMatDir = fullfile(dataDir, 'Regression_Matlab');
ccsevMatDir = fullfile(dataDir, 'CCSEV_Matlab');

% 输出图表路径.
modelEstDir = fullfile(figDir, 'ModelEstimation');
if ~exist(modelEstDir, 'dir')
    mkdir(modelEstDir)
end

%% 模型评估.
for i = 1: length(yearList)
    yearStr = num2str(yearList(i));
%     yearStr = '2017';
    timestamp = sprintf('%s %s', yearStr, daynight);

    % 模型Mat数据路径.
    regressMatName = sprintf('Regression_Pure_%s_%s.mat', yearStr, daynight);
    regressMatPath = fullfile(regressMatDir, regressMatName);
    if ~exist(regressMatPath, 'file')
        error('%s年%s的模型文件不存在, 请检查!', yearStr, daynight);
    end

    % 输出图表的年度文件夹.
    modelEstYearDir = fullfile(modelEstDir, yearStr);
    if ~exist(modelEstYearDir, 'dir')
        mkdir(modelEstYearDir)
    end

    % 分精度评价指标制图.
    for j = 1: length(staVarNameList)
        staVarName = staVarNameList{j};
        staName = staNameList{j};
        load(regressMatPath, 'pureLcCodeList', staVarName);

        % 不统计积雪覆盖区(601, 602, ..., 698).
        pureLcCodeIndex = pureLcCodeList < 600;
        staVarArray = eval(staVarName);
        staVarArray = staVarArray(pureLcCodeIndex, :);
        pureLcCodeList = pureLcCodeList(pureLcCodeIndex);

        % 模型分昼夜和月份的指标均值, 方差统计.
        [staMean, staStd] = deal(zeros(monthN, 1));
        for k = 1: monthN
            staMonth = staVarArray(:, k);
            staMean(k) = mean(staMonth);
            staStd(k) = std(staMonth);
        end

        % 导出原始数据为CSV.
        modelStaCsvPath = fullfile(modelEstYearDir, sprintf('Sta_%s_%s.csv', staName, timestamp));
        if ~exist(modelStaCsvPath, 'file')
            writetable(array2table(staVarArray), modelStaCsvPath)
        end

        % 导出统计值为CSV.
        modelHistCsvPath = fullfile(modelEstYearDir, sprintf('Hist_%s_%s.csv', staName, timestamp));
        if ~exist(modelHistCsvPath, 'file')
            staArray = [(1: monthN)', staMean, staStd];
            staTable = array2table(staArray, VariableNames={'Month', 'Mean', 'Std'});
            writetable(staTable, modelHistCsvPath)
        end
        
        % 直方图制图.
        modelHistFigPath = fullfile(modelEstYearDir, sprintf('Hist_%s_%s.png', staName, timestamp));
        if j == 1, staName = replace(staName, ' ', '\n'); end
        if ~exist(modelHistFigPath, 'file')
            f1 = figure; f1.Position = [10 100 2000 500];
            edges = edgesList{j}; edgesN = length(edges);
            xtickNum = 1: edgesN-1; staTextX = 1; monthTextX = 3;
            for k = 1: monthN
                b1 = bar(xtickNum, histcounts(staVarArray(:, k), edges)); b1.BarWidth = 1;

                meanStr = sprintf('Mean: %.2f', staMean(k));
                stdStr = sprintf('STD: %.2f', staStd(k));
                text(staTextX, staTextY(j), meanStr, FontSize=11, FontWeight='bold');
                text(staTextX, staTextY(j)-2, stdStr, FontSize=11, FontWeight='bold');
                text(monthTextX, monthTextY(j), monthList{k}, FontSize=11, FontWeight='bold');

                xtickNum = xtickNum + (edgesN + 1);
                staTextX = staTextX + (edgesN + 1);
                monthTextX = monthTextX + (edgesN + 1);

                hold on
            end
            xlabelStr = repmat([string(edges), ''], 1, monthN);
            ax = gca; ax.FontWeight = 'bold'; grid on;
            ax.YLim = yLimList{j}; ax.YLabel.String = sprintf('Zones Number（%s）', daynight);
            ax.XTick = 0.5:length(xlabelStr); ax.XTickLabel = xlabelStr; ax.XTickLabelRotation = 90;
            if j == 1, staName = replace(staName, ' ', '\n'); end
            text(staNameX(j), 0, sprintf(staName), FontWeight='bold')

            exportgraphics(f1, modelHistFigPath)
            close all
        end

        % 空间分布图.
        if j == 1, staName = replace(staName, '\n', ' '); end
        modelZonesMeanFigName = sprintf('Zones_%s_Mean_%s.png', staName, timestamp);
        modelZonesMeanFigPath = fullfile(modelEstYearDir, modelZonesMeanFigName);
        modelZonesStdFigName = sprintf('Zones_%s_Std_%s.png', staName, timestamp);
        modelZonesStdFigPath = fullfile(modelEstYearDir, modelZonesStdFigName);
        if ~exist(modelZonesMeanFigPath, 'file') || ~exist(modelZonesStdFigPath, 'file')
            load(fullfile(ccsevMatDir, sprintf('CCSEV_%s', yearStr)), 'zonesLcLayer');
            zonesLcLayer = single(zonesLcLayer);
            zonesLcLayer(zonesLcLayer == 128) = nan;

            [staMeanZonesLayer, staStdZonesLayer] = deal(zeros(size(zonesLcLayer))*nan);

            staVarMeanVector = mean(staVarArray, 2);
            staVarStdVector = std(staVarArray, 0, 2);
            for k = 1: length(pureLcCodeList)
                staMeanZonesLayer(zonesLcLayer == pureLcCodeList(k)) = staVarMeanVector(k);
                staStdZonesLayer(zonesLcLayer == pureLcCodeList(k)) = staVarStdVector(k);
            end

            f2 = figure;
            imshow(staMeanZonesLayer, meanValueRange{j}, Colormap=turbo);
            colorbar(FontSize=12, FontWeight='bold', Ticks=meanValueTicks{j});
            titleStr = sprintf('Annual Mean of %s in %s', staNameList{j}, daynight);
            title(titleStr, FontSize=12, FontWeight='bold');
            exportgraphics(f2, modelZonesMeanFigPath);

            f3 = figure;
            imshow(staStdZonesLayer, stdValueRange{j}, Colormap=turbo);
            colorbar(FontSize=12, FontWeight='bold', Ticks=stdValueTicks{j});
            titleStr = sprintf('Annual STD of %s in %s', staNameList{j}, daynight);
            title(titleStr, FontSize=12, FontWeight='bold');
            exportgraphics(f3, modelZonesStdFigPath);

            close all
        end
    end
end

