%% AMSR2地表温度的验证, 参考数据为MODIS LST.

%% 功能标记和预设参数.
% 指定白天和晚上的标记. 1表示白天(升轨), 2表示晚上(降轨).
flg1 = 2;

% 数据年份列表(时间区间2012/07/02-2019/12/31).
yearList = 2012: 2019;
yearN = length(yearList);

% 昼夜标记.
daynight = {'Day', 'Night'};
daynight = daynight{flg1};

%% 路径.
% 根目录.
rootDir = 'E:\AMSR2_MODIS_AW_LST';
retrievalDir = fullfile(rootDir, 'AMSR2_LST_Retrieval');
dataPath = fullfile(retrievalDir, 'Data');

% 输入数据路径.
modisLstMaskMatDir = fullfile(dataPath, 'MYD11A1_3_MaskCn_Matlab');
amsr2LstMatDir = fullfile(dataPath, 'AMSR2_4_LSTCN_Matlab');
regressMatDir = fullfile(dataPath, 'Regression_Matlab'); 

% 输出的统计指标数据路径.
lstScatterDir = fullfile(retrievalDir, 'Figures\AMSR2_MODIS_LST_Scatter');
if ~exist(lstScatterDir, 'dir')
    mkdir(lstScatterDir)
end

%% 统计作图.
for i = 1: yearN
%     yearStr = num2str(yearList(i));
    yearStr = '2020';

    lstScatterYearDir = fullfile(lstScatterDir, yearStr);
    if ~exist(lstScatterYearDir, 'dir')
        mkdir(lstScatterYearDir)
    end

    % 从Mat文件中读取Mask后的MODIS LST数据.
    modisLstMaskMatName = sprintf('MYD11A1_MaskCn_%s_%s.mat', yearStr, daynight);
    modisLstMaskMatPath = fullfile(modisLstMaskMatDir, modisLstMaskMatName);
    load(modisLstMaskMatPath, 'modisLstMaskYearArray', 'lstDateFilterList');  % single, 无效值是Nan.

    % 从Mat文件中读取反演的AMSR2 LST数据.
    amsr2LstYearMatName = sprintf('AMSR2_Lst_%s_%s_CN.mat', daynight, yearStr);
    amsr2LstYearMatPath = fullfile(amsr2LstMatDir, amsr2LstYearMatName);
    load(amsr2LstYearMatPath, 'amsr2LstCnYearArray', 'validDateList');  % single, 无效值是0.

    [dateList, modisDateIndex, amsr2DateIndex] = intersect(lstDateFilterList, validDateList);

    modisLstMaskYearArray = modisLstMaskYearArray(:, :, modisDateIndex);
    amsr2LstCnYearArray = amsr2LstCnYearArray(:, :, amsr2DateIndex);

    % !!! 逐步回归得到的AMSR2 LST值与MODIS LST的散点图 !!!
    if 0
        regressPureMatName = sprintf('Regression_Pure_%s_%s.mat', yearStr, daynight);
        regressPureMatPath = fullfile(regressMatDir, regressPureMatName);
        load(regressPureMatPath, 'amsr2LstCnYearArray3', 'amsr2LstMaskYearArray3');
        amsr2LstCnYearArray3 = amsr2LstCnYearArray3(:,:,modisDateIndex);
        amsr2LstMaskYearArray3 = amsr2LstMaskYearArray3(:,:,modisDateIndex);
        amsr2LstCnYearArray = amsr2LstMaskYearArray3;
%         modisLstMaskYearArray = amsr2LstMaskYearArray3;
    end

    fprintf('输出%s年AMSR2和MODIS地表温度的散点图.\n', yearStr);
    % 输出AMSR2和MODIS LST之间的年度散点图.
    timestamp = sprintf('%s %s', yearStr, daynight);
    lstScatterName = sprintf('AMSR2_MODIS_LST_Scatter_%s.tif', timestamp);
    lstScatterPath = fullfile(lstScatterDir, lstScatterName);
    if ~exist(lstScatterPath, 'file')
        validIndexVector = find((amsr2LstCnYearArray ~= 0) & ~isnan(modisLstMaskYearArray));
%         validIndexVector = find((amsr2LstCnYearArray ~= 0) & (modisLstMaskYearArray ~= 0));
        amsr2LstCnVector = amsr2LstCnYearArray(validIndexVector);
        modisLstMaskVector = modisLstMaskYearArray(validIndexVector);

        f = lstScatter(amsr2LstCnVector, modisLstMaskVector, timestamp);
        exportgraphics(f, lstScatterPath);
        close all
    end

%     continue

    % 输出AMSR2和MODIS LST之间的每日散点图.
    for j = 1 : length(dateList)
        timestamp = sprintf('%s %s', dateList{j}, daynight);
        lstScatterName = sprintf('AMSR2_MODIS_LST_Scatter_%s.tif', timestamp);
        lstScatterPath = fullfile(lstScatterYearDir, lstScatterName);
        if ~exist(lstScatterPath, 'file')
            amsr2LstCnDailyLayer = amsr2LstCnYearArray(:, :, j);
            modisLstMaskDailyLayer = modisLstMaskYearArray(:, :, j);

            validIndexVector = find((amsr2LstCnDailyLayer ~= 0) & ~isnan(modisLstMaskDailyLayer));
%             validIndexVector = find((amsr2LstCnDailyLayer ~= 0) & (modisLstMaskDailyLayer ~= 0));
            amsr2LstCnVector = amsr2LstCnDailyLayer(validIndexVector);
            modisLstMaskVector = modisLstMaskDailyLayer(validIndexVector);

            f = lstScatter(amsr2LstCnVector, modisLstMaskVector, timestamp);
            exportgraphics(f, lstScatterPath);
            close all
        end
    end
end

%% 自定义函数.
% AMSR2和MODIS地表温度的散点图.
function f = lstScatter(amsr2LstVector, modisLstVector, timestamp)
% index1 = (220 < amsr2LstVector) & (amsr2LstVector < 360);
% index2 = (220 < modisLstVector) & (modisLstVector < 360);
% 
% validIndex = index1 & index2;
% amsr2LstVector = amsr2LstVector(validIndex);
% modisLstVector = modisLstVector(validIndex);

lstBias = mean(amsr2LstVector - modisLstVector);
lstMAE = mean(abs(amsr2LstVector - modisLstVector));
lstR = corrcoef(amsr2LstVector, modisLstVector);
lstR2 = lstR(1, 2) .^ 2;
lstRMSE = sqrt(sum((amsr2LstVector - modisLstVector).^2) / length(amsr2LstVector));

f = figure; f.Visible = false;
plot(amsr2LstVector, modisLstVector, '.k', [220, 360], [220, 360], 'r');
xlabel('AMSR2 LST (K)'); ylabel('MODIS LST (K)');

txt0 = timestamp;
txt1 = ['N: ', num2str(sum(~isnan(modisLstVector)))];
txt2 = ['R^2: ', num2str(lstR2, '%.3f')];
txt3 = ['Bias: ', num2str(lstBias, '%.3f')];
txt4 = ['MAE: ', num2str(lstMAE, '%.3f')];
txt5 = ['RMSE: ', num2str(lstRMSE, '%.3f')];
text(0.7, 0.35, txt0, Units='normalized', FontSize=12, Color='r');
text(0.7, 0.29, txt1, Units='normalized', FontSize=12);
text(0.7, 0.23, txt2, Units='normalized', FontSize=12);
text(0.7, 0.17, txt3, Units='normalized', FontSize=12);
text(0.7, 0.11, txt4, Units='normalized', FontSize=12);
text(0.7, 0.05, txt5, Units='normalized', FontSize=12);
end
