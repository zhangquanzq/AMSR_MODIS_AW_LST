%% AMSR 地表温度的验证, 参考数据为MODIS LST.

%% 功能标记和预设参数.
% 指定研究区的标识. 1表示Antarctic, 2表示Greenland.
flg1 = 1;
% 指定微波数据类型的标识. 1表示AMSRE, 2表示AMSR2.
flg2 = 2;
% 指定白天和晚上的标记. 1表示白天(升轨), 2表示晚上(降轨).
flg3 = 2;

% 研究区, AMSRE/2类型, 空间参考变量字符串.
region = {'Antarctic', 'Greenland'};
region = region{flg1};

zoneN = [5, 6];
zoneN = zoneN(flg1);

amsrType = {'AMSRE', 'AMSR2'};
amsrType = amsrType{flg2};

thd = {[50, 50], [23, 50]};
thd = thd{flg1}; thd = thd(flg2);

% 数据年份列表(时间区间2012/07/02-2019/12/31).
yearList = {2003: 2011, 2012: 2020}; 
yearList = yearList{flg2};

% 昼夜标记.
daynight = {'Day', 'Night'};
daynight = daynight{flg3};

%% 路径.
% 根目录.
rootDir = 'K:\AMSR_LST_IceSheet\';
dataDir = fullfile(rootDir, 'Data');

% 输入数据路径.
modisLstMaskMatDir = fullfile(dataDir, sprintf('MYD11A1_3_Mask%s_Matlab', region));
amsrLstMatDir = fullfile(dataDir, sprintf('%s_4_LST_%s_Matlab_%dzones', amsrType, region, zoneN));
regressMatDir = fullfile(dataDir, sprintf('Regression_%s_Matlab', region));

% 输出的统计指标数据路径.
lstScatterDir = fullfile(rootDir, 'Figures', sprintf('%s_MODIS_%s_LST_Scatter', amsrType, region));
if ~exist(lstScatterDir, 'dir')
    mkdir(lstScatterDir)
end

%% 统计作图.
for i = 1: length(yearList)
    yearStr = num2str(yearList(i));
%     yearStr = '2020';

    lstScatterYearDir = fullfile(lstScatterDir, yearStr);
    if ~exist(lstScatterYearDir, 'dir')
        mkdir(lstScatterYearDir)
    end

    % 从Mat文件中读取Mask后的MODIS LST数据.
    modisLstMaskMatName = sprintf('MYD11A1_Mask%s_%s_%s.mat', region, yearStr, daynight);
    modisLstMaskMatPath = fullfile(modisLstMaskMatDir, modisLstMaskMatName);
    load(modisLstMaskMatPath, 'modisLstMaskYearArray', 'lstDateFilterList');  % single, 无效值是Nan.

    % 从Mat文件中读取反演的AMSR2 LST数据.
    amsrLstYearMatName = sprintf('%s_LST_%s_%s_CN.mat', amsrType, daynight, yearStr);
    amsrLstYearMatPath = fullfile(amsrLstMatDir, amsrLstYearMatName);
    load(amsrLstYearMatPath, 'amsrLstYearArray', 'dateYearList');  % single, 无效值是0.

    [dateList, modisDateIndex, amsrDateIndex] = intersect(lstDateFilterList, dateYearList);

    modisLstMaskYearArray = modisLstMaskYearArray(:, :, modisDateIndex);
    amsrLstYearArray = amsrLstYearArray(:, :, amsrDateIndex);

    % !!! 逐步回归得到的AMSR2 LST值与MODIS LST的散点图 !!!
    if 0
        regressPureMatName = sprintf('Regression_Pure_%s_%s.mat', yearStr, daynight);
        regressPureMatPath = fullfile(regressMatDir, regressPureMatName);
        load(regressPureMatPath, 'amsr2LstCnYearArray3', 'amsr2LstMaskYearArray3');
        amsr2LstCnYearArray3 = amsr2LstCnYearArray3(:,:,modisDateIndex);
        amsr2LstMaskYearArray3 = amsr2LstMaskYearArray3(:,:,modisDateIndex);
        amsrLstYearArray = amsr2LstMaskYearArray3;
%         modisLstMaskYearArray = amsr2LstMaskYearArray3;
    end

    fprintf('输出%s年AMSR2和MODIS地表温度的散点图.\n', yearStr);
    % 输出AMSR2和MODIS LST之间的年度散点图.
    timestamp = sprintf('%s %s', yearStr, daynight);
    lstScatterName = sprintf('%s_MODIS_LST_Scatter_%s.tif', amsrType, timestamp);
    lstScatterPath = fullfile(lstScatterDir, lstScatterName);
    if ~exist(lstScatterPath, 'file')
        validIndexVector = find((amsrLstYearArray ~= 0) & ~isnan(modisLstMaskYearArray) & ...
            abs(amsrLstYearArray - modisLstMaskYearArray) < thd);
%         validIndexVector = find((amsr2LstCnYearArray ~= 0) & (modisLstMaskYearArray ~= 0));
        amsrLstVector = amsrLstYearArray(validIndexVector);
        modisLstMaskVector = modisLstMaskYearArray(validIndexVector);

        f = lstScatter(amsrLstVector, modisLstMaskVector, timestamp);
        exportgraphics(f, lstScatterPath);
        close all
    end

    continue

    % 输出AMSR2和MODIS LST之间的每日散点图.
    for j = 1: length(dateList)
        timestamp = sprintf('%s %s', dateList{j}, daynight);
        lstScatterName = sprintf('%s_MODIS_LST_Scatter_%s.tif', amsrType, timestamp);
        lstScatterPath = fullfile(lstScatterYearDir, lstScatterName);
        if ~exist(lstScatterPath, 'file')
            amsrLstDailyLayer = amsrLstYearArray(:, :, j);
            modisLstMaskDailyLayer = modisLstMaskYearArray(:, :, j);

            validIndexVector = find((amsrLstDailyLayer ~= 0) & ~isnan(modisLstMaskDailyLayer) & ...
                 abs(amsrLstYearArray - modisLstMaskYearArray) < thd);
%             validIndexVector = find((amsr2LstCnDailyLayer ~= 0) & (modisLstMaskDailyLayer ~= 0));
            amsrLstVector = amsrLstDailyLayer(validIndexVector);
            modisLstMaskVector = modisLstMaskDailyLayer(validIndexVector);

            f = lstScatter(amsrLstVector, modisLstMaskVector, timestamp);
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
