# 绘制AMSR2 LST反演模型的质量统计图.
import os
import matplotlib.pyplot as plt
import numpy as np
from os import path
from glob import glob

# 预设参数.
staNameList = ['PTS', 'R2', 'RMSE']
ylimList = [(-1, 36), (-0.05, 1.05), (-0.2, 7.2)]
yticksList = [np.sort(np.append(np.arange(5, 36, 5), 1)), np.arange(0, 1.1, 0.2),
              np.arange(0, 8, 1)]
dayColor, nightColor = {'color': 'r'}, {'color': 'b'}

# 路径.
rootDir = r'E:\AMSR2_MODIS_AW_LST\AMSR2_LST_Retrieval\Figures\ModelEstimation'

# 读取CSV文件中的统计数据.
yearDirList = glob(path.join(rootDir, '*'))
for yearDir in yearDirList:
    yearStr = path.basename(yearDir)

    # 创建年度文件夹.
    figYearDir = path.join(rootDir, yearStr)
    if not path.exists(figYearDir):
        os.mkdir(figYearDir)

    for i in range(len(staNameList)):
        staName = staNameList[i]

        figPath = path.join(figYearDir, 'Bplot_{0}_{1}.png'.format(staName, yearStr))
        if path.exists(figPath):
            continue

        staDayCsvPath = path.join(yearDir, 'Sta_{0}_{1} Day.csv'.format(staName, yearStr))
        staNightCsvPath = path.join(yearDir, 'Sta_{0}_{1} Night.csv'.format(staName, yearStr))
        staDayArray = np.loadtxt(staDayCsvPath, delimiter=',', skiprows=1)
        staNightArray = np.loadtxt(staNightCsvPath, delimiter=',', skiprows=1)

        # 绘图.
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        # fig.suptitle('The {0} in {1}'.format(staName, yearStr))

        staZip = zip([ax1, ax2], [staDayArray, staNightArray], [dayColor, nightColor])
        for ax, staArray, color in staZip:
            q = np.percentile(staArray, [25, 75], axis=0)
            iqr = q[1] - q[0]
            outlier1 = q[0] - 1.5 * iqr
            outlier2 = q[1] + 1.5 * iqr
            capList = np.zeros((2, len(iqr))) * np.nan
            for j in range(len(iqr)):
                staVector = staArray[:, j]
                capList[1, j] = min(staVector[np.where(staVector > outlier1[j])])
                capList[0, j] = max(staVector[np.where(staVector < outlier2[j])])
            capDiffVector = capList[0] - capList[1]

            ax.boxplot(staArray, boxprops=color, capprops=color, whiskerprops=color,
                       medianprops=color)
            ax.plot(range(1, 13), np.mean(staDayArray, 0), 'o', mec='r', mfc='lightcoral',
                    alpha=0.7, label='Daytime Mean')
            ax.plot(range(1, 13), np.mean(staNightArray, 0), 'o', mec='b', mfc='steelblue',
                    alpha=0.7, label='Nighttime Mean')
            ax.set(ylim=ylimList[i], yticks=yticksList[i], xlabel='Month', ylabel=staName)
            ax.yaxis.grid(True)
            ax.legend()

        plt.tight_layout()
        # plt.show()
        plt.savefig(figPath, dpi=200)

    continue

    for staName in staNameList:
        staDayCsvPath = path.join(yearDir, 'Hist_{0}_{1} Day.csv'.format(staName, yearStr))
        staNightCsvPath = path.join(yearDir, 'Hist_{0}_{1} Night.csv'.format(staName, yearStr))
        staDayArray = np.loadtxt(staDayCsvPath, delimiter=',', skiprows=1)
        staNightArray = np.loadtxt(staNightCsvPath, delimiter=',', skiprows=1)

        xDay = staDayArray[:, 0]
        yDay0 = staDayArray[:, 1]
        yDay1 = yDay0 + staDayArray[:, 2]
        yDay2 = yDay0 - staDayArray[:, 2]

        xNight = staNightArray[:, 0]
        yNight0 = staNightArray[:, 1]
        yNight1 = yNight0 + staDayArray[:, 2]
        yNight2 = yNight0 - staDayArray[:, 2]

        yMin = min([min(yDay2), min(yNight2)])
        yMax = max([max(yNight1), max(yDay1)])

        # 绘图.
        fig, ax = plt.subplots(figsize=(6.4, 3))
        ax.fill_between(xDay, yDay1, yDay2, alpha=.5, linewidth=0)
        ax.plot(xDay, yDay0, 'o-', linewidth=2)
        ax.fill_between(xNight, yNight1, yNight2, alpha=.5, linewidth=0)
        ax.plot(xNight, yNight0, 'o-', linewidth=2)
        ax.set(xlim=(xDay[0], xDay[-1]), xticks=xDay, ylim=(0, 6))

        plt.tight_layout()
        plt.show()
