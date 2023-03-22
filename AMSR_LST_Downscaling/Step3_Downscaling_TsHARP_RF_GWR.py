# coding=utf-8
import arcpy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from functions import *
from glob import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.polynomial import polynomial as poly
from os import path

# 标记与预设参数.
# 指定研究区的标记. 1表示内蒙古, 2表示晋南豫西, 3表示云贵高原, 4表示北大河, 5表示那曲.
flg1 = 2
# 指定昼夜的标记. 1表示白天, 2表示夜晚.
flg2 = 1
# 指定微波传感器的标记. 1表示AMSRE, 2表示AMSR2.
flg3 = 1
# 指定降尺度方法的标记. 1表示TsHARP, 2表示Random Forest, 3表示GWR.
flg4 = 3
# 指定降尺度策略的标记. 1表示逐级降尺度, 2表示直接降尺度.
flg5 = 2
# 指定降尺度数据源的标记. 1表示重采样的MODIS LST, 2表示真实的AMSRE/2 LST.
flg6 = 1
# 指定降尺度模型是否包含残差项的标记. 1表示包含, 0表示不包含.
flg7 = 1

gsSigma = 0.7

region = ['NMG', 'JNYX', 'YGGY', 'BRB', 'Naqu'][flg1 - 1]
daynight = ['Day', 'Night'][flg2 - 1]
sensor = ['AMSRE', 'AMSR2'][flg3 - 1]
yearList = [[2010], range(2013, 2021)][flg3 - 1]
cellsizeList = [['0.25', '0.08', '0.03', '0.01'], ['0.1', '0.03', '0.01']][flg3 - 1]
cellsizeList = [cellsizeList, [cellsizeList[0], cellsizeList[-1]]][flg5 - 1]

method = ['TsHARP', 'RF', 'GWR'][flg4 - 1]
strategy = ['Stepwise', 'Straight'][flg5 - 1]
strategy = strategy if flg7 == 1 else strategy + '_noResidual'
sourceLst = ['MYD11A1', 'AMSR'][flg6 - 1]
arcpy.env.overwriteOutput = True

#%% 路径.
# 根目录.
rootDir = r'E:\AMSR_MODIS_AW_LST'
downscalingDir = path.join(rootDir, 'AMSR_LST_Downscaling')
dataDir = path.join(downscalingDir, 'Data')

# 筛选数据和输出图表的路径.
regionDir = path.join(dataDir, 'Region_{0}_{1}'.format(region, cellsizeList[0]))
srtmDir = path.join(regionDir, 'SRTM_{0}'.format(region))

figRegionDir = path.join(downscalingDir, r'Figures\Region_{0}_{1}'.format(region, cellsizeList[0]))
if not path.exists(figRegionDir):
    os.mkdir(figRegionDir)

#%% 分年份执行降尺度实验.
for yearNum in yearList:
    # 研究区年份昼夜字符串.
    rydStr = '{0}_{1}_{2}'.format(region, yearNum, daynight)

    # 从统计文件中读取符合条件的AMSR2和MODIS地表温度配对指标.
    dateSelectCsvPath = path.join(regionDir, 'DateSelect_{0}.csv'.format(rydStr))
    dateSelectFrame = pd.read_csv(dateSelectCsvPath)  # 字段名: YearDate, PixelPercent, R

    # 遍历当年有效数据日期, 对LST进行降尺度, 保存精度评价指标.
    downscalingStaList = []
    for dateNum in dateSelectFrame['YearDate']:
        # dateNum = 20130524
        dateStr = ymd2yday(str(dateNum))

        # 判断当前日期数据的可用性.
        regionDateStr = '{0}_{1}_{2}'.format(region, dateNum, daynight)
        regionDateDir = path.join(regionDir, rydStr, regionDateStr)
        if not path.exists(regionDateDir):
            print(u'{0} 不存在', regionDateStr)
            continue

        # 创建存放指定降尺度方法数据的文件夹.
        methodDir = path.join(regionDateDir, '{0}_{1}'.format(method, strategy))
        if not path.exists(methodDir):
            os.mkdir(methodDir)

        # 初始分辨率LST数据的路径.
        amsrLstInitialName = '{0}_LST_{1}_{2}_{3}.tif'.format(sensor, daynight, region, dateNum)
        modisLstInitialName = 'MYD11A1.A{0}.LST_{1}_{2}_{3}.tif'.format(dateStr, daynight,
                                                                        region, cellsizeList[0])
        initialLstName = [modisLstInitialName, amsrLstInitialName][flg6 - 1]
        initialLstPath = path.join(regionDateDir, initialLstName)

        print(u'使用{0}方法将{1} {2} LST的分辨率从{3} {4}降尺度到{5}.'.
              format(method, regionDateStr, sourceLst, cellsizeList[0], strategy, cellsizeList[-1]))
        # 按分辨率级别进行逐级降尺度.
        predictLstStr = '_'.join([sourceLst, 'LST', daynight, region, str(dateNum)])
        if method == 'TsHARP':
            tsharpStepwise(initialLstPath, regionDateDir, methodDir, cellsizeList)
        elif method == 'RF':
            rfStepwise(initialLstPath, regionDateDir, srtmDir, methodDir, cellsizeList, gsSigma)
        elif method == 'GWR':
            gwrStepwise(initialLstPath, regionDateDir, srtmDir, methodDir, cellsizeList)

        # %% 绘制LST, NDVI的栅格图和散点图, 计算精度评价指标.
        # 读取各数据矩阵.
        initialLstLayer = arcpy.RasterToNumPyArray(initialLstPath, nodata_to_value=np.nan)

        # !!! AMSR2降尺度时, 没有此对应数据, 后面绘制图表时, 需要排除 !!!
        originalLstName = 'MYD11A1.A{0}.LST_{1}_{2}_0.01.tif'.format(dateStr, daynight, region)
        originalLstPath = path.join(regionDateDir, originalLstName)
        originalLstLayer = arcpy.RasterToNumPyArray(originalLstPath, nodata_to_value=np.nan)

        targetLstName = '{0}_{1}.tif'.format(predictLstStr, cellsizeList[-1])
        targetLstPath = path.join(methodDir, targetLstName)
        targetLstLayer = arcpy.RasterToNumPyArray(targetLstPath, nodata_to_value=np.nan)

        originalNdviName = '*NDVI*{0}.tif'.format(cellsizeList[-1])
        originalNdviPath = glob(path.join(regionDateDir, originalNdviName))[0]
        originalNdviLayer = arcpy.RasterToNumPyArray(originalNdviPath) / 10000.0

        # 计算降尺度LST的精度指标.
        validIndexLayer = np.logical_not(np.isnan(originalLstLayer) | np.isnan(targetLstLayer))
        lstXVector, lstYVector = originalLstLayer[validIndexLayer], targetLstLayer[validIndexLayer]
        bias = np.mean(lstYVector - lstXVector)
        rmse = np.sqrt(np.sum((np.power(lstXVector - lstYVector, 2))) / len(lstXVector))
        r2 = np.power(np.corrcoef(lstXVector, lstYVector)[0, 1], 2)

        # 创建保存图表的文件夹.
        figRegionDateDir = path.join(figRegionDir, rydStr, '_'.join([method, regionDateStr]))
        if not path.exists(figRegionDateDir):
            os.makedirs(figRegionDateDir)

        # 绘制栅格图.
        # 获取温度显示范围值.
        pct1 = np.percentile(initialLstLayer[np.logical_not(np.isnan(initialLstLayer))], 0.5)
        pct2 = np.percentile(originalLstLayer[np.logical_not(np.isnan(originalLstLayer))], 0.5)
        pct3 = np.percentile(targetLstLayer[np.logical_not(np.isnan(targetLstLayer))], 0.5)
        pct4 = np.percentile(initialLstLayer[np.logical_not(np.isnan(initialLstLayer))], 99.5)
        pct5 = np.percentile(originalLstLayer[np.logical_not(np.isnan(originalLstLayer))], 99.5)
        pct6 = np.percentile(targetLstLayer[np.logical_not(np.isnan(targetLstLayer))], 99.5)
        lstMin, lstMax = np.round(np.min([pct1, pct2, pct3])), np.round(np.max([pct4, pct5, pct6]))

        layerList = [initialLstLayer, originalLstLayer, targetLstLayer, originalNdviLayer]
        layerStrList = ['Initial LST ({0})'.format(sourceLst), 'Original LST (MODIS)',
                        'Target LST ({0})'.format(sourceLst), 'Original NDVI']
        for i in range(len(layerList)):
            # 判断栅格图是否存在.
            figName = '_'.join([regionDateStr, sourceLst, strategy, 'Image', layerStrList[i]])
            figPath = path.join(figRegionDateDir, figName + '.png')
            if path.exists(figPath):
                continue

            # 绘制栅格数据.
            fig, ax = plt.subplots()
            vMin = lstMin - np.mod(lstMin, 5) if 'LST' in layerStrList[i] else 0
            vMax = lstMax - np.mod(lstMax, 5) + 5 if 'LST' in layerStrList[i] else 1
            im = ax.imshow(layerList[i], cmap='turbo', interpolation='nearest', vmin=vMin,
                           vmax=vMax)
            plt.title(layerStrList[i])
            # 创建Colorbar, 并控制其位置.
            ax_cb = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.2)
            plt.colorbar(im, cax=ax_cb, format='%3.1f', orientation='vertical')
            # 保存和显示栅格图.
            plt.savefig(figPath)
            # plt.show()
            plt.clf()
            plt.close()

        # 绘制散点图.
        vsPairList = [(originalLstLayer, targetLstLayer), (originalNdviLayer, originalLstLayer),
                      (originalNdviLayer, targetLstLayer)]
        vaPairStrList = ['Original LST (MODIS) vs Target LST ({0})'.format(sourceLst),
                         'NDVI vs Original LST (MODIS)',
                         'NDVI vs Target LST ({0})'.format(sourceLst)]
        for i in range(len(vsPairList)):
            # 判断散点图是否存在.
            figName = '_'.join([regionDateStr, sourceLst, strategy, 'Scatter', vaPairStrList[i]])
            figPath = path.join(figRegionDateDir, figName + '.png')
            if path.exists(figPath):
                continue
            # 绘制散点图.
            xLayer, yLayer = vsPairList[i][0], vsPairList[i][1]
            if 'NDVI' not in vaPairStrList[i]:
                xEdge = yEdge = [240, 330]
                fig, ax = plt.subplots()
                ax.plot(lstXVector, lstYVector, 'o', mfc='grey')
                ax.plot(xEdge, yEdge, 'r-')
                ax.text(0.05, 0.95, 'RMSE: {0:.3f}'.format(rmse), transform=ax.transAxes)
                ax.text(0.05, 0.90, 'Bias: {0:.3f}'.format(bias), transform=ax.transAxes)
                ax.text(0.05, 0.85, 'R2: {0:.3f}'.format(r2), transform=ax.transAxes)
            else:
                validIndexLayer = np.logical_not(np.isnan(xLayer) | np.less_equal(xLayer, 0) |
                                                 np.isnan(yLayer))
                xVector, yVector = xLayer[validIndexLayer], yLayer[validIndexLayer]
                p1 = poly.polyfit(xVector, yVector, 1)
                xEdge = [0, 1]
                fig, ax = plt.subplots()
                ax.plot(xVector, yVector, 'o', mfc='grey')
                ax.plot(xEdge, poly.polyval(xEdge, p1), 'r-')
                ax.set_xlim(0, 1)
            plt.title(vaPairStrList[i])
            # 保存和显示散点图.
            plt.savefig(figPath)
            # plt.show()
            plt.clf()
            plt.close()

        # 精度评价.
        downscalingStaList.append('{0},{1:.3f},{2:.3f},{3:.3f}\n'.format(dateNum, bias, rmse, r2))

    # 保存精度评价统计文件.
    downscalingCsvName = '_'.join([method, 'Downscaling', rydStr, sourceLst, strategy]) + '.csv'
    downscalingCsvPath = path.join(regionDir, downscalingCsvName)
    with open(downscalingCsvPath, 'w') as f2:
        f2.writelines('YearDate,Bias,RMSE,R2\n')
        f2.writelines(downscalingStaList)
