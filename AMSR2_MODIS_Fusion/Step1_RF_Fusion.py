# 随机森林融合降尺度后AMSR2 LST和MODIS LST.

import numpy as np
import os
import pandas as pd
import sklearn.ensemble
from functions import *
from glob import glob
from os import path
from osgeo import gdal
from scipy.ndimage import gaussian_filter

# %% 自定义函数.
# 导出TIF格式文档.


# %% 标记和预设参数.
# 指定区域的标记. 1表示云贵高原, 2表示晋南豫西, 3表示内蒙古, 4表示北大河流域, 5表示那曲地区.
flg1 = 2
# 指定昼夜的标记. 1表示白天, 2表示夜晚.
flg2 = 1
# 指定模型采用哪种策略的降尺度结果. 1表示逐级降尺度, 2表示直接降尺度.
flg3 = 1
# 指定是否将RF回归的LST进行高斯平滑. 1表示是, 2表示否.
flg4 = 1

# 预设参数.
testYear = '2013'
gsFilterSigma = 0.7
region = ['YGGY', 'JNYX', 'NMG', 'BDH', 'Naqu'][flg1 - 1]
dtm = ['Elev', 'Slp', 'Slp', 'Elev', 'Slp'][flg1 - 1]
dayNight = ['Day', 'Night'][flg2 - 1]
strategy = ['Stepwise', 'Straight'][flg3 - 1]

# %% 路径.
rootDir = r'E:\AMSR2_MODIS_All-weather_LST'
dataDir1 = path.join(rootDir, r'AMSR2_LST_Downscaling\Data')
dataDir2 = path.join(rootDir, r'AMSR2_MODIS_Fusion\Data')

regionDir = path.join(dataDir2, 'Region_{0}_{1}'.format(region, dayNight))
srtmDir = path.join(dataDir1, 'Region_' + region, 'SRTM_' + region)

rfFolder = ['0_RF_GSFilter_{0}'.format(gsFilterSigma), '0_RF'][flg4 - 1]
rfDir = path.join(regionDir, rfFolder)
if not path.exists(rfDir):
    os.mkdir(rfDir)

# %% 融合过程.

print('使用随机森林回归融合{0}地区的全天候地表温度数据.'.format(region))
regionDateDirList = glob(path.join(regionDir, '*Stepwise_Raster'))
for regionDateDir in regionDateDirList:
    dateStr = path.basename(regionDateDir)[:8]
    dateStr2 = ymd2yday(dateStr)

    # 输出数据路径.
    halfName = 'PredictedLst_{0}_{1}_{2}_{3}'.format(region, dateStr, dayNight, strategy)
    rfLstPath = path.join(rfDir, '{0}_Rf.tif'.format(halfName))
    rfSmoothLstPath = path.join(rfDir, '{0}_RfSmooth.tif'.format(halfName))
    fusionLstPath = path.join(rfDir, '{0}_Fusion.tif'.format(halfName))

    # 输入数据路径.
    modisLstPath = glob(path.join(regionDir, 'MYD11A1*0.01.tif'))[0]
    amsr2LstPath = glob(path.join(regionDir, 'AMSR2*0.01.tif'))[0]
    modisNdviPath = glob(path.join(regionDir, 'MYD13A2*0.01.tif'))[0]
    srtmPath = path.join(srtmDir, 'SRTM_0.01_{0}_{1}.tif'.format(region, dtm))

    # 打开输入数据.
    modisLstObj = gdal.Open(modisLstPath)
    amsr2LstObj = gdal.Open(amsr2LstPath)
    modisNdviObj = gdal.Open(modisNdviPath)
    srtmObj = gdal.Open(srtmPath)

    # 获取输入数据属性.
    modisLstColN, modisLstRowN = modisLstObj.RasterXSize, modisLstObj.RasterYSize
    amsreLstColN, amsreLstRowN = amsr2LstObj.RasterXSize, amsr2LstObj.RasterYSize
    modisNdviColN, modisNdviRowN = modisNdviObj.RasterXSize, modisNdviObj.RasterYSize
    srtmColN, srtmRowN = srtmObj.RasterXSize, srtmObj.RasterYSize

    modisLstGeotrans = modisLstObj.GetGeoTransform()
    amsr2LstGeotrans = amsr2LstObj.GetGeoTransform()
    modisNdviGeotrans = modisNdviObj.GetGeoTransform()
    srtmGeotrans = srtmObj.GetGeoTransform()

    # 各输入数据左上角地理坐标.
    modisLstXMin, modisLstYMax = modisLstGeotrans[0], modisLstGeotrans[3]
    amsr2LstXMin, amsr2LstYMax = amsr2LstGeotrans[0], amsr2LstGeotrans[3]
    modisNdviXMin, modisNdviYMax = modisNdviGeotrans[0], modisNdviGeotrans[3]
    srtmXMin, srtmYMax = srtmGeotrans[0], srtmGeotrans[3]

    # 各输入数据右下角地理坐标.
    modisLstXMax = modisLstGeotrans[0] + modisLstColN * modisLstGeotrans[1]
    modisLstYMin = modisLstGeotrans[3] + modisLstRowN * modisLstGeotrans[5]
    amsr2LstXMax = amsr2LstGeotrans[0] + amsreLstColN * amsr2LstGeotrans[1]
    amsr2LstYMin = amsr2LstGeotrans[3] + amsreLstRowN * amsr2LstGeotrans[5]
    modisNdviXMax = modisNdviGeotrans[0] + modisNdviColN * modisNdviGeotrans[1]
    modisNdviYMin = modisNdviGeotrans[3] + modisNdviRowN * modisNdviGeotrans[5]
    srtmXMax = srtmGeotrans[0] + srtmColN * srtmGeotrans[1]
    srtmYMin = srtmGeotrans[3] + srtmRowN * srtmGeotrans[5]

    # 计算各输入数据的公共区域边界坐标.
    xMin = max([modisLstXMin, amsr2LstXMin, modisNdviXMin, srtmXMin])
    xMax = min([modisLstXMax, amsr2LstXMax, modisNdviXMax, srtmXMax])
    yMin = max([modisLstYMin, amsr2LstYMin, modisNdviYMin, srtmYMin])
    yMax = min([modisLstYMax, amsr2LstYMax, modisNdviYMax, srtmYMax])

    # 读取输入数据矩阵.
    modisLstLayer = modisLstObj.ReadAsArray()
    amsr2LstLayer = amsr2LstObj.ReadAsArray()
    modisNdviLayer = modisNdviObj.ReadAsArray()
    srtmLayer = srtmObj.ReadAsArray()

    # 裁剪输入数据的公共区域.
    modisLstLayer = modisLstLayer[(modisLstYMax - yMax) / 0.01: (modisLstYMax - yMin) / 0.01,
                                  (xMin - modisLstXMin) / 0.01: (xMax - modisLstXMin) / 0.01]
    amsr2LstLayer = amsr2LstLayer[(amsr2LstYMax - yMax) / 0.01: (amsr2LstYMax - yMin) / 0.01,
                                  (xMin - amsr2LstXMin) / 0.01: (xMax - amsr2LstXMin) / 0.01]
    modisNdviLayer = modisNdviLayer[(modisNdviYMax - yMax) / 0.01: (modisNdviYMax - yMin) / 0.01,
                                    (xMin - modisNdviXMin) / 0.01: (xMax - modisNdviXMin) / 0.01]
    srtmLayer = srtmLayer[(srtmYMax - yMax) / 0.01: (srtmYMax - yMin) / 0.01,
                          (xMin - srtmXMin) / 0.01: (xMax - srtmXMin) / 0.01]



