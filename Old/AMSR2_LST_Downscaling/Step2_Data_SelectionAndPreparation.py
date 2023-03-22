# coding=utf-8
import arcpy
import os
import numpy as np
import pandas as pd
import sys
import shutil
import datetime as dt
from glob import glob
from os import path
from functions import functions as fn

# 标记与预设参数.
# 指定区域的标记. 1表示内蒙古, 2表示晋南豫西, 3表示云贵高原, 4表示北大河, 5表示那曲.
flg1 = 2
# 指定昼夜的标记. 1表示白天, 2表示夜晚.
flg2 = 1
# 指定降尺度策略的标记. 1表示逐级降尺度, 2表示直接降尺度.
flg3 = 1

region = ['NMG', 'JNYX', 'YGGY', 'BDH', 'Naqu'][flg1 - 1]
dayNight = ['Day', 'Night'][flg2 - 1]
cellsizeList = [['0.1', '0.03', '0.01'], ['0.1', '0.01']][flg3 - 1][::-1]
dayNightRegion = '{0}_{1}'.format(dayNight, region)

yearNum = 2014
cloudPctEdges = [0, 0.1]  # 针对降尺度试验.
topoParams = ['Elev', 'Slp']
# cloudPctEdges = [0.1, 0.9]  # 针对融合试验.

# ArcPy环境参数设置.
arcpy.env.overwriteOutput = True
arcpy.env.pyramid = 'NONE'

# %% 数据路径.
rootDir1 = r'J:\AMSR2_LST_Retrieval'
rootDir2 = r'J:\AMSR2_LST_Downscaling'
dataDir1 = path.join(rootDir1, 'Data')
dataDir2 = path.join(rootDir2, 'Data')

amsr2LstDir = path.join(dataDir1, 'AMSR2_4_LstCn_TIF')
modisLstDir = path.join(dataDir1, 'MYD11A1_2_PrjCN_TIF')
ndviDir = path.join(dataDir2, 'MYD13A2_2_PrjCN_TIF')
regionExtentPath = path.join(dataDir2, 'Feature', '{0}_Extent.shp'.format(region))
regionBufferPath = path.join(dataDir2, 'Feature', '{0}_Buffer.shp'.format(region))

# 研究区和年份文件夹路径.
regionDir = path.join(dataDir2, 'Region_{0}'.format(region))
if not path.exists(regionDir):
    os.mkdir(regionDir)
yearDir = path.join(regionDir, '{0}_{1}_{2}'.format(region, yearNum, dayNight))
if not path.exists(yearDir):
    os.mkdir(yearDir)

# 读取MODIS地表温度数据文件名和日期列表.
modisLstYearDir = path.join(modisLstDir, 'MYD11A1_{0}XXX_TIF'.format(yearNum))
modisLstPathList = glob(path.join(modisLstYearDir, 'MYD11A1*LST_{0}.tif'.format(dayNight)))
modisQcPathList = glob(path.join(modisLstYearDir, 'MYD11A1*QC_{0}.tif'.format(dayNight)))
modisDateList = [fn.yday2ymd(path.basename(lstPath)[9:16]) for lstPath in modisLstPathList]
modisQcDateList = [fn.yday2ymd(path.basename(qcPath)[9:16]) for qcPath in modisQcPathList]
if modisDateList != modisQcDateList:
    print(u'MODIS LST数据与QC数据不匹配, 请检查, 程序退出.')
    sys.exit()

# 读取AMSRE/2地表温度数据文件名和日期列表.
amsr2LstYearDir = path.join(amsr2LstDir, 'AMSR2_LST_{0}XXXX_TIF'.format(yearNum))
amsr2LstPathList = glob(path.join(amsr2LstYearDir, 'AMSR2*{0}*.TIF'.format(dayNight)))
amsr2DateList = [path.basename(amsr2Path).split('_')[3][:-4] for amsr2Path in amsr2LstPathList]

# 读取MODIS NDVI数据文件名和日期列表.
ndviYearDir = path.join(ndviDir, 'MYD13A2_{0}XXX_TIF'.format(yearNum))
ndviPastYearDir = path.join(ndviDir, 'MYD13A2_{0}XXX_TIF'.format(yearNum - 1))
ndviNextYearDir = path.join(ndviDir, 'MYD13A2_{0}XXX_TIF'.format(yearNum + 1))

ndvi361Path = path.join(ndviPastYearDir, 'MYD13A2.A{0}361.061_NDVI.tif'.format(yearNum - 1))
ndvi009Path = path.join(ndviNextYearDir, 'MYD13A2.A{0}009.061_NDVI.tif'.format(yearNum + 1))
ndviQa361Path = path.join(ndviPastYearDir, 'MYD13A2.A{0}361.061_QA.tif'.format(yearNum - 1))
ndviQa009Path = path.join(ndviNextYearDir, 'MYD13A2.A{0}009.061_QA.tif'.format(yearNum + 1))
if not (path.exists(ndvi361Path) and path.exists(ndvi009Path) and path.exists(ndviQa361Path) and
        path.exists(ndviQa009Path)):
    print(u'上一年最后一个或下一年第一个MODIS NDVI数据与QA数据不存在, 请检查, 程序退出.')
    sys.exit()

ndviPathList = glob(path.join(ndviYearDir, 'MYD13A2*NDVI.tif'))
ndviPathList.insert(0, ndvi361Path)
ndviPathList.append(ndvi009Path)

ndviQaPathList = glob(path.join(ndviYearDir, 'MYD13A2*QA.tif'))
ndviQaPathList.insert(0, ndviQa361Path)
ndviQaPathList.append(ndviQa009Path)

ndviDateList = [fn.yday2ymd(path.basename(ndviPath)[9:16]) for ndviPath in ndviPathList]
ndviDateList = [dt.date(int(ndviDate[0:4]), int(ndviDate[4:6]), int(ndviDate[6:8]))
                for ndviDate in ndviDateList]
ndviQaDateList = [fn.yday2ymd(path.basename(ndviQaPath)[9:16]) for ndviQaPath in ndviQaPathList]
ndviQaDateList = [dt.date(int(ndviQaDate[0:4]), int(ndviQaDate[4:6]), int(ndviQaDate[6:8]))
                  for ndviQaDate in ndviQaDateList]
if ndviDateList != ndviQaDateList:
    print(u'MODIS NDVI数据与QA数据不匹配, 请检查, 程序退出.')
    sys.exit()

# 筛选数据指标统计文件路径.
staRegionCsvName = 'DateSelect_{0}_{1}_{2}.csv'.format(region, yearNum, dayNight)
staRegionCsvPath = path.join(regionDir, staRegionCsvName)

# %% SRTM数据准备. 裁剪研究区的SRTM高程, 坡度数据, 并生成分辨率序列.
srtmDir = path.join(dataDir1, 'SRTM')
srtmRegionDir = path.join(regionDir, 'SRTM_{0}'.format(region))
if not path.exists(srtmRegionDir):
    os.mkdir(srtmRegionDir)
arcpy.env.snapRaster = path.join(dataDir1, r'AMSR2_2_CN_TIF\L3.TB6GHz_10\2012\07',
                                 'GW1AM2_20120703_01D_EQMA_L3SGT06HA2220220_BtH.tif')
for topo in topoParams:  # elev, slp
    srtmTopoPath = path.join(srtmDir, 'SRTM_0d01_CN_{0}.tif'.format(topo))
    srtmTopoExtentPath = path.join(srtmRegionDir, 'SRTM_0.01_{0}_{1}.tif'.format(region, topo))
    if not path.exists(srtmTopoExtentPath):
        arcpy.Clip_management(srtmTopoPath, '#', srtmTopoExtentPath, regionExtentPath)
    srtmTopoBufferPath = path.join(srtmRegionDir, 'SRTM_0.01_buffer_{0}.tif'.format(topo))
    if not path.exists(srtmTopoBufferPath):
        arcpy.Clip_management(srtmTopoPath, '#', srtmTopoBufferPath, regionBufferPath)
    for cellsize in cellsizeList[1:]:
        srtmTopoResample1Name = 'SRTM_{0}_{1}_{2}.tif'.format(cellsize, region, topo)
        srtmTopoResample1Path = path.join(srtmRegionDir, srtmTopoResample1Name)
        if path.exists(srtmTopoResample1Path):
            continue

        srtmTopoResample2Name = 'SRTM_{0}_buffer_{1}.tif'.format(cellsize, topo)
        srtmTopoResample2Path = path.join(srtmRegionDir, srtmTopoResample2Name)
        if path.exists(srtmTopoResample2Path):
            arcpy.Delete_management(srtmTopoResample2Path)

        arcpy.Resample_management(srtmTopoBufferPath, srtmTopoResample2Path, cellsize, 'BILINEAR')
        arcpy.Clip_management(srtmTopoResample2Path, '#', srtmTopoResample1Path, regionExtentPath)
        arcpy.Delete_management(srtmTopoResample2Path)
    arcpy.Delete_management(srtmTopoBufferPath)
arcpy.ClearEnvironment('snapRaster')

# %% 筛选可用日期的数据.
# 创建年度日期列表.
yearDateRange = pd.date_range('{0}0101'.format(yearNum), '{0}1231'.format(yearNum))
yearDateList = [dt.datetime.strftime(yearDate, '%Y%m%d') for yearDate in list(yearDateRange)]

# 数据日期筛选.
recordList = []
print(u'筛选{0}区{1}年{2}的数据.'.format(region, yearNum, dayNight))
for yearDate in yearDateList:
    # 获取均有数据的AMSR2和MODIS地表温度文件路径, 跳过至少有其中一个没有数据的日期.
    amsr2DateIndex = amsr2DateList.index(yearDate) if (yearDate in amsr2DateList) else -1
    modisDateIndex = modisDateList.index(yearDate) if (yearDate in modisDateList) else -1
    if -1 not in [amsr2DateIndex, modisDateIndex]:
        amsr2LstPath = amsr2LstPathList[amsr2DateIndex]
        modisLstPath = modisLstPathList[modisDateIndex]
    else:
        continue

    # 创建AMSR2和MODIS都有数据的日期文件夹.
    yearDateDir = path.join(yearDir, '{0}_{1}_{2}'.format(region, yearDate, dayNight))
    if not path.exists(yearDateDir):
        os.mkdir(yearDateDir)

    # 裁剪研究区范围的AMSR2数据, temp数据用于处理Matlab输出的影像中的Nodata在ArcGIS显示不正常的问题.
    amsr2LstRegionName = path.basename(amsr2LstPath).replace(dayNight, dayNightRegion)
    amsr2LstRegionPath = path.join(yearDateDir, amsr2LstRegionName)
    if not arcpy.Exists(amsr2LstRegionPath):
        arcpy.Clip_management(amsr2LstPath, '#', amsr2LstRegionPath, regionExtentPath)

    # 如果AMSR2地表温度影像中存在Nodata或0, 则舍弃当日的数据.
    amsr2LstArray = arcpy.RasterToNumPyArray(amsr2LstRegionPath, nodata_to_value=0)
    if np.sum(amsr2LstArray == 0) > 0:
        arcpy.Delete_management(yearDateDir)
        continue

    # 裁剪研究区范围的MODIS数据, 计算其云覆盖面积, 确定数据可用性. 若不可用删除当天的AMSR2和MODIS数据.
    modisLstName = path.basename(modisLstPath)
    modisLstRegionName = modisLstName.replace(dayNight, dayNightRegion + '_0.01')
    modisLstRegionPath = path.join(yearDateDir, modisLstRegionName)
    if not arcpy.Exists(modisLstRegionPath):
        arcpy.Clip_management(modisLstPath, '#', modisLstRegionPath, regionExtentPath)
        allNodata = arcpy.GetRasterProperties_management(modisLstRegionPath, 'ALLNODATA')
        allNodata = allNodata.getOutput(0)
        if allNodata == '0':
            modisLstRegionRas = arcpy.Raster(modisLstRegionPath) * 0.02
            modisLstSetNanRas = arcpy.sa.SetNull(modisLstRegionRas, modisLstRegionRas, 'Value = 0')
            modisLstSetNanRas.save(modisLstRegionPath)
            del modisLstSetNanRas
    modisLstRegionLayer = arcpy.RasterToNumPyArray(modisLstRegionPath, nodata_to_value=0)
    modisNodataLayer = (modisLstRegionLayer == 0)
    modisNodataPct = float(np.sum(modisNodataLayer)) / np.size(modisNodataLayer)
    if not cloudPctEdges[0] < modisNodataPct < cloudPctEdges[1]:
        arcpy.Delete_management(yearDateDir)
        continue

    # 创建存放重采样过程数据的临时文件夹.
    tempDir = path.join(dataDir2, 'Temp_{0}_{1}_{2}'.format(region, yearDate, dayNight))
    if not path.exists(tempDir):
        os.mkdir(tempDir)

    # 若MODIS影像的Nodata像元数在指定范围内, 重采样生成MODIS LST分辨率序列.
    modisLstRegionPathList = [modisLstRegionPath.replace('0.01', cs) for cs in cellsizeList]
    modisLstRegionValidList = [arcpy.Exists(modisPath) for modisPath in modisLstRegionPathList]
    if False in modisLstRegionValidList:
        modisLstBufferName = modisLstName.replace(dayNight, dayNight + '_Buffer_0.01')
        modisLstBufferPath = path.join(tempDir, modisLstBufferName)
        arcpy.Clip_management(modisLstPath, '#', modisLstBufferPath, regionBufferPath)
        modisLstBufferRas = arcpy.Raster(modisLstBufferPath) * 0.02
        modisLstSetNanRas = arcpy.sa.SetNull(modisLstBufferRas, modisLstBufferRas, 'Value = 0')
        modisLstSetNanRas.save(modisLstBufferPath)

        arcpy.env.snapRaster = amsr2LstRegionPath  # 这个捕捉栅格估计有问题.
        fn.resamplePixelSeries(modisLstBufferPath, regionExtentPath, yearDateDir, cellsizeList,
                               region, 'initial')
        arcpy.ClearEnvironment('snapRaster')
        arcpy.Delete_management(modisLstBufferPath)

    # 准备NDVI数据.
    # 获取离MODIS LST最近的NDVI数据日期.
    yearDate2 = dt.date(int(yearDate[0:4]), int(yearDate[4:6]), int(yearDate[6:8]))
    ndviDateDiff = [abs((yearDate2 - ndviDate).days) for ndviDate in ndviDateList]

    # NDVI数据裁剪与质量控制. 裁剪使用研究区缓冲区范围, 避免重采样时边界像元插值问题.
    ndviPath = ndviPathList[ndviDateDiff.index(min(ndviDateDiff))]
    ndviRegionName = path.basename(ndviPath).replace('NDVI', 'NDVI_{0}_0.01'.format(region))
    ndviRegionPath = path.join(yearDateDir, ndviRegionName)

    ndviRegionPathList = [ndviRegionPath.replace('0.01', cs) for cs in cellsizeList]
    ndviRegionValidList = [arcpy.Exists(nPath) for nPath in ndviRegionPathList]
    if False in ndviRegionValidList:
        ndviBufferName = path.basename(ndviPath).replace('NDVI', 'NDVI_Buffer_0.01')
        ndviBufferPath = path.join(tempDir, ndviBufferName)
        if not arcpy.Exists(ndviBufferPath):
            # 裁剪NDVI数据.
            ndviClipName = path.basename(ndviPath).replace('NDVI', 'NDVI_Clip_0.01')
            ndviClipPath = path.join(tempDir, ndviClipName)
            arcpy.Clip_management(ndviPath, '#', ndviClipPath, regionBufferPath)
            ndviClipRaster = arcpy.Raster(ndviClipPath)
            lowerLeft = ndviClipRaster.extent.lowerLeft

            # 裁剪QA数据.
            ndviQaPath = ndviQaPathList[ndviDateDiff.index(min(ndviDateDiff))]
            ndviQaClipName = path.basename(ndviQaPath).replace('QA', 'QA_Clip_0.01')
            ndviQaClipPath = path.join(tempDir, ndviQaClipName)
            arcpy.Clip_management(ndviQaPath, '#', ndviQaClipPath, regionBufferPath)

            # 将QA数据转为Numpy array对NDVI进行质量控制, 并导出控制后的NDVI数据.
            ndviQaArray = arcpy.RasterToNumPyArray(ndviQaClipPath)
            qaRowN, qaColN = ndviQaArray.shape
            ndviQaBinArray = np.reshape([np.binary_repr(ndviQaArray[i, j], 16) for i in
                                         range(qaRowN) for j in range(qaColN)], (qaRowN, qaColN))
            ndviQaBinCutArray1 = np.reshape([ndviQaBinArray[i, j][-2:] for i in range(qaRowN)
                                             for j in range(qaColN)], (qaRowN, qaColN))
            ndviQaBinCutArray2 = np.reshape([ndviQaBinArray[i, j][-6:-4] for i in range(qaRowN)
                                             for j in range(qaColN)], (qaRowN, qaColN))
            ndviIndexLayer = ((ndviQaBinCutArray1 == '00') | (ndviQaBinCutArray1 == '01') |
                              (ndviQaBinCutArray2 != '11')).astype(int)
            ndviIndexRaster = arcpy.NumPyArrayToRaster(ndviIndexLayer, lowerLeft, 0.01, 0.01)
            (ndviClipRaster * ndviIndexRaster).save(ndviBufferPath)

            # 删除质量控制前的NDVI和QA Clip文件.
            arcpy.Delete_management(ndviClipPath)
            arcpy.Delete_management(ndviQaClipPath)

        arcpy.Clip_management(ndviBufferPath, '#', ndviRegionPath, regionExtentPath)
        arcpy.env.snapRaster = amsr2LstRegionPath  # 这个捕捉栅格估计有问题.
        fn.resamplePixelSeries(ndviBufferPath, regionExtentPath, yearDateDir, cellsizeList,
                               region, 'initial')
        arcpy.ClearEnvironment('snapRaster')
        arcpy.Delete_management(ndviBufferPath)

    arcpy.Delete_management(tempDir)

    # 获取AMSR和MODIS地表温度数据的重叠区域.
    modisLstLowestRaster = arcpy.Raster(modisLstRegionPathList[-1])
    amsr2LstRaster = arcpy.Raster(amsr2LstRegionPath)
    xMin = max([modisLstLowestRaster.extent.XMin, amsr2LstRaster.extent.XMin])
    yMin = max([modisLstLowestRaster.extent.YMin, amsr2LstRaster.extent.YMin])
    xMax = min([modisLstLowestRaster.extent.XMax, amsr2LstRaster.extent.XMax])
    yMax = min([modisLstLowestRaster.extent.YMax, amsr2LstRaster.extent.YMax])
    lowerLeft = arcpy.Point(xMin, yMin)
    colN = (xMax - xMin) / float(cellsizeList[-1])
    rowN = (yMax - yMin) / float(cellsizeList[-1])
    modisLstArray = arcpy.RasterToNumPyArray(modisLstLowestRaster, lowerLeft, colN, rowN, 0)
    amsr2LstArray = arcpy.RasterToNumPyArray(amsr2LstRaster, lowerLeft, colN, rowN, 0)

    # 计算AMSR和MODIS地表温度数据之间的相关系数.
    amsr2LstVector = amsr2LstArray.reshape(np.size(amsr2LstArray), 1)
    modisLstVector = modisLstArray.reshape(np.size(modisLstArray), 1)
    validIndexVector = (amsr2LstVector != 0) & (modisLstVector != 0)
    amsr2LstVector = amsr2LstVector[validIndexVector]
    modisLstVector = modisLstVector[validIndexVector]
    cc = np.corrcoef(amsr2LstVector, modisLstVector)[0, 1]
    recordList.append('{0},{1:.3f},{2:.3f}\n'.format(yearDate, 1 - modisNodataPct, cc))
    print(u'{0}, {1}, {2}, Area: {3:.3f}, R: {4:.3f}'
          .format(region, yearDate, dayNight, 1 - modisNodataPct, cc))

# 保存统计数据到CSV文件.
with open(staRegionCsvPath, 'w') as f:
    f.writelines('YearDate,PixelPercent,R\n')
    f.writelines(recordList)
