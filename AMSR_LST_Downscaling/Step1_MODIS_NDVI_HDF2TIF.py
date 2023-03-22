# coding=utf-8
# 拼接中国地区的MODIS NDVI数据.
import arcpy
import os
import numpy as np
from os import path
from glob import glob

# 标记和预设参数.
# 指定微波传感器的标记. 1表示AMSRE, 2表示AMSR2.
flg1 = 1

sensor = ['AMSRE', 'AMSR2'][flg1 - 1]
yearList = [[2011], range(2022, 2023)][flg1 - 1]
modisType = 'MYD13A2'
region = 'CN'
layerNameList, layerNumList = ['NDVI', 'QA'], ['0', '2']

# ArcPy环境设置.
arcpy.env.overwriteOutput = True
arcpy.env.pyramid = 'NONE'

#%% 路径.
# 根目录.
rootDir = r'F:\AMSR_MODIS_AW_LST'
retrievalDir = path.join(rootDir, 'AMSR_LST_Retrieval')
DownscalingDir = path.join(rootDir, 'AMSR_LST_Downscaling')
dataDir1 = path.join(retrievalDir, 'Data')
dataDir2 = path.join(DownscalingDir, 'Data')

# 输入数据路径.
ndviTileCnDir = path.join(dataDir2, '{0}_1_Tile{1}_HDF'.format(modisType, region))

# 输出数据路径.
ndviPrjCnDir = path.join(dataDir2, '{0}_2_Prj{1}_TIF'.format(modisType, region))
if not path.exists(ndviPrjCnDir):
    os.mkdir(ndviPrjCnDir)

# 投影文件和示例数据路径.
prjPath = path.join(dataDir2, 'MODIS_Sinusoidal.prj')
refRasPath = path.join(dataDir1, r'MYD11A1_2_PrjCN_TIF\MYD11A1_2012XXX_TIF',
                       'MYD11A1.A2012001.LST_Day.tif')

#%% 数据处理过程.
# 按年份提取HDF格式文件中的MODIS NDVI数据并保存为TIF格式, 然后拼接MODIS NDVI.
for yearNum in yearList:
    # 创建存储拼接和投影后NDVI数据的年度文件夹.
    ndviYearTifDir = path.join(ndviPrjCnDir, '{0}_{1}XXX'.format(modisType, yearNum))
    if not path.exists(ndviYearTifDir):
        os.mkdir(ndviYearTifDir)

    # 获取每年的NDVI HDF Tile文件列表.
    ndviHdfPathList = glob(path.join(ndviTileCnDir, '{0}_{1}XXX'.format(modisType, yearNum),
                                     modisType + '*.hdf'))

    # 获取每年的NDVI日期列表.
    ndviDateList = [path.basename(ndviHdfPath)[9:16] for ndviHdfPath in ndviHdfPathList]
    ndviDateType = np.unique(ndviDateList)

    # 拼接和投影有效日期的NDVI数据.
    for ndviDate in ndviDateType:
        print(u'提取和镶嵌{0} {1}的{2}数据.'.format(ndviDate, region, modisType))

        # 获取当天的NDVI数据路径列表.
        ndviHdfPathList2 = [ndviPath for ndviPath in ndviHdfPathList if 'A' + ndviDate in ndviPath]

        # 创建临时文件夹.
        tempDir = path.join(DownscalingDir, 'temp_{0}_{1}'.format(modisType, ndviDate))
        if not path.exists(tempDir):
            os.mkdir(tempDir)

        # 提取和镶嵌各NDVI数据层.
        for i in range(len(layerNameList)):
            layer, layerNum = layerNameList[i], layerNumList[i]

            # 判断镶嵌后的MODIS NDVI文件是否存在.
            ndviPrjName = '{0}.A{1}.061_{2}.tif'.format(modisType, ndviDate, layer)
            ndviPrjPath = path.join(ndviYearTifDir, ndviPrjName)
            if arcpy.Exists(ndviPrjPath):
                continue

            # 从HDF文件中提取各NDVI数据层, 并保存为TIF格式.
            for ndviHdfPath in ndviHdfPathList2:
                ndviTifName = path.basename(ndviHdfPath).replace('.hdf', '_{0}.tif'.format(layer))
                ndviTifPath = path.join(tempDir, ndviTifName)
                if not arcpy.Exists(ndviTifPath):
                    arcpy.ExtractSubDataset_management(ndviHdfPath, ndviTifPath, layerNum)

            # 使用地理数据库的镶嵌数据集拼接MODIS LC数据, 并导出.
            gdbName = 'modisMosaic.gdb'
            mosaicName = layer + 'Mosaic'
            gdbPath = path.join(tempDir, gdbName)
            mosaicPath = path.join(gdbPath, mosaicName)
            if arcpy.Exists(gdbPath):
                arcpy.Delete_management(gdbPath)

            arcpy.CreateFileGDB_management(tempDir, gdbName)
            arcpy.CreateMosaicDataset_management(gdbPath, mosaicName, prjPath)
            arcpy.AddRastersToMosaicDataset_management(mosaicPath, 'Raster Dataset', tempDir,
                                                       filter='*{0}.tif'.format(layer))

            ndviTempPath = ndviPrjPath.replace('.tif', '2.tif')
            arcpy.env.snapRaster = refRasPath
            arcpy.ProjectRaster_management(mosaicPath, ndviTempPath, refRasPath, 'NEAREST', '0.01')
            arcpy.ClearEnvironment('snapRaster')
            arcpy.Clip_management(ndviTempPath, '#', ndviPrjPath, refRasPath)
            arcpy.Delete_management(ndviTempPath)

        # 删除临时文件夹.
        arcpy.Delete_management(tempDir)
