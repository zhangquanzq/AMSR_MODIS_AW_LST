# %% coding=utf-8
# 将分年份, 日期组织的MODIS地表温度HDF文件拼接成一幅TIF文件, 然后投影转换.
# 注意: 此程序拼接得到的TIF文件中, 原HDF为nodata的数值变成数字, 需要下一步将其转为Nodata.
import arcpy
import os
from collections import Counter
from glob import glob
from os import path

'''
MxD11A1数据层: ['LST_Day','QC_Day','Time_Day','Angle_Day','LST_Night','QC_Night','Time_Night',
               'Angle_Night', 'Emis_31', 'Emis_32', 'Clear_Day', 'Clear_Night']
'''

# ArcPy环境参数设置.
arcpy.env.overwriteOutput = True
arcpy.env.pyramid = 'NONE'

# 预设参数.
region = 'CN'
modisName = 'MYD11A1'
yearList = [2005, 2009, 2010, 2011]
layerList = ['LST_Day', 'QC_Day', 'LST_Night', 'QC_Night', 'Emis_31', 'Emis_32']
layerNumList = ['0', '1', '4', '5', '8', '9']
layerN = len(layerList)

# %% 路径.
# 根目录.
rootDir = r'J:\AMSRE_MODIS_AW_LST'
dataDir = path.join(rootDir, r'AMSRE_LST_Retrieval\Data')

# 输入数据路径.
modisTilesDir = path.join(dataDir, '{0}_1_Tile{1}_HDF'.format(modisName, region))

# 输出数据路径.
modisMosaicDir = path.join(dataDir, '{0}_2_Mosaic{1}_TIF'.format(modisName, region))
if not path.exists(modisMosaicDir):
    os.makedirs(modisMosaicDir)

modisPrjDir = path.join(dataDir, '{0}_2_Prj{1}_TIF'.format(modisName, region))
if not path.exists(modisPrjDir):
    os.makedirs(modisPrjDir)

# 空间坐标系统文件, 以及栅格捕捉影像路径.
prjPath = path.join(dataDir, 'Modis_Sinusoidal.prj')

snapRasterPath = path.join(dataDir, r'AMSRE_2_CN_TIF\AMSRE_2004XXXX\AMSRE_20040103',
                           'AMSRE_D25_20040103A_v03_06H.tif')

# %% 数据处理.
# 按年份拼接MODIS数据图层.
for year in yearList:  # [2020]:  # :
    modisTilesYearDir = path.join(modisTilesDir, '{0}_{1}XXX'.format(modisName, year))

    # 创建拼接后的MODIS数据的年度文件夹.
    modisMosaicYearDir = path.join(modisMosaicDir, '{0}_{1}XXX'.format(modisName, year))
    if not path.exists(modisMosaicYearDir):
        os.makedirs(modisMosaicYearDir)

    # 获取已完成所有数据层拼接的日期列表.
    arcpy.env.workspace = modisMosaicYearDir
    modisMosaicLayerDateList, modisMosaicCompleteDateList = [], []
    for layerName in layerList:
        mosaicLayerList = arcpy.ListRasters('{0}*{1}.tif'.format(modisName, layerName))
        for mosaicLayer in mosaicLayerList:
            modisMosaicLayerDateList.append(mosaicLayer.split('.')[1][1:])
    modisMosaicDateCounter = Counter(modisMosaicLayerDateList)
    modisMosaicDateNumberList = modisMosaicDateCounter.values()
    modisMosaicDateStrList = modisMosaicDateCounter.keys()
    for i in range(len(modisMosaicDateNumberList)):
        if modisMosaicDateNumberList[i] == layerN:
            modisMosaicCompleteDateList.append(modisMosaicDateStrList[i])

    # 提取MODIS数据层并拼接它们, 跳过已完成所有图层拼接的日期.
    for modisTilesYearDayPath in glob(path.join(modisTilesYearDir, '*')):
        yearDay = path.basename(modisTilesYearDayPath).split('_')[-1]
        if yearDay in modisMosaicCompleteDateList:
            continue  # 跳过已完成所有数据层拼接的日期.

        # 从MODIS HDF文件中提取数据层，并保存到临时文件夹中.
        print(u'提取{0} {1} {2}'.format(region, yearDay, modisName))
        tmpDir = path.join(rootDir, 'tmp_{0}_{1}'.format(modisName, yearDay))
        if not path.exists(tmpDir):
            os.mkdir(tmpDir)
        for modisTilePath in glob(path.join(modisTilesYearDayPath, '*.hdf')):
            for i in range(layerN):
                layerName = layerList[i]
                modisMosaicTifName = '{0}.A{1}.{2}.tif'.format(modisName, yearDay, layerName)
                modisMosaicTifPath = path.join(modisMosaicYearDir, modisMosaicTifName)
                if not arcpy.Exists(modisMosaicTifPath):
                    modisTileName = path.basename(modisTilePath)
                    modisLayerName = modisTileName.replace('.hdf', '_{0}.tif'.format(layerName))
                    modisLayerPath = path.join(tmpDir, modisLayerName)
                    if not arcpy.Exists(modisLayerPath):
                        try:
                            arcpy.ExtractSubDataset_management(modisTilePath, modisLayerPath,
                                                               layerNumList[i])
                        except:
                            errorHDFText = path.join(rootDir, modisName,
                                                     'errorHDF{0}.txt'.format(yearDay))
                            with open(errorHDFText, 'a') as f:
                                f.write(modisTilePath + '\n')

        # 拼接提取的数据层.
        print(u'拼接{0} {1} {2}'.format(region, yearDay, modisName))
        for layerName in layerList:
            arcpy.env.workspace = tmpDir
            modisTileList = arcpy.ListRasters("{0}*_{1}.tif".format(modisName, layerName))
            modisMosaicTifName = '{0}.A{1}.{2}.tif'.format(modisName, yearDay, layerName)
            modisMosaicTifPath = path.join(modisMosaicYearDir, modisMosaicTifName)
            modisTileStr = ';'.join(modisTileList)
            if not arcpy.Exists(modisMosaicTifPath):
                gdbName = '{0}_{1}.gdb'.format(modisName, yearDay)
                gdbPath = path.join(rootDir, gdbName)
                mosaicPath = path.join(gdbPath, layerName)
                arcpy.CreateFileGDB_management(rootDir, gdbName)
                arcpy.CreateMosaicDataset_management(gdbPath, layerName, prjPath)
                arcpy.AddRastersToMosaicDataset_management(mosaicPath, 'Raster Dataset', tmpDir,
                                                           filter='*{0}.tif'.format(layerName))
                # 以下pixel_type和nodata_value仅是Emis的参数设置. , nodata_value='255'
                arcpy.CopyRaster_management(mosaicPath, modisMosaicTifPath,
                                            pixel_type='16_BIT_UNSIGNED')
                if 'LST' in path.basename(modisMosaicTifPath):
                    arcpy.SetRasterProperties_management(modisMosaicTifPath, nodata='1 0')
                arcpy.Delete_management(gdbPath)

        # 删除临时文件夹.
        arcpy.Delete_management(tmpDir)

# %% 投影转换.
arcpy.env.snapRaster = snapRasterPath
for year in yearList:
    # 创建存放投影转换后的MODIS数据的年度文件夹.
    modisPrjYearDir = path.join(modisPrjDir, '{0}_{1}XXX_TIF'.format(modisName, year))
    if not path.exists(modisPrjYearDir):
        os.makedirs(modisPrjYearDir)

    modisMosaicYearDir = path.join(modisMosaicDir, '{0}_{1}XXX_TIF'.format(modisName, year))
    for modisMosaicTifPath in glob(path.join(modisMosaicYearDir, '*.tif')):
        modisTifName = path.basename(modisMosaicTifPath)
        modisPrjTifPath = path.join(modisPrjYearDir, modisTifName)
        if not path.exists(modisPrjTifPath):
            print(modisTifName)
            arcpy.ProjectRaster_management(modisMosaicTifPath, modisPrjTifPath,
                                           arcpy.SpatialReference(4326), 'NEAREST', '0.01')
