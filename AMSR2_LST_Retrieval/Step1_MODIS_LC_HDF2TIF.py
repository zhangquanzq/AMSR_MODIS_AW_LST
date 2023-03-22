# coding=utf-8
# 拼接研究区的MODIS地表覆盖数据.
import arcpy
import os
from os import path
from glob import glob

# 预设参数.
modisName = 'MCD12Q1'
region = 'CN'
yearList = range(2012, 2020)

# ArcPy环境设置.
arcpy.env.overwriteOutput = True
arcpy.env.pyramid = 'NONE'

# 路径.
rootDir = r'E:\AMSR2_MODIS_AW_LST\AMSR2_LST_Retrieval'
dataDir = path.join(rootDir, 'Data')

# 输入数据路径.
modisLcTileRegionDir = path.join(dataDir, '{0}_1_Tile{1}_HDF'.format(modisName, region))

# 输出数据路径.
modisLcMosaicRegionDir = path.join(dataDir, '{0}_2_Mosaic{1}_TIF'.format(modisName, region))
if not path.exists(modisLcMosaicRegionDir):
    os.mkdir(modisLcMosaicRegionDir)

# 坐标系统文件路径.
prjPath = path.join(dataDir, 'MODIS_Sinusoidal.prj')

# 按年份提取HDF格式文件中的MODIS LC数据并保存为TIF格式, 然后拼接MODIS LC.
for yearNum in [2020, 2021]:  # yearList:
    # 判断镶嵌后的MODIS LC数据是否存在.
    modisVersion = '061' if yearNum == 2021 else '006'
    modisLcMosaicName = '{0}.A{1}001.{2}.tif'.format(modisName, yearNum, modisVersion)
    modisLcMosaicPath = path.join(modisLcMosaicRegionDir, modisLcMosaicName)
    if arcpy.Exists(modisLcMosaicPath):
        continue

    # 创建临时文件夹.
    tempDir = path.join(rootDir, 'temp_{0}_{1}'.format(modisName, yearNum))
    if not path.exists(tempDir):
        os.mkdir(tempDir)

    # 从HDF文件中提取LC数据, 并保存为TIF格式.
    print(u'提取{0}年{1}的{2}数据.'.format(yearNum, region, modisName))
    modisLcYearDir = path.join(modisLcTileRegionDir, '{0}_{1}XXX_HDF'.format(modisName, yearNum))
    for modisLcHdfPath in glob(path.join(modisLcYearDir, '*.hdf')):
        modisLcTifPath = path.join(tempDir, path.basename(modisLcHdfPath).replace('hdf', 'tif'))
        if not arcpy.Exists(modisLcTifPath):
            arcpy.ExtractSubDataset_management(modisLcHdfPath, modisLcTifPath, '0')

    # 使用地理数据库的镶嵌数据集拼接MODIS LC数据, 并导出.
    print(u'镶嵌{0}年{1}的{2}数据.'.format(yearNum, region, modisName))
    gdbName = 'modisMosaic.gdb'
    mosaicName = 'modisMosaic'
    gdbPath = path.join(tempDir, gdbName)
    mosaicPath = path.join(gdbPath, mosaicName)
    if arcpy.Exists(gdbPath):
        arcpy.Delete_management(gdbPath)
    arcpy.CreateFileGDB_management(tempDir, gdbName)
    arcpy.CreateMosaicDataset_management(gdbPath, mosaicName, prjPath)
    arcpy.AddRastersToMosaicDataset_management(mosaicPath, 'Raster Dataset', tempDir,
                                               filter='*.tif')
    arcpy.CopyRaster_management(mosaicPath, modisLcMosaicPath)

    # 删除临时文件夹.
    arcpy.Delete_management(tempDir)

# 投影, 重分类, 重采样.
lcRemap = arcpy.sa.RemapValue([[1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2], 
                               [9, 2], [10, 2], [11, 3], [12, 2], [13, 5], [14, 2], [15, 4],
                               [16, 1], [17, 3]])
snapRasterPath = path.join(dataDir, r'AMSR2_2_CN_TIF\L3.TB6GHz_10\2012\07',
                           'GW1AM2_20120700_01M_EQMA_L3SGT06HA2220220_BtH.tif')
arcpy.env.snapRaster = snapRasterPath
arcpy.env.workspace = modisLcMosaicRegionDir
for modisLcMosaicPath in glob(path.join(modisLcMosaicRegionDir, 'MCD12Q1*[0-9].tif')):
    modisLcMosaicName = path.basename(modisLcMosaicPath)
    print(u'投影，重分类，重采样 {0}'.format(modisLcMosaicName))

    modisLcGcs = modisLcMosaicName.replace('.tif', '_gcs.tif')
    if not arcpy.Exists(modisLcGcs):
        arcpy.ProjectRaster_management(modisLcMosaicName, modisLcGcs, arcpy.SpatialReference(4326),
                                       'NEAREST', '0.005')

    modisLcReclassify = modisLcGcs.replace('.tif', '_reclsfy.tif')
    if not arcpy.Exists(modisLcReclassify):

        arcpy.sa.Reclassify(modisLcGcs, 'Value', lcRemap, 'NODATA').save(modisLcReclassify)

    modisLcResmpl = modisLcReclassify.replace('.tif', '_resmpl.tif')
    if not arcpy.Exists(modisLcResmpl):
        arcpy.env.extent = arcpy.Raster(snapRasterPath).extent
        arcpy.Resample_management(modisLcReclassify, modisLcResmpl, '0.01', 'NEAREST')
        arcpy.ClearEnvironment('extent')
