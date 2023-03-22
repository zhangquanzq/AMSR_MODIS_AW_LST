# coding=utf-8
# 拼接研究区的MODIS地表覆盖数据.
import arcpy
import os
from os import path
from glob import glob

# 标记.
# 指定区域的标记. 1表示Greenland, 2表示Antarctic.
flg1 = 1

# 预设参数.
modisName = 'MCD12Q1'
region = ['Greenland', 'Antarctic'][flg1 - 1]
exampleRasName = ['AMSRE_Greenland_Example.tif', 'AMSRE_Antarctic_Example.tif'][flg1 - 1]
yearList = range(2001, 2021)


# 路径.
rootDir = r'F:\AMSR_LST_IceSheet'
dataDir = path.join(rootDir, 'Data')

# 输入数据路径.
lcTileRegionDir = path.join(dataDir, '{0}_1_Tile{1}_HDF'.format(modisName, region))

# 输出数据路径.
lcMosaicRegionDir = path.join(dataDir, '{0}_2_Mosaic{1}_TIF'.format(modisName, region))
if not path.exists(lcMosaicRegionDir):
    os.mkdir(lcMosaicRegionDir)

# 坐标系统文件路径.
prjPath = path.join(dataDir, 'MODIS_Sinusoidal.prj')

# 按年份提取HDF格式文件中的MODIS LC数据并保存为TIF格式, 然后拼接MODIS LC.
for yearNum in yearList:
    # 判断镶嵌后的MODIS LC数据是否存在.
    lcMosaicPath = path.join(lcMosaicRegionDir, '{0}.A{1}001.006.tif'.format(modisName, yearNum))
    if arcpy.Exists(lcMosaicPath):
        continue

    # 创建临时文件夹.
    tempDir = path.join(rootDir, 'temp_{0}_{1}'.format(modisName, yearNum))
    if not path.exists(tempDir):
        os.mkdir(tempDir)

    # 从HDF文件中提取LC数据, 并保存为TIF格式.
    print(u'提取{0}年{1}的{2}数据.'.format(yearNum, region, modisName))
    lcYearDir = path.join(lcTileRegionDir, '{0}_{1}XXX_HDF'.format(modisName, yearNum))
    for lcHdfPath in glob(path.join(lcYearDir, '*.hdf')):
        lcTifPath = path.join(tempDir, path.basename(lcHdfPath).replace('hdf', 'tif'))
        if not arcpy.Exists(lcTifPath):
            arcpy.ExtractSubDataset_management(lcHdfPath, lcTifPath, '0')

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
    arcpy.CopyRaster_management(mosaicPath, lcMosaicPath)

    # 删除临时文件夹.
    arcpy.Delete_management(tempDir)

# 投影, 重分类, 重采样.
lcRemap = arcpy.sa.RemapValue([[1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2],
                               [9, 2], [10, 2], [11, 3], [12, 2], [13, 5], [14, 2], [15, 4],
                               [16, 1], [17, 3]])
exampleRasPath = path.join(dataDir, 'Projection_Example', exampleRasName)
arcpy.env.snapRaster = exampleRasPath
arcpy.env.workspace = lcMosaicRegionDir
for lcMosaic in arcpy.ListRasters('MCD12Q1*006.tif'):
    print(u'重分类, 投影 {0}'.format(lcMosaic))

    lcReclassify = lcMosaic.replace('.tif', '_reclsfy.tif')
    if not arcpy.Exists(lcReclassify):
        arcpy.sa.Reclassify(lcMosaic, 'Value', lcRemap, 'NODATA').save(lcReclassify)

    lcPrjTrans = lcReclassify.replace('.tif', '_pt.tif')
    if not arcpy.Exists(lcPrjTrans):
        arcpy.ProjectRaster_management(lcReclassify, lcPrjTrans, exampleRasPath, 'NEAREST', '1000')

