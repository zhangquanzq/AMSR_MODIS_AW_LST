# coding=utf-8
# 拼接中国地区的Landsat Global30地表覆盖数据.
import arcpy
import os
from os import path
from glob import glob

# LC编码: 耕地 10, 林地 20, 草地 30, 灌木 40, 湿地 50, 水体 60, 苔原 70, 人造地表 80, 裸地 90, 冰川积雪 100.

# 标记.
# 指定重采样的范围和分辨率的标记. 1表示AMSR2的0.1度, 2表示MODIS的0.01度.
flg1 = 2

# 预设参数.
arcpy.env.pyramid = 'NONE'

cellsize = [0.1, 0.01][flg1 - 1]
yearList = [2000, 2010, 2020]
prjRef = arcpy.SpatialReference(4326)

remap = arcpy.sa.RemapRange([[0, 1, 'NODATA'], [10, 19, 2], [20, 29, 2], [30, 39, 2], [40, 49, 2],
                             [50, 59, 3], [60, 69, 3], [70, 79, 2], [80, 89, 5], [90, 99, 1],
                             [100, 101, 4]])

# 路径.
rootDir = r'J:\AMSR2_MODIS_AW_LST'
dataDir = path.join(rootDir, r'AMSR2_LST_Retrieval\Data')

# 输入数据路径.
landsatLcTileDir = path.join(dataDir, 'LandsatLC_1_TileCN_TIF')

# 参考数据路径.
amsr2CnRefPath = path.join(dataDir, r'AMSR2_2_CN_TIF\L3.TB6GHz_10\2012\07',
                           'GW1AM2_20120700_01M_EQMA_L3SGT06HA2220220_BtH.tif')
modisCnRefPath = path.join(dataDir, r'Zones\ExtentCN_0d01.tif')
extentRefPath = [amsr2CnRefPath, modisCnRefPath][flg1 - 1]

glacierCnPath = path.join(dataDir, r'Landcover\Glacier_CN_gcs_0.01.tif')

# 输出数据路径.
landsatLcMosaicDir = path.join(dataDir, 'LandsatLC_2_MosaicCN_TIF')
if not path.exists(landsatLcMosaicDir):
    os.mkdir(landsatLcMosaicDir)

# Landsat LC数据投影转换.
for yearNum in yearList:
    yearCell = '{0}_{1}'.format(yearNum, cellsize)
    lcPrjYearDir = path.join(landsatLcMosaicDir, 'LandsatLC_Prj_{0}'.format(yearNum))
    if not path.exists(lcPrjYearDir):
        os.mkdir(lcPrjYearDir)

    lcReclsYearDir = path.join(landsatLcMosaicDir, 'LandsatLC_Reclass_{0}'.format(yearNum))
    if not path.exists(lcReclsYearDir):
        os.mkdir(lcReclsYearDir)

    lcResmplYearFolder = 'LandsatLC_Resmpl_{0}'.format(yearCell)
    lcResmplYearDir = path.join(landsatLcMosaicDir, lcResmplYearFolder)
    if not path.exists(lcResmplYearDir):
        os.mkdir(lcResmplYearDir)

    lcYearDir = path.join(landsatLcTileDir, 'LandsatLC_{0}'.format(yearNum))
    for lcPath in glob(path.join(lcYearDir, '*.tif')):
        lcName = path.basename(lcPath)
        lcPrjPath = path.join(lcPrjYearDir, lcName)
        if not arcpy.Exists(lcPrjPath):
            print(u'投影转换: {0}'.format(lcName))
            arcpy.env.snapRaster = path.join(dataDir, 'Reference Raster', 'N38E111.hgt')
            arcpy.ProjectRaster_management(lcPath, lcPrjPath, prjRef, 'NEAREST', '0.00027777778')
            arcpy.ClearEnvironment('snapRaster')

        lcReclassPath = path.join(lcReclsYearDir, lcName)
        if not arcpy.Exists(lcReclassPath):
            print(u'重分类: {0}'.format(lcName))
            arcpy.sa.Reclassify(lcPrjPath, 'Value', remap, 'NODATA').save(lcReclassPath)

        lcResmplPath = path.join(lcResmplYearDir, lcName)
        if not arcpy.Exists(lcResmplPath):
            print(u'重采样: {0}'.format(lcName))
            arcpy.env.snapRaster = extentRefPath
            arcpy.Resample_management(lcReclassPath, lcResmplPath, cellsize, 'MAJORITY')
            arcpy.ClearEnvironment('snapRaster')

    # 拼接每年重采样和重分类的Landsat LC.
    lcResmplMosaicName = 'LandsatLC_CN_{0}.tif'.format(yearCell)
    lcResmplMosaicPath = path.join(landsatLcMosaicDir, lcResmplMosaicName)
    if not arcpy.Exists(lcResmplMosaicPath):
        print(u'输出拼接后的CN区Landsat LC: {0}'.format(lcResmplMosaicName))
        gdbName = 'LandsatLC_{0}.gdb'.format(yearNum)
        mosaicName = 'LandsatLC_Mosaic'
        gdbPath = path.join(landsatLcMosaicDir, gdbName)
        if arcpy.Exists(gdbPath):
            arcpy.Delete_management(gdbPath)

        mosaicPath = path.join(gdbPath, mosaicName)
        arcpy.CreateFileGDB_management(landsatLcMosaicDir, gdbName)
        arcpy.CreateMosaicDataset_management(gdbPath, mosaicName, prjRef)
        arcpy.AddRastersToMosaicDataset_management(mosaicPath, 'Raster Dataset', lcResmplYearDir)
        arcpy.env.extent = arcpy.Raster(extentRefPath).extent
        arcpy.CopyRaster_management(mosaicPath, lcResmplMosaicPath, nodata_value='127')
        arcpy.ClearEnvironment('extent')
        arcpy.Delete_management(gdbPath)

    # 替换LandsatLC中错分的冰川覆盖类型.
    lcReplacePath = path.join(landsatLcMosaicDir, 'LandsatLC_CN_Update_{0}.tif'.format(yearCell))
    if not arcpy.Exists(lcReplacePath):
        print(u'替换{0}年Landsat LC中的冰川覆盖类型'.format(yearNum))
        remap = arcpy.sa.RemapValue([[4, 'NODATA']])
        lcReclsRas = arcpy.sa.Reclassify(lcResmplMosaicPath, 'Value', remap)
        lcNibbleRas = arcpy.sa.Nibble(lcResmplMosaicPath, lcReclsRas)
        glacierCnRas = arcpy.Raster(glacierCnPath)
        arcpy.sa.Con(glacierCnRas == 1, glacierCnRas * 4, lcNibbleRas).save(lcReplacePath)

        arcpy.Delete_management(lcReclsRas)
        arcpy.Delete_management(lcNibbleRas)
