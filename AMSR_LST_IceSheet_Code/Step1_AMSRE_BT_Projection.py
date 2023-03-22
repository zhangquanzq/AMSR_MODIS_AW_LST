# coding=utf-8
# 将格陵兰或南极的AMSRE BT数据的坐标从GCS转为PCS.
# 注意: 投影转换后的分辨率, 坐标系统, 像元对齐方式还需要重新确定.
import arcpy
import numpy as np
import os
from os import path
from glob import glob

# ArcPy环境变量设置.
arcpy.env.overwriteOutput = True
arcpy.env.pyramid = 'NONE'

# 指定输出范围的标记. 1表示格陵兰, 2表示南极.
flg1 = 1

# 预设参数.
region = ['Greenland', 'Antarctic'][flg1 - 1]
exampleRasName = ['AMSRE_Greenland_Example.tif', 'AMSRE_Antarctic_Example.tif'][flg1 - 1]

# 路径.
rootDir = r'H:\AMSR_LST_IceSheet'
dataDir = path.join(rootDir, 'Data')
amsreRegionTifDir = path.join(dataDir, 'AMSRE_2_BT_{0}_TIF'.format(region))
amsreRegionPrjDir = path.join(dataDir, 'AMSRE_2_BT_Prj{0}_TIF'.format(region))
if not path.exists(amsreRegionPrjDir):
    os.mkdir(amsreRegionPrjDir)

# 输出AMSRE TIF文件的坐标系统, 格陵兰地区坐标的中央经线-45度.
exampleRasPath = path.join(dataDir, 'Projection_Example', exampleRasName)
arcpy.env.snapRaster = exampleRasPath
amsreRegionYearDirList = glob(path.join(amsreRegionTifDir, 'AMSRE*'))
for amsreRegionYearDir in amsreRegionYearDirList[4:]:
    amsreRegionYearPrjDir = path.join(amsreRegionPrjDir, path.basename(amsreRegionYearDir))
    if not path.exists(amsreRegionYearPrjDir):
        os.mkdir(amsreRegionYearPrjDir)

    amsreRegionDailyDirList = glob(path.join(amsreRegionYearDir, 'AMSRE*'))
    for amsreRegionDailyDir in amsreRegionDailyDirList:
        amsreDate = path.basename(amsreRegionDailyDir)
        print(u'投影转换{0} {1} AMSRE BT数据.'.format(region, amsreDate))

        amsreRegionDailyPrjDir = path.join(amsreRegionYearPrjDir, amsreDate)
        if not path.exists(amsreRegionDailyPrjDir):
            os.mkdir(amsreRegionDailyPrjDir)

        amsreTifPathList = glob(path.join(amsreRegionDailyDir, 'AMSRE*.tif'))
        for amsreTifPath in amsreTifPathList:
            amsreTifPrjPath = path.join(amsreRegionDailyPrjDir, path.basename(amsreTifPath))
            amsreTifPrj2Path = amsreTifPrjPath.replace('.tif', '_b.tif')
            if not arcpy.Exists(amsreTifPrjPath):
                arcpy.ProjectRaster_management(amsreTifPath, amsreTifPrj2Path, exampleRasPath,
                                               cell_size='25000')
                arcpy.env.extent = exampleRasPath
                arcpy.CopyRaster_management(amsreTifPrj2Path, amsreTifPrjPath)
                arcpy.Delete_management(amsreTifPrj2Path)
