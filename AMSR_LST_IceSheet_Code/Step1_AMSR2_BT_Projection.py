# coding=utf-8
# 将格陵兰或南极的AMSR2 BT数据的坐标从GCS转为PCS.
# 注意: 投影转换后的分辨率, 坐标系统, 像元对齐方式还需要重新确定.
import arcpy
import numpy as np
import os
from os import path
from glob import glob

# ArcPy环境变量设置.
arcpy.env.overwriteOutput = True
arcpy.env.pyramid = 'NONE'

# 标记和预设参数.
# 指定输出范围的标记. 1表示格陵兰, 2表示南极.
flg1 = 1
# 指定白天和晚上的标记. 1表示白天(升轨), 2表示晚上(降轨).
flg2 = 2

# AMSR2数据的范围, 轨道.
region = ['Greenland', 'Antarctic'][flg1 - 1]
exampleRasName = ['AMSRE_Greenland_Example.tif', 'AMSRE_Antarctic_Example.tif'][flg1 - 1]
orbit = ['A', 'D'][flg2 - 1]
dayNight = ['Day', 'Night'][flg2 - 1]

# 路径.
rootDir = r'F:\AMSR_LST_IceSheet'
dataDir = path.join(rootDir, 'Data')
amsr2RegionTifDir = path.join(dataDir, 'AMSR2_2_BT_{0}_TIF'.format(region))
amsr2RegionPrjDir = path.join(dataDir, 'AMSR2_2_BT_Prj{0}_TIF'.format(region))
if not path.exists(amsr2RegionPrjDir):
    os.mkdir(amsr2RegionPrjDir)

# 获取所有年份AMSR2 TIF文件的路径列表.
amsr2PathList = []
for amsr2ChannelPath in glob(path.join(amsr2RegionTifDir, '*')):
    for amsr2YearPath in glob(path.join(amsr2ChannelPath, '*')):
        for amsr2MonthPath in glob(path.join(amsr2YearPath, '*')):
            amsr2DailyPathList = glob(path.join(amsr2MonthPath, '*01D_EQM{0}*.tif'.format(orbit)))
            amsr2PathList = amsr2PathList + amsr2DailyPathList

# 所有AMSR2 TIF文件的日期和年份列表.
amsr2DateList = [path.basename(amsr2Path).split('_')[1] for amsr2Path in amsr2PathList]
amsr2DateList2 = np.unique(amsr2DateList)  # 唯一化日期列表.
amsr2YearList = np.unique([amsr2Date[0:4] for amsr2Date in amsr2DateList2])

# 输出AMSR2 TIF文件的坐标系统, 格陵兰地区坐标的中央经线-45度.
# 分年度, 日期组织投影转换后的AMSR2 TIF文件.
exampleRasPath = path.join(dataDir, 'Projection_Example', exampleRasName)
arcpy.env.snapRaster = exampleRasPath
for yearStr in amsr2YearList:
    amsr2RegionYearDir = path.join(amsr2RegionPrjDir, 'AMSR2_{0}XXXX_TIF'.format(yearStr))
    if not path.exists(amsr2RegionYearDir):
        os.mkdir(amsr2RegionYearDir)

    # 获取当年所有的AMSR2文件.
    amsr2YearPathList, amsr2YearDateList = [], []
    for i in range(len(amsr2DateList)):
        if yearStr == amsr2DateList[i][0:4]:
            amsr2YearDateList.append(amsr2DateList[i])
            amsr2YearPathList.append(amsr2PathList[i])

    # 创建保存投影转换后AMSR2文件的日期文件夹.
    for amsr2Date in np.unique(amsr2YearDateList):
        amsr2RegionDateDir = path.join(amsr2RegionYearDir, 'AMSR2_{0}'.format(amsr2Date))
        if not path.exists(amsr2RegionDateDir):
            os.mkdir(amsr2RegionDateDir)

        # 获取当前日期的所有通道AMSR2数据路径.
        amsr2DatePathList = []
        for i in range(len(amsr2YearPathList)):
            if amsr2Date in amsr2YearPathList[i]:
                amsr2DatePathList.append(amsr2YearPathList[i])

        # 当前日期所有通道AMSR2 BT数据的投影转换.
        print(u'投影转换{0} {1} {2}time AMSR2 BT数据.'.format(region, amsr2Date, dayNight))
        for amsr2DatePath in amsr2DatePathList:
            amsr2PrjPath = path.join(amsr2RegionDateDir, path.basename(amsr2DatePath))
            if not path.exists(amsr2PrjPath):
                arcpy.ProjectRaster_management(amsr2DatePath, amsr2PrjPath, exampleRasPath,
                                               cell_size='10000')
