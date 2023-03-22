# %% coding=utf-8
# 将分年份, 日期组织的MODIS地表温度和反照率HDF文件拼接成一幅TIF文件, 然后投影转换.
# 注意: 此程序拼接得到的TIF文件中, 原HDF为nodata的数值变成数字, 需要下一步将其转为Nodata.
import arcpy
import os
from glob import glob
from os import path

"""
MxD11A1数据图层: ['LST_Day','QC_Day','Time_Day','Angle_Day','LST_Night','QC_Night','Time_Night',
                'Angle_Night','Emis_31','Emis_32','Clear_Day','Clear_Night']
MxD10A1数据图层: ['Snow_cover','Basic_QA','Algorithm_QA','NDSI','Albedo','Orbit_pnt','Granule_pnt']
"""

# ArcPy环境参数设置.
arcpy.env.overwriteOutput = True
arcpy.env.pyramid = 'NONE'

# 指定区域的标记. 1表示Greenland, 2表示Antarctic.
flg1 = 2

modisStdRowN = [3150, 6024][flg1 - 1]
modisStdColN = [3850, 6599][flg1 - 1]

# 预设参数.
region = ['Greenland', 'Antarctic'][flg1 - 1]
exampleRasName = ['AMSRE_Greenland_Example.tif', 'AMSRE_Antarctic_Example.tif'][flg1 - 1]
yearList = range(2002, 2020)

# MYD11A1数据的属性信息.
modisName = 'MYD11A1'
layerList = ['LST_Day', 'QC_Day', 'LST_Night', 'QC_Night', 'Emis_31', 'Emis_32']
layerNumList = ['0', '1', '4', '5', '8', '9']
layerListN = len(layerList)

# 路径.
# 根目录.
rootDir = r'F:\AMSR_LST_IceSheet'
dataDir = path.join(rootDir, 'Data')

# 输入数据路径.
modisTilesDir = path.join(dataDir, '{0}_1_Tile{1}_HDF'.format(modisName, region))

# 输出数据路径.
modisMosaicDir = path.join(dataDir, '{0}_2_Mosaic{1}_TIF'.format(modisName, region))
if not path.exists(modisMosaicDir):
    os.makedirs(modisMosaicDir)

modisPrjDir = path.join(dataDir, '{0}_2_Prj{1}_TIF'.format(modisName, region))
if not path.exists(modisPrjDir):
    os.makedirs(modisPrjDir)

# 拼接MODIS数据图层.
mosaicPrjPath = path.join(dataDir, 'Modis_Sinusoidal.prj')
for year in yearList:
    break
    modisTilesYearDir = path.join(modisTilesDir, '{0}_{1}XXX_HDF'.format(modisName, year))
    modisMosaicYearDir = path.join(modisMosaicDir, '{0}_{1}XXX_TIF'.format(modisName, year))
    if not path.exists(modisMosaicYearDir):
        os.makedirs(modisMosaicYearDir)

    # 获取已完成所有数据层拼接的日期列表.
    arcpy.env.workspace = modisMosaicYearDir
    mosaicTifList = arcpy.ListRasters('{0}*.tif'.format(modisName))
    dateList = [mosaicTif.split('.')[1][1:] for mosaicTif in mosaicTifList]
    dateCounter = Counter(dateList)
    dateNumList = dateCounter.values()
    dateStrList = dateCounter.keys()
    dateNumListN = len(dateNumList)
    mosaicDateList = [dateStrList[i] for i in range(dateNumListN) if dateNumList[i] == layerListN]

    # 提取MODIS数据层并拼接它们, 跳过已完成所有图层拼接的日期.
    for modisTileDateDir in glob(path.join(modisTilesYearDir, '*'))[::-1]:
        dateStr = path.basename(modisTileDateDir).split('_')[-1]
        if dateStr in mosaicDateList:
            continue  # 跳过已完成所有数据层拼接的日期.

        # 从MODIS HDF文件中提取数据层，并保存到临时文件夹中.
        print(u'提取{0} {1} {2}'.format(region, dateStr, modisName))
        tmpDir = path.join(rootDir, 'tmp_{0}_{1}'.format(modisName, dateStr))
        if not path.exists(tmpDir):
            os.mkdir(tmpDir)
        for modisHdfPath in glob(path.join(modisTileDateDir, '*.hdf')):
            for i in range(layerListN):
                mosaicTifName = '{0}.A{1}.{2}.tif'.format(modisName, dateStr, layerList[i])
                mosaicTifPath = path.join(modisMosaicYearDir, mosaicTifName)
                if not arcpy.Exists(mosaicTifPath):
                    modisTileName = path.basename(modisHdfPath)
                    modisLayerName = modisTileName.replace('.hdf', '_{0}.tif'.format(layerList[i]))
                    modisLayerPath = path.join(tmpDir, modisLayerName)
                    if not arcpy.Exists(modisLayerPath):
                        try:
                            arcpy.ExtractSubDataset_management(modisHdfPath, modisLayerPath,
                                                               layerNumList[i])
                        except:
                            errorHDFTxtName = 'errorHDF{0}.txt'.format(dateStr)
                            errorHDFTxtPath = path.join(rootDir, modisName, errorHDFTxtName)
                            with open(errorHDFTxtPath, 'a') as f:
                                f.write(modisHdfPath + '\n')

        # 拼接提取的数据层.
        print(u'拼接{0} {1} {2}'.format(region, dateStr, modisName))
        for layerName in layerList:
            arcpy.env.workspace = tmpDir
            modisTileList = arcpy.ListRasters("{0}*_{1}.tif".format(modisName, layerName))
            mosaicTifName = '{0}.A{1}.{2}.tif'.format(modisName, dateStr, layerName)
            mosaicTifPath = path.join(modisMosaicYearDir, mosaicTifName)
            # modisTileStr = ';'.join(modisTileList)
            if not arcpy.Exists(mosaicTifPath):
                gdbName = '{0}_{1}.gdb'.format(modisName, dateStr)
                gdbPath = path.join(rootDir, gdbName)
                mosaicPath = path.join(gdbPath, layerName)
                arcpy.CreateFileGDB_management(rootDir, gdbName)
                arcpy.CreateMosaicDataset_management(gdbPath, layerName, mosaicPrjPath)
                arcpy.AddRastersToMosaicDataset_management(mosaicPath, 'Raster Dataset', tmpDir,
                                                           filter='*{0}.tif'.format(layerName))
                # 以下pixel_type和nodata_value仅是Emis的参数设置, nodata_value='255'
                arcpy.CopyRaster_management(mosaicPath, mosaicTifPath, pixel_type='16_BIT_UNSIGNED')
                if 'LST' in path.basename(mosaicTifPath):
                    arcpy.SetRasterProperties_management(mosaicTifPath, nodata='1 0')
                arcpy.Delete_management(gdbPath)

        # 删除临时文件夹.
        arcpy.Delete_management(tmpDir)

# MODIS数据投影转换.
exampleRasPath = path.join(dataDir, 'Projection_Example', exampleRasName)
arcpy.env.snapRaster = exampleRasPath
# 输出MODIS TIF文件的坐标系统, 格陵兰地区坐标的中央经线-45度.
# for year in yearList:
for year in [2020]:
    modisMosaicYearDir = path.join(modisMosaicDir, '{0}_{1}XXX_TIF'.format(modisName, year))
    modisPrjYearPath = path.join(modisPrjDir, '{0}_{1}XXX_TIF'.format(modisName, year))
    if not path.exists(modisPrjYearPath):
        os.makedirs(modisPrjYearPath)
    modisMosaicTifPathList = glob(path.join(modisMosaicYearDir, '*.tif'))
    for mosaicTifPath in modisMosaicTifPathList:
        mosaicTifName = path.basename(mosaicTifPath)
        modisPrjTifPath = path.join(modisPrjYearPath, mosaicTifName)
        modisTmpTifPath = modisPrjTifPath.replace('.tif', '_b.tif')
        if not path.exists(modisPrjTifPath):
            print(u'投影转换: {0}'.format(mosaicTifName))
            arcpy.ProjectRaster_management(mosaicTifPath, modisTmpTifPath, exampleRasPath,
                                           cell_size='1000')
            arcpy.Clip_management(modisTmpTifPath, '#', modisPrjTifPath, exampleRasPath)
            arcpy.Delete_management(modisTmpTifPath)

        # 删除影像行列数不对的数据.
        rowN = arcpy.GetRasterProperties_management(modisPrjTifPath, 'ROWCOUNT').getOutput(0)
        colN = arcpy.GetRasterProperties_management(modisPrjTifPath, 'COLUMNCOUNT').getOutput(0)
        if int(rowN) != modisStdRowN and int(colN) != modisStdColN:
            arcpy.Delete_management(modisPrjTifPath)
