# coding=utf-8
import arcpy
import copy
import os
import pandas as pd
from functions import *
from glob import glob
from os import path

# 标记与预设参数.
# 指定研究区的标记. 1表示内蒙古, 2表示晋南豫西, 3表示云贵高原, 4表示北大河, 5表示那曲.
flg1 = 2
# 指定昼夜的标记. 1表示白天, 2表示夜晚.
flg2 = 2
# 指定降尺度策略的标记. 1表示逐级降尺度, 2表示直接降尺度.
flg3 = 1
# 指定降尺度数据源的标记. 1表示重采样的MODIS LST, 2表示真实的AMSR2 LST.
flg4 = 1
# 指定降尺度模型是否包含残差项的标记. 1表示包含, 0表示不包含.
flg5 = 1

yearNum = 2013

region = ['NMG', 'JNYX', 'YGGY', 'BDH', 'Naqu'][flg1 - 1]
dayNight = ['Day', 'Night'][flg2 - 1]
cellsizeList = [['0.1', '0.03', '0.01'], ['0.1', '0.01']][flg3 - 1]
strategy = ['Stepwise', 'Straight'][flg3 - 1]
sourceLst = ['MODIS', 'AMSR2'][flg4 - 1]
factorList = ['ELEV', 'NDVI', 'SLP']

factorStr = ';'.join(factorList)
fieldList = copy.copy(factorList)
fieldList.append('LST')
strategy = strategy + '_noResidual' if flg5 == 0 else strategy

rydStr = '{0}_{1}_{2}'.format(region, yearNum, dayNight)

arcpy.env.overwriteOutput = True

# %% 路径.
# 根目录.
rootDir = r'J:\AMSR2_MODIS_AW_LST'
dataDir = path.join(rootDir, 'AMSR2_LST_Downscaling', 'Data')

# 区域路径和SRTM数据路径.
regionDir = path.join(dataDir, 'Region_{0}'.format(region))
srtmRegionDir = path.join(regionDir, 'SRTM_{0}'.format(region))

# 从统计文件中读取符合条件的AMSR2和MODIS地表温度配对指标.
dateSelectCsvName = 'DateSelect_{0}.csv'.format(rydStr)
dateSelectCsvPath = path.join(regionDir, dateSelectCsvName)
with open(dateSelectCsvPath, 'r') as f1:
    staFieldList, recordList = f1.readline(), f1.readlines()
dateList = [record[:-1].split(',')[0] for record in recordList]
pctList = [float(record[:-1].split(',')[1]) for record in recordList]
ccList = [float(record[:-1].split(',')[2]) for record in recordList]

dateSelectFrame = pd.read_csv(dateSelectCsvPath)
dateSeries = dateSelectFrame['YearDate']

# 保存降尺度后LST精度评价统计文件.
downscalingCsvName = 'Downscaling_{0}_{1}.csv'.format(rydStr, sourceLst, strategy)
downscalingCsvPath = path.join(regionDir, downscalingCsvName)

# %% 遍历有效数据日期, 并进行降尺度.
errorRecordList = []
regionYearDir = path.join(regionDir, rydStr)
regionDateDirList = glob(path.join(regionYearDir, '{0}*'.format(region)))
for regionDateDir in regionDateDirList[1:]:
    dateStr1 = path.basename(regionDateDir).split('_')[1]
    dateStr2 = ymd2yday(dateStr1)

    # 创建存放中间栅格数据的文件夹.
    regionRasDir = path.join(regionDateDir, '{0}_Raster'.format(strategy))
    if not path.exists(regionRasDir):
        os.mkdir(regionRasDir)

    # 创建存放GWR结果的文件夹.
    regionVarDir = path.join(regionDateDir, '{0}_Variables'.format(strategy))
    if not path.exists(regionVarDir):
        os.mkdir(regionVarDir)

    # 确定降尺度LST数据类型.
    amsr2LstName = 'AMSR2_LST_{0}_{1}_{2}.tif'.format(dayNight, region, dateStr1)
    amsr2LstPath = path.join(regionDateDir, amsr2LstName)
    modisLstName = 'MYD11A1.A{0}.LST_{1}_{2}_0.1.tif'.format(dateStr2, dayNight, region)
    modisLstPath = path.join(regionDateDir, modisLstName)
    originalLstPath = [modisLstPath, amsr2LstPath][flg4 - 1]

    # 逐级降尺度.
    halfName = '{0}_LST_{1}_{2}_{3}'.format(sourceLst, dayNight, region, dateStr1)
    for i in range(len(cellsizeList) - 1):
        cellsizeNow = cellsizeList[i]
        cellsizeNow2 = cellsizeNow.replace('.', 'p')
        cellsizeNext = cellsizeList[i + 1]
        print(u'将{0} LST的分辨率从{1}降尺度到{2}.'.format(sourceLst, cellsizeNow, cellsizeNext))

        # 判断当前分辨率级别的降尺度结果是否存在.
        predictedLstNextPath = path.join(regionRasDir, '{0}_{1}.tif'.format(halfName, cellsizeNext))
        if path.exists(predictedLstNextPath):
            continue

        # 创建存放GWR变量的文件夹.
        cellsizeFolder = 'r{0}_{1}LstDependentVar'.format(cellsizeNext, sourceLst)
        cellsizeDir = path.join(regionVarDir, cellsizeFolder)
        if not path.exists(cellsizeDir):
            os.mkdir(cellsizeDir)

        # 当前和下一分辨率级别的环境变量路径.
        ndviNowPath = glob(path.join(regionDateDir, 'MYD13A2*{0}.tif'.format(cellsizeNow)))[0]
        elevNowPath = glob(path.join(srtmRegionDir, 'SRTM_{0}*Elev.tif'.format(cellsizeNow)))[0]
        slpNowPath = glob(path.join(srtmRegionDir, 'SRTM_{0}*Slp.tif'.format(cellsizeNow)))[0]
        ndviNextPath = ndviNowPath.replace(cellsizeNow, cellsizeNext)
        elevNextPath = elevNowPath.replace(cellsizeNow, cellsizeNext)
        slpNextPath = slpNowPath.replace(cellsizeNow, cellsizeNext)

        # 将原始LST影像转为点文件, 并提取各点位置的环境变量值.
        varPointPath = path.join(cellsizeDir, 'Variable_{0}.shp'.format(cellsizeNow2))
        varPointExist = path.exists(varPointPath)
        intersectField = set()
        if varPointExist:
            varPointFieldList = [field.name for field in arcpy.ListFields(varPointPath)]
            intersectField = set(fieldList) & set(varPointFieldList)
        if not varPointExist or (varPointExist and intersectField != set(fieldList)):
            inRasterList = [[ndviNowPath, 'NDVI'], [elevNowPath, 'ELEV'], [slpNowPath, 'SLP']]
            if cellsizeNow != cellsizeList[0]:
                predictedLstNowName = '{0}_{1}.tif'.format(halfName, cellsizeNow)
                originalLstPath = path.join(regionRasDir, predictedLstNowName)
            arcpy.RasterToPoint_conversion(originalLstPath, varPointPath)
            arcpy.AddField_management(varPointPath, 'LST', 'DOUBLE')
            arcpy.CalculateField_management(varPointPath, 'LST', "!grid_code!", "PYTHON_9.3")
            arcpy.DeleteField_management(varPointPath, ["grid_code"])
            arcpy.sa.ExtractMultiValuesToPoints(varPointPath, inRasterList)

        # 执行GWR工具.
        arcpy.env.snapRaster = ndviNextPath
        gwrPointPath = path.join(cellsizeDir, 'Variable_{0}_gwr.shp'.format(cellsizeNow2))
        arcpy.GeographicallyWeightedRegression_stats(varPointPath, 'LST', factorStr, gwrPointPath,
                                                     'ADAPTIVE', 'AICc', '#', '#', '#',
                                                     cellsizeDir, cellsizeNext)
        # 利用GWR模型, 从解释变量和回归系数估算LST.
        factorRasterList = [arcpy.Raster(elevNextPath), arcpy.Raster(ndviNextPath),
                            arcpy.Raster(slpNextPath)]
        coefRasterList = [arcpy.Raster(path.join(cellsizeDir, factor)) for factor in factorList]
        interceptRaster = arcpy.Raster(path.join(cellsizeDir, 'intercept'))
        # 残差插值.
        if flg5 == 1:
            gwrResidualPath = path.join(cellsizeDir, 'residual')
            arcpy.EmpiricalBayesianKriging_ga(gwrPointPath, 'Residual', '#', gwrResidualPath,
                                              cellsizeNext)
            predictedLstRaster = interceptRaster + arcpy.Raster(gwrResidualPath)
        else:
            predictedLstRaster = interceptRaster
        # 估算LST并保存.
        for j in range(len(coefRasterList)):
            predictedLstRaster = predictedLstRaster + coefRasterList[j] * factorRasterList[j]
        predictedLstRaster.save(predictedLstNextPath)

    # 精度评价.
    predictedLstPath = path.join(regionRasDir, '{0}_{1}.tif'.format(halfName, cellsizeList[-1]))
    lstDiffRaster = (arcpy.Raster(predictedLstPath) - arcpy.Raster(modisLstPath)) * 0.02
    bias = lstDiffRaster.mean
    rmse = arcpy.sa.SquareRoot(arcpy.sa.Power(lstDiffRaster, 2).mean)
    print('Bias:{0:.3f}, RMSE:{1:.3f}\n'.format(bias, rmse))
    errorRecordList.append('{0},{1:.3f},{2:.3f}\n'.format(dateStr1, bias, rmse))

# 保存精度评价统计文件.
with open(downscalingCsvPath, 'w') as f2:
    f2.writelines('YearDate, Bias, RMSE\n')
    f2.writelines(errorRecordList)
