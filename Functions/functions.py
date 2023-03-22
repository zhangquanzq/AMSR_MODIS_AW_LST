# coding=utf-8
import os
import sys
import re
from glob import glob
import arcpy
import numpy as np
import calendar
from os import path
from numpy.polynomial import polynomial as poly
from scipy.ndimage import gaussian_filter
from sklearn import ensemble


def ymd2yday(ymd):
    # Convert YearMonthDay to YearDayOfYear. Both input and output are strings.
    # Format of input: 'yyyymmdd'. For example: '20100503'.
    # Format of output: 'yyyyday'. For example: '2010123'
    year = ymd[0:4]
    month = int(ymd[4:6])
    day = int(ymd[6:8])

    month_day = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if calendar.isleap(int(year)):
        month_day[1] = 29
    monthSum = sum(month_day[0:month - 1])
    dday = monthSum + day
    yday = '{0}{1:03d}'.format(year, dday)

    return yday


def yday2ymd(year_day):
    year = int(year_day[0:4])
    day = int(year_day[4:7])
    month_day = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if calendar.isleap(year):
        month_day[1] = 29

    i = 0
    while day > 0:
        day = day - month_day[i]
        i = i + 1

    return '{0}{1:>02}{2:>02}'.format(year, i, day + month_day[i - 1])


def resamplePixelSeries(rasterBufferPath, regionExtentPath, resampleDir, cellsizeList, region,
                        downscale='initial'):
    # 限定downscale参数的取值范围.
    if downscale not in ['initial', 'previous']:
        raise ValueError(u"downscale参数的取值必须为字符串 'initial' 或 'previous' 之一.")

    # 按分辨率序列重采样.
    rasterBufferName = path.basename(rasterBufferPath)
    rasterRegionName = rasterBufferName.replace('Buffer', region)
    rasterBufferDir = path.dirname(rasterBufferPath)
    rasterTempPathList = []
    for i in range(len(cellsizeList) - 1):
        cellsizeNow = cellsizeList[i]
        cellsizeNext = cellsizeList[i + 1]

        rasterBufferNext = rasterBufferName.replace('0.01', cellsizeNext)
        rasterBufferResamplePath = path.join(rasterBufferDir, rasterBufferNext)

        rasterNext = rasterRegionName.replace('0.01', cellsizeNext)
        rasterResamplePath = path.join(resampleDir, rasterNext)

        # 两种重采样方式. 1, 上一步重采样结果作为下一步输入数据, 2, 原始MODIS数据生成所有分辨率序列中的数据.
        if downscale == 'initial':
            rasterPath = rasterBufferPath
        elif downscale == 'previous':
            rasterPath = rasterBufferPath if cellsizeNow == cellsizeList[0] else \
                rasterBufferPath.replace('0.01', cellsizeNow)

        arcpy.Resample_management(rasterPath, rasterBufferResamplePath, cellsizeNext, 'BILINEAR')
        arcpy.Clip_management(rasterBufferResamplePath, '#', rasterResamplePath, regionExtentPath)
        rasterTempPathList.append(rasterBufferResamplePath)

    # 删除Buffer过程数据.
    for rasterTempPath in rasterTempPathList:
        arcpy.Delete_management(rasterTempPath)


def ScalingRegress(largePixelImagePath, smallPixelImagePath, fittedSmallPixelImagePath, neighborN):
    # Get the arrays storing the pixel values and pixel coordinates from the image with large pixel.
    largePixelRaster = arcpy.Raster(largePixelImagePath)
    nodata = largePixelRaster.noDataValue
    extentLargePixel = largePixelRaster.extent
    cellsizeXLarge = largePixelRaster.meanCellWidth
    cellsizeYLarge = largePixelRaster.meanCellHeight
    largePixelXVector = np.arange(extentLargePixel.XMin + cellsizeXLarge / 2.0,
                                  extentLargePixel.XMax, cellsizeXLarge)
    largePixelYVector = np.arange(extentLargePixel.YMax - cellsizeYLarge / 2.0,
                                  extentLargePixel.YMin, - cellsizeYLarge)
    largePixelXArray, largePixelYArray = np.meshgrid(largePixelXVector, largePixelYVector)

    largePixelArray = arcpy.RasterToNumPyArray(largePixelImagePath)
    largePixelNanIndex = (largePixelArray == nodata)
    largePixelArray = largePixelArray.astype(float)
    largePixelArray[largePixelNanIndex] = np.nan
    largePixelXArray[largePixelNanIndex] = np.nan
    largePixelYArray[largePixelNanIndex] = np.nan

    # Get the arrays storing the pixel values and pixel coordinates from the image with small pixel.
    smallPixelRaster = arcpy.Raster(smallPixelImagePath)
    nodata = smallPixelRaster.noDataValue
    extentSmallPixel = smallPixelRaster.extent
    cellsizeXSmall = smallPixelRaster.meanCellWidth
    cellsizeYSmall = smallPixelRaster.meanCellHeight
    smallPixelXVector = np.arange(extentSmallPixel.XMin + cellsizeXSmall / 2.0,
                                  extentSmallPixel.XMax, cellsizeXSmall)
    smallPixelYVector = np.arange(extentSmallPixel.YMax - cellsizeYSmall / 2.0,
                                  extentSmallPixel.YMin, - cellsizeYSmall)
    smallPixelXArray, smallPixelYArray = np.meshgrid(smallPixelXVector, smallPixelYVector)

    smallPixelArray = arcpy.RasterToNumPyArray(smallPixelImagePath)
    smallPixelNanIndex = (smallPixelArray == nodata)
    smallPixelArray = smallPixelArray.astype(float)
    smallPixelArray[smallPixelNanIndex] = np.nan
    smallPixelXArray[smallPixelNanIndex] = np.nan
    smallPixelYArray[smallPixelNanIndex] = np.nan

    # Find the top N nearest pixels around each pixel in the large-pixel image, and construct the
    #   regression model between these nearest pixels and their nearest pixels in the small-pixel
    #   image. N equals neighborN, in which each pixel itself is included.
    fittedSmallPixelArray = np.zeros_like(smallPixelArray) * np.nan
    for ii in range(len(largePixelYVector)):
        for jj in range(len(largePixelXVector)):
            # Skip the pixels of nan values.
            largePixelValue = largePixelArray[ii, jj]
            if np.isnan(largePixelValue):
                continue

            # Get the image values and coordinates of top N nearest pixels around each pixel in the
            #   large-pixel image.
            largePixelX = largePixelXArray[ii, jj]
            largePixelY = largePixelYArray[ii, jj]
            distanceXArray = largePixelX - largePixelXArray
            distanceYArray = largePixelY - largePixelYArray
            distanceArray = np.sqrt(distanceXArray ** 2 + distanceYArray ** 2)
            distanceMax = np.sort(distanceArray, axis=None)[neighborN - 1]
            neighborPixelIndex = (distanceArray <= distanceMax)
            neighborPixelXVector = largePixelXArray[neighborPixelIndex]
            neighborPixelYVector = largePixelYArray[neighborPixelIndex]
            neighborPixelVector = largePixelArray[neighborPixelIndex]

            # Get the nearest pixels of these top N nearest pixels from the small-pixel image.
            nearestPixelVector = np.zeros_like(neighborPixelVector)
            for j in range(len(neighborPixelVector)):
                distanceXArray = neighborPixelXVector[j] - smallPixelXArray
                distanceYArray = neighborPixelYVector[j] - smallPixelYArray
                distanceArray = np.sqrt(distanceXArray ** 2 + distanceYArray ** 2)
                nearestPixelIndex = (distanceArray == np.nanmin(distanceArray))
                nearestPixelVector[j] = smallPixelArray[nearestPixelIndex][0]

            # Regress the image values in two cellsize levels.
            p = np.polyfit(nearestPixelVector, neighborPixelVector, 1)
            distanceXIndexArray = np.abs(smallPixelXArray - largePixelX) <= cellsizeXLarge / 2.0
            distanceYIndexArray = np.abs(smallPixelYArray - largePixelY) <= cellsizeYLarge / 2.0
            distanceIndexArray = (distanceXIndexArray & distanceYIndexArray)
            fittedSmallPixelArray[distanceIndexArray] = \
                p[0] * smallPixelArray[distanceIndexArray] + p[1]

    # Export the adjusted image.
    lowerLeft = arcpy.Point(extentSmallPixel.XMin, extentSmallPixel.YMin)
    reference = smallPixelRaster.spatialReference
    fittedSmallPixelRaster = arcpy.NumPyArrayToRaster(fittedSmallPixelArray, lowerLeft,
                                                      cellsizeXSmall, cellsizeYSmall, nodata)
    arcpy.DefineProjection_management(fittedSmallPixelRaster, reference)
    fittedSmallPixelRaster.save(fittedSmallPixelImagePath)


# TsHARP降尺度模型.
def tsharp(lstNowPath, ndviNowPath, ndviNextPath, predictLstNextPath, cellsizeNext):
    # 当前分辨率下, 用于建立回归关系的LST和NDVI(FVC)数据层.
    lstNowLayer = arcpy.RasterToNumPyArray(lstNowPath, nodata_to_value=np.nan)

    ndviNowLayer = arcpy.RasterToNumPyArray(ndviNowPath) / 10000.0
    fvcNowLayer = 1 - np.power(1 - ndviNowLayer, 0.625)

    # 下一分辨率级别的NDVI(FVC)数据层.
    ndviNextLayer = arcpy.RasterToNumPyArray(ndviNextPath) / 10000.0
    fvcNextLayer = 1 - np.power(1 - ndviNextLayer, 0.625)

    # 排除无效值像元, 非植被区(水体, 等), 以及定义为NDVI <= 0的像元.
    validIndexLayer = np.logical_not(np.isnan(ndviNowLayer) | np.isnan(lstNowLayer) |
                                     np.less_equal(ndviNowLayer, 0))
    fvcNowVector, lstNowVector = fvcNowLayer[validIndexLayer], lstNowLayer[validIndexLayer]

    # 建立回归关系, 获取回归参数.
    p1 = poly.polyfit(fvcNowVector, lstNowVector, 1)

    # 计算LST残差.
    residualLayer = np.zeros_like(lstNowLayer) * np.nan
    residualLayer[validIndexLayer] = lstNowVector - poly.polyval(fvcNowVector, p1)

    # 将残差重采样到下一分辨率级别.
    ndviNowRas = arcpy.sa.Raster(ndviNowPath)
    residualName = path.basename(lstNowPath).replace('.tif', '_residual.tif')
    residualPath = path.join(path.dirname(predictLstNextPath), residualName)
    residualRas = arcpy.NumPyArrayToRaster(residualLayer, ndviNowRas.extent.lowerLeft,
                                           ndviNowRas.meanCellWidth, ndviNowRas.meanCellHeight)
    arcpy.env.snapRaster = ndviNextPath
    arcpy.Resample_management(residualRas, residualPath, cellsizeNext, 'NEAREST')
    arcpy.DefineProjection_management(residualPath, ndviNowRas.spatialReference)

    # 计算下一分辨率级别的LST.
    lstPredictLayer = poly.polyval(fvcNextLayer, p1)
    ndviNextRas = arcpy.sa.Raster(ndviNextPath)
    lstPredictRas = arcpy.NumPyArrayToRaster(lstPredictLayer, ndviNextRas.extent.lowerLeft,
                                             ndviNextRas.meanCellWidth, ndviNextRas.meanCellHeight)
    lstPredictRas = lstPredictRas + arcpy.sa.Raster(residualPath)
    lstPredictRas.save(predictLstNextPath)
    arcpy.DefineProjection_management(predictLstNextPath, ndviNextRas.spatialReference)


# RF降尺度模型.
def rfDownscaling(lstNowPath, factorNowPathList, factorNextPathList, predictLstNextPath,
                  cellsizeNext):
    # 判断两个分辨率级别的环境变量数量是否一致.
    factorNowN, factorNextN = len(factorNowPathList), len(factorNextPathList)
    if factorNowN != factorNextN:
        print('降尺度前后的环境变量个数不一致!')
        os.system("pause")
        sys.exit()

    # 当前分辨率下, 用于建立回归关系的LST数据层.
    lstNowLayer = arcpy.RasterToNumPyArray(lstNowPath, nodata_to_value=np.nan)

    # 当前分辨率下, 用于建立回归关系的环境变量数据层.
    # factorNowPathList = [ndviNowPath, elevNowPath, slpNowPath]
    rowNowN = arcpy.sa.Raster(factorNowPathList[0]).height
    colNowN = arcpy.sa.Raster(factorNowPathList[0]).width
    factorNowArray = np.zeros((rowNowN, colNowN, factorNowN)) * np.nan
    ndviNowIndex = 0
    for i in range(factorNowN):
        factorNowPath = factorNowPathList[i]
        if 'NDVI' in factorNowPath:
            factorLayer = arcpy.RasterToNumPyArray(factorNowPath) / 10000.0
            factorNowArray[:, :, i] = 1 - np.power(1 - factorLayer, 0.625)
            ndviNowIndex = i
        else:
            factorNowArray[:, :, i] = arcpy.RasterToNumPyArray(factorNowPath)

    # 下一分辨率级别的环境变量数据层.
    # factorNextPathList = [ndviNextPath, elevNextPath, slpNextPath]
    rowNextN = arcpy.sa.Raster(factorNextPathList[0]).height
    colNextN = arcpy.sa.Raster(factorNextPathList[0]).width
    factorNextArray = np.zeros((rowNextN, colNextN, factorNextN)) * np.nan
    ndviNextIndex = 0
    for i in range(factorNextN):
        factorNextPath = factorNextPathList[i]
        if 'NDVI' in factorNextPath:
            factorLayer = arcpy.RasterToNumPyArray(factorNextPath) / 10000.0
            factorNextArray[:, :, i] = 1 - np.power(1 - factorLayer, 0.625)
            ndviNextIndex = i
        else:
            factorNextArray[:, :, i] = arcpy.RasterToNumPyArray(factorNextPath)

    # 排除当前分辨率级别的无效值像元, 以及非植被区(水体, 等).
    validIndexNowLayer = np.logical_not(np.isnan(factorNowArray[:, :, ndviNowIndex]) |
                                        np.isnan(lstNowLayer))
    lstNowVector = lstNowLayer[validIndexNowLayer]
    factorNowMatrix = np.zeros((len(lstNowVector), factorNowN)) * np.nan
    for i in range(factorNowN):
        factorNowMatrix[:, i] = factorNowArray[:, :, i][validIndexNowLayer]

    # 排除下一分辨率级别的无效值像元.
    validIndexNextLayer = np.logical_not(np.isnan(factorNextArray[:, :, ndviNextIndex]))
    factorNextMatrix = np.zeros((np.sum(validIndexNextLayer), factorNowN)) * np.nan
    for i in range(factorNextN):
        factorNextMatrix[:, i] = factorNextArray[:, :, i][validIndexNextLayer]

    # 建立当前分辨率级别的RF回归关系.
    rfRegModel = ensemble.RandomForestRegressor(n_estimators=100, oob_score=True)
    rfRegModel.fit(factorNowMatrix, lstNowVector)

    # 计算LST残差.
    residualLayer = np.zeros_like(lstNowLayer) * np.nan
    residualLayer[validIndexNowLayer] = lstNowVector - rfRegModel.predict(factorNowMatrix)

    # 将残差重采样到下一分辨率级别.
    ndviNowRas = arcpy.sa.Raster(factorNowPathList[ndviNowIndex])
    residualName = path.basename(lstNowPath).replace('.tif', '_residual.tif')
    residualPath = path.join(path.dirname(predictLstNextPath), residualName)
    residualRas = arcpy.NumPyArrayToRaster(residualLayer, ndviNowRas.extent.lowerLeft,
                                           ndviNowRas.meanCellWidth, ndviNowRas.meanCellHeight)
    arcpy.env.snapRaster = factorNextPathList[ndviNextIndex]
    arcpy.Resample_management(residualRas, residualPath, cellsizeNext, 'BILINEAR')
    arcpy.DefineProjection_management(residualPath, ndviNowRas.spatialReference)

    # 计算下一分辨率级别的LST.
    lstPredictLayer = np.zeros((rowNextN, colNextN)) * np.nan
    lstPredictLayer[validIndexNextLayer] = rfRegModel.predict(factorNextMatrix)
    ndviNextRas = arcpy.sa.Raster(factorNextPathList[ndviNextIndex])
    lstPredictRas = arcpy.NumPyArrayToRaster(lstPredictLayer, ndviNextRas.extent.lowerLeft,
                                             ndviNextRas.meanCellWidth,
                                             ndviNextRas.meanCellHeight)
    lstPredictRas = lstPredictRas + arcpy.sa.Raster(residualPath)
    lstPredictRas.save(predictLstNextPath)
    arcpy.DefineProjection_management(predictLstNextPath, ndviNextRas.spatialReference)


# GWR降尺度模型.
def gwrDownscaling(lstNowPath, factorNowPathList, factorNextPathList, lstNextPath, cellsizeNext):
    # 判断两个分辨率级别的环境变量数量是否一致.
    factorNowN, factorNextN = len(factorNowPathList), len(factorNextPathList)
    if factorNowN != factorNextN:
        print('降尺度前后的环境变量个数不一致!')
        os.system("pause")
        sys.exit()

    # ArcPy环境变量设置.
    arcpy.env.overwriteOutput = True

    # 环境因子名称列表, 当前和下一分辨率级别.
    factorList = ['NDVI', 'ELEV', 'SLP']
    cellsizeNow = '{0:.2f}'.format(arcpy.Raster(lstNowPath).meanCellWidth)
    cellsizeNow2, cellsizeNext2 = cellsizeNow.replace('.', 'p'), cellsizeNext.replace('.', 'p')

    # 创建存放GWR中间数据的文件夹.
    lstNextDir = path.dirname(lstNextPath)
    cellsizeDir = path.join(lstNextDir, 'R{0}ToR{1}'.format(cellsizeNow2, cellsizeNext2))
    if not path.exists(cellsizeDir):
        os.mkdir(cellsizeDir)

    # 将待降尺度的LST影像转为点文件, 并提取各点位置的环境变量值.
    varPointPath = path.join(cellsizeDir, 'Variable_{0}.shp'.format(cellsizeNow2))
    if arcpy.Exists(varPointPath):
        arcpy.Delete_management(varPointPath)
    else:
        inRasterList = [[factorNowPathList[0], 'NDVI'], [factorNowPathList[1], 'ELEV'],
                        [factorNowPathList[2], 'SLP']]
        arcpy.RasterToPoint_conversion(lstNowPath, varPointPath)
        arcpy.AddField_management(varPointPath, 'LST', 'DOUBLE')
        arcpy.CalculateField_management(varPointPath, 'LST', "!grid_code!", "PYTHON_9.3")
        arcpy.DeleteField_management(varPointPath, ["grid_code"])
        arcpy.sa.ExtractMultiValuesToPoints(varPointPath, inRasterList)

    # 执行GWR工具, 并计算残差.
    arcpy.env.snapRaster = factorNextPathList[0]
    gwrPointPath = path.join(cellsizeDir, 'Variable_{0}_gwr.shp'.format(cellsizeNow2))
    arcpy.GeographicallyWeightedRegression_stats(varPointPath, 'LST', ';'.join(factorList),
                                                 gwrPointPath, 'ADAPTIVE', 'AICc', '#', '#', '#',
                                                 cellsizeDir, cellsizeNext)
    arcpy.env.snapRaster = factorNextPathList[0]
    residualPath = path.join(cellsizeDir, 'residual')
    arcpy.EmpiricalBayesianKriging_ga(gwrPointPath, 'Residual', '#', residualPath, cellsizeNext)

    # 根据GWR回归系数和环境因子影像估算降尺度后LST.
    coefRasterList = [arcpy.Raster(path.join(cellsizeDir, factor)) for factor in factorList]
    factorRasterList = [arcpy.Raster(factorPath) for factorPath in factorNextPathList]
    interceptPath = path.join(cellsizeDir, 'intercept')
    predictLstRaster = arcpy.Raster(interceptPath) + arcpy.Raster(residualPath)
    for i in range(len(coefRasterList)):
        predictLstRaster = predictLstRaster + coefRasterList[i] * factorRasterList[i]
    predictLstRaster.save(lstNextPath)


# 使用TsHARP模型对LST进行逐级降尺度(直接降尺度时逐级降尺度的特例).
def tsharpStepwise(initialLstPath, ndviDir, regionMethodDir, cellsizeList):
    # 从初始LST文件名中获取数据属性信息.
    nameParts = re.split('[._]', path.basename(initialLstPath))
    sourceLst, daynight, region = nameParts[0], nameParts[3], nameParts[4]
    dateNum = nameParts[2] if 'AMSR' in sourceLst else nameParts[1][1:]
    predictLstStr = '_'.join([sourceLst, 'LST', daynight, region, str(dateNum)])

    for i in range(len(cellsizeList) - 1):
        # 当前和下一分辨率级别.
        cellsizeNow, cellsizeNext = cellsizeList[i], cellsizeList[i + 1]

        # 判断当前分辨率级别的降尺度结果是否存在.
        predictLstNextName = '{0}_{1}.tif'.format(predictLstStr, cellsizeNext)
        predictLstNextPath = path.join(regionMethodDir, predictLstNextName)
        if not path.exists(predictLstNextPath):
            # 读取当前分辨率级别需要降尺度的LST数据.
            predictLstNowName = '{0}_{1}.tif'.format(predictLstStr, cellsizeNow)
            predictLstNowPath = path.join(regionMethodDir, predictLstNowName)
            lstNowPath = initialLstPath if cellsizeNow == cellsizeList[0] else predictLstNowPath

            # 当前和下一分辨率级别的NDVI(FVC)数据路径.
            ndviNowName = '*NDVI*{0}.tif'.format(cellsizeNow)
            ndviNowPath = glob(path.join(ndviDir, ndviNowName))[0]
            ndviNextName = path.basename(ndviNowPath).replace(cellsizeNow, cellsizeNext)
            ndviNextPath = path.join(path.dirname(ndviNowPath), ndviNextName)

            # 执行TsHARP算法.
            tsharp(lstNowPath, ndviNowPath, ndviNextPath, predictLstNextPath, cellsizeNext)


# 使用RF模型对LST进行逐级降尺度(直接降尺度时逐级降尺度的特例).
def rfStepwise(initialLstPath, ndviDir, srtmDir, regionMethodDir, cellsizeList, gsSigma):
    # 从初始LST文件名中获取数据属性信息.
    nameParts = re.split('[._]', path.basename(initialLstPath))
    sourceLst, daynight, region = nameParts[0], nameParts[3], nameParts[4]
    dateNum = nameParts[2] if 'AMSR' in sourceLst else nameParts[1][1:]
    predictLstStr = '_'.join([sourceLst, 'LST', daynight, region, str(dateNum)])

    for i in range(len(cellsizeList) - 1):
        # 当前和下一分辨率级别.
        cellsizeNow, cellsizeNext = cellsizeList[i], cellsizeList[i + 1]

        # 判断当前分辨率级别的降尺度结果是否存在.
        predictLstNextName = '{0}_{1}.tif'.format(predictLstStr, cellsizeNext)
        predictLstNextPath = path.join(regionMethodDir, predictLstNextName)
        if not path.exists(predictLstNextPath):
            # 读取当前分辨率级别需要降尺度的LST数据.
            predictLstNowName = '{0}_{1}.tif'.format(predictLstStr, cellsizeNow)
            predictLstNowPath = path.join(regionMethodDir, predictLstNowName)
            lstNowPath = initialLstPath if cellsizeNow == cellsizeList[0] else predictLstNowPath

            # 当前和下一分辨率级别的NDVI(FVC), 高程, 坡度数据路径.
            ndviNowName = '*NDVI*{0}.tif'.format(cellsizeNow)
            ndviNowPath = glob(path.join(ndviDir, ndviNowName))[0]
            elevNowName = 'SRTM_{0}_{1}_Elev.tif'.format(cellsizeNow, region)
            elevNowPath = path.join(srtmDir, elevNowName)
            slpNowName = 'SRTM_{0}_{1}_Slp.tif'.format(cellsizeNow, region)
            slpNowPath = path.join(srtmDir, slpNowName)

            ndviNextName = path.basename(ndviNowPath).replace(cellsizeNow, cellsizeNext)
            ndviNextPath = path.join(path.dirname(ndviNowPath), ndviNextName)
            elevNextName = 'SRTM_{0}_{1}_Elev.tif'.format(cellsizeNext, region)
            elevNextPath = path.join(srtmDir, elevNextName)
            slpNextName = 'SRTM_{0}_{1}_Slp.tif'.format(cellsizeNext, region)
            slpNextPath = path.join(srtmDir, slpNextName)

            # 执行RF算法.
            factorNowPathList = [ndviNowPath, elevNowPath, slpNowPath]
            factorNextPathList = [ndviNextPath, elevNextPath, slpNextPath]
            rfDownscaling(lstNowPath, factorNowPathList, factorNextPathList, predictLstNextPath,
                          cellsizeNext)

    # 高斯平滑最终降尺度LST.
    predictLstName = '{0}_{1}.tif'.format(predictLstStr, cellsizeList[-1])
    predictLstPath = path.join(regionMethodDir, predictLstName)
    smoothLstPath = predictLstPath.replace('.tif', '_gs{0}.tif'.format(gsSigma))
    if not path.exists(smoothLstPath):
        lstPredictRas = arcpy.sa.Raster(predictLstPath)
        lstPredictLayer = arcpy.RasterToNumPyArray(lstPredictRas)
        arcpy.NumPyArrayToRaster(gaussian_filter(lstPredictLayer, gsSigma),
                                 lstPredictRas.extent.lowerLeft, lstPredictRas.meanCellWidth,
                                 lstPredictRas.meanCellHeight).save(smoothLstPath)
        arcpy.DefineProjection_management(smoothLstPath, lstPredictRas.spatialReference)


# 使用GWR模型对LST进行逐级降尺度(直接降尺度时逐级降尺度的特例).
def gwrStepwise(initialLstPath, ndviDir, srtmDir, regionMethodDir, cellsizeList):
    # 从初始LST文件名中获取数据属性信息.
    nameParts = re.split('[._]', path.basename(initialLstPath))
    sourceLst, daynight, region = nameParts[0], nameParts[3], nameParts[4]
    dateNum = nameParts[2] if 'AMSR' in sourceLst else nameParts[1][1:]
    predictLstStr = '_'.join([sourceLst, 'LST', daynight, region, str(dateNum)])

    for i in range(len(cellsizeList) - 1):
        # 当前和下一分辨率级别.
        cellsizeNow, cellsizeNext = cellsizeList[i], cellsizeList[i + 1]

        # 判断当前分辨率级别的降尺度结果是否存在.
        predictLstNextName = '{0}_{1}.tif'.format(predictLstStr, cellsizeNext)
        predictLstNextPath = path.join(regionMethodDir, predictLstNextName)
        if not path.exists(predictLstNextPath):
            # 读取当前分辨率级别需要降尺度的LST数据.
            predictLstNowName = '{0}_{1}.tif'.format(predictLstStr, cellsizeNow)
            predictLstNowPath = path.join(regionMethodDir, predictLstNowName)
            lstNowPath = initialLstPath if cellsizeNow == cellsizeList[0] else predictLstNowPath

            # 当前和下一分辨率级别的NDVI(FVC), 高程, 坡度数据路径.
            ndviNowName = '*NDVI*{0}.tif'.format(cellsizeNow)
            ndviNowPath = glob(path.join(ndviDir, ndviNowName))[0]
            elevNowName = 'SRTM_{0}_{1}_Elev.tif'.format(cellsizeNow, region)
            elevNowPath = path.join(srtmDir, elevNowName)
            slpNowName = 'SRTM_{0}_{1}_Slp.tif'.format(cellsizeNow, region)
            slpNowPath = path.join(srtmDir, slpNowName)

            ndviNextName = path.basename(ndviNowPath).replace(cellsizeNow, cellsizeNext)
            ndviNextPath = path.join(path.dirname(ndviNowPath), ndviNextName)
            elevNextName = 'SRTM_{0}_{1}_Elev.tif'.format(cellsizeNext, region)
            elevNextPath = path.join(srtmDir, elevNextName)
            slpNextName = 'SRTM_{0}_{1}_Slp.tif'.format(cellsizeNext, region)
            slpNextPath = path.join(srtmDir, slpNextName)

            # 执行RF算法.
            factorNowPathList = [ndviNowPath, elevNowPath, slpNowPath]
            factorNextPathList = [ndviNextPath, elevNextPath, slpNextPath]
            gwrDownscaling(lstNowPath, factorNowPathList, factorNextPathList, predictLstNextPath,
                          cellsizeNext)

