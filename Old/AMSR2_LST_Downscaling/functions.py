import arcpy
import numpy as np


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
    #   image. N equals neighborN, in which the each pixel itself is included.
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


def leapyear(year_number):
    if year_number % 100 == 0:
        if year_number % 400 == 0:
            return True
        else:
            return False
    else:
        if year_number % 4 == 0:
            return True
        else:
            return False


def ymd2yday(ymd):
    # Convert YearMonthDay to YearDayofyear. Both input and output are strings.
    # Format of input: 'yyyymmdd'. For example: '20100503'.
    # Format of output: 'yyyyday'. For example: '2010123'
    year = ymd[0:4]
    month = int(ymd[4:6])
    day = int(ymd[6:8])

    monthday = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if leapyear(int(year)):
        monthday[1] = 29
    monthsum = sum(monthday[0:month - 1])
    dday = monthsum + day
    yday = '{0}{1:03d}'.format(year, dday)

    return yday


def yday2ymd(year_dayofyear):
    year = int(year_dayofyear[0:4])
    day = int(year_dayofyear[4:7])
    month_day = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if leapyear(year):
        month_day[1] = 29

    i = 0
    while day > 0:
        day = day - month_day[i]
        i = i + 1

    return '{0}{1:>02}{2:>02}'.format(year, i, day + month_day[i - 1])
