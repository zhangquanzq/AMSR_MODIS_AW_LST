import os
import copy
import arcpy
import datetime as dt
import functions as fn

# The constant SRTM DTM scaling factors are stored repeatedly in each paired day folder.
# Add the flag that controls whether add the residual into the estimated LST.

# Flag that assigns the region. 1 means [GansuNW], 2 means [HNSX], 3 means [YunNan], 4 means [NMG],
#   5 means [YGGY], 6 means [Heihe], 7 means [Naqu].
flg1 = 2
# Flag that assigns day or night. 1 means [Day], 2 means [Night].
flg2 = 1
# Flag that assigns the downscaling strategy. 1 means [Stepwise], 2 means [Straight].
flg3 = 1
# Flag that assigns the explanatory variables introduced into the GWR. 1 means ['NDVI;ELEV;SLP'], 2
#   means ['NDVI;ELEV'] or other combinations.
flg4 = 2
# Flag that controls whether adjust the explanatory variables to reduce the scale effect. 1 means
#   Yes, 0 means No. The adjusting method is proposed by (Jeganathan, 2011), which is verified to be
#   useless in improving the quality of downscaling result.
flg5 = 0
# Flag that assigns the source LST for downscaling. 1 means [resampled MODIS LST]. 2 means
#   [retrieved AMSRE LST].
flg6 = 1
# Flag that assigns which type of dependent variable is used in GWR model. 1 means [the estimated
#   downscaled LST] in each step, 2 means [the downscaled MODIS LST] in each step. The second method
#   is wrong, because it uses the intermediate LST data that doesn't exist in the stepwise
#   downscaling of AMSR-E LST.
flg7 = 1
# Flag that controls whether add the residual into the estimated LST. 1 means [Yes], 0 means [No].
flg8 = 1

# Preset parameters.
arcpy.CheckOutExtension("Spatial")
arcpy.env.overwriteOutput = True

testYear = '2010'
regionName = ['GansuNW', 'HNSX', 'YunNan', 'NMG', 'YGGY', 'Heihe', 'Naqu'][flg1 - 1]
dayNight = ['Day', 'Night'][flg2 - 1]
downscalingStrategy = ['Stepwise', 'Straight'][flg3 - 1]
cellsizeList = [['0.25', '0.08', '0.03', '0.01'], ['0.25', '0.01']][flg3 - 1]

# The stepwise GWR model goes wrong in the GansuNW region when Elev variable is included.
factorList1 = ['NDVI', 'ELEV', 'SLP']
factorList2 = ['NDVI', 'ELEV']
# factorList2 = ['NDVI', 'SLP']
# factorList2 = ['ELEV', 'SLP']
# factorList2 = ['NDVI']
explanatoryVarList = [factorList1, factorList2][flg4 - 1]
explanatoryVarStr = ';'.join(explanatoryVarList)

explanatoryVarType = ['Original', 'Adjusted'][flg5]
sourceLstType = ['Resampled Modis Lst', 'Retrieved Amsre Lst'][flg6 - 1]
dependentVarType = ['Modis', 'Amsre'][flg6 - 1]
# dependentVarType = ['estimated', 'modis'][flg7 - 1]

if flg8 == 0:
    downscalingStrategy = downscalingStrategy + '_noResidual'

# --------------------------------------------- Paths ----------------------------------------------
# Root path.
rootPath = 'F:\\PaperFusionLST\\'
lstRetrievalPath = rootPath + 'Data\\'
lstDownscalingPath = rootPath + 'Downscaling\\Data\\'

# The region extent feature.
regionExtentPath = lstDownscalingPath + 'Feature\\' + regionName + '_Extent.shp'

# The region folder path.
regionFolder = 'Region_{0}_{1}\\'.format(regionName, dayNight)
regionFolderPath = lstDownscalingPath + regionFolder

# The region statistics CSV file.
regionStaCsvName = 'Statistics_DateSelect_{0}_{1}.csv'.format(regionName, testYear)
regionStaCsv2Name = 'Statistics_Downscaling_{0}_{1}_{2}_{3}.csv'.\
    format(regionName, testYear, dependentVarType, downscalingStrategy)
regionStaCsvPath = regionFolderPath + regionStaCsvName
regionStaCsv2Path = regionFolderPath + regionStaCsv2Name


# The raw SRTM DTM.
srtmDtmFolderPath = lstDownscalingPath + 'Raw_SRTM_DTM\\'
srtmElevCnPath = srtmDtmFolderPath + 'SRTM_Elev_3arc_CN.tif'
srtmSlpCnPath = srtmDtmFolderPath + 'SRTM_Slp_3arc_CN.tif'

# The raw AMSRE LST.
amsreGridFolder = 'AMSRE_QuarterDegreeGrid\\'
amsreLstFolder = 'AMSRE_7_LST_TIF\\'
amsreLstYearFolder = 'AMSRE_{0}XXXX\\'.format(testYear)
amsreLstYearFolderPath = lstRetrievalPath + amsreGridFolder + amsreLstFolder + amsreLstYearFolder

# The raw MODIS LST.
modisLstFolder = 'MYD11A1\\'
modisLstMosaicFolder = 'MYD11A1_2_LST_MosaicCN_TIF\\'
modisLstYearFolder = 'MYD11A1_{0}XXXX\\'.format(testYear)
modisLstYearFolderPath = lstRetrievalPath + modisLstFolder + modisLstMosaicFolder + \
                         modisLstYearFolder

# The raw MODIS NDVI.
modisNdviFolder = 'MYD13A2\\'
modisNdviMosaicFolder = 'MYD13A2_2_NDVI_MosaicCN_TIF\\'
modisNdviYearFolder = 'MYD13A2_{0}XXXX\\'.format(testYear)
modisNdviYearFolderPath = lstRetrievalPath + modisNdviFolder + modisNdviMosaicFolder + \
                          modisNdviYearFolder

# ------------------------------------------- Data list --------------------------------------------
# Get the AMSRE LST and date lists.
arcpy.env.workspace = amsreLstYearFolderPath
amsreLstList = arcpy.ListRasters('AMSRE_LST*{0}.tif'.format(dayNight))
amsreLstDateList = [amsreLst.split('_')[3] for amsreLst in amsreLstList]

# Get the MODIS LST and date lists.
arcpy.env.workspace = modisLstYearFolderPath
modisLstList = arcpy.ListRasters('*LST_{0}.tif'.format(dayNight))
modisLstDateList = [modisLst[9:17] for modisLst in modisLstList]

# Get the MODIS NDVI, QC and their date lists. The date lists are datetime object
arcpy.env.workspace = modisNdviYearFolderPath
modisNdviList = arcpy.ListRasters('*NDVI.tif')
modisNdviQcList = arcpy.ListRasters('*QC.tif')
modisNdviDateObjList, modisNdviQcDateObjList = [], []
for modisNdvi in modisNdviList:
    modisDate = fn.yday2ymd(modisNdvi[9:16])
    ndviDateObj = dt.date(int(modisDate[0:4]), int(modisDate[4:6]), int(modisDate[6:8]))
    modisNdviDateObjList.append(ndviDateObj)
for modisNdviQc in modisNdviQcList:
    modisQcDate = fn.yday2ymd(modisNdviQc[9:16])
    modisQcDateObj = dt.date(int(modisQcDate[0:4]), int(modisQcDate[4:6]), int(modisQcDate[6:8]))
    modisNdviQcDateObjList.append(modisQcDateObj)

# Get the records of paired AMSRE and MODIS LST from the region statistics csv file.
staFileId = open(regionStaCsvPath, 'r')
fieldName = staFileId.readline()
recordList = staFileId.readlines()
staFileId.close()
yearDateList, rList, pixelPercentList = [], [], []
for record in recordList:
    record = record[:-1].split(',')
    yearDateList.append(record[0])
    rList.append(float(record[1]))
    pixelPercentList.append(float(record[2]))

# -------------------------------------------- Procedures ------------------------------------------
# Preform the GWR in each day recorded in the csv file. The order is complied with the correlation
#   coefficients (R) from max to min.
errorRecordList = []
rTempList = copy.deepcopy(rList)
n = 0
while len(rTempList) > 0:
    # Pick out the record (r, yearDate, pixelPercent) in which the r is the max in current cycle.
    rMax = max(rTempList)
    rIndex = rList.index(rMax)
    r = rList[rIndex]
    yearDate = yearDateList[rIndex]
    pixelPercent = pixelPercentList[rIndex]
    rTempList.remove(rMax)
    n = n + 1

    print('LST Downscaling {0}/{1}\n   Region: {2},   Date: {3},   R: {4},   Pixel percentage: {5}'
          '\n   Explanatory variables: {6} {7}\n   Dependent variable: {8}.'
          .format(n, len(rList), regionName, yearDate, r, pixelPercent, explanatoryVarType,
                  explanatoryVarStr, sourceLstType))

    # [Step 1]: Clip and resample the dependent and independent variables.
    print('1. Clipping and resampling raw images.')

    # Get the filename of AMSRE and MODIS LST in each day.
    rawAmsreLstName = amsreLstList[amsreLstDateList.index(yearDate)]
    rawModisLstName = modisLstList[modisLstDateList.index(yearDate)]

    # Get the filenames of NDVI and its Qc that are closest to the date of MODIS and AMSRE LSTs.
    yearDateObj = dt.date(int(yearDate[0:4]), int(yearDate[4:6]), int(yearDate[6:8]))
    ndviDateDiff, ndviQcDateDiff = [], []
    for ndviDateObj in modisNdviDateObjList:
        ndviDateDiff.append(abs((ndviDateObj - yearDateObj).days))
    for ndviQcDateObj in modisNdviQcDateObjList:
        ndviQcDateDiff.append(abs((ndviQcDateObj - yearDateObj).days))
    rawNdviName = modisNdviList[ndviDateDiff.index(min(ndviDateDiff))]
    rawNdviQcName = modisNdviQcList[ndviQcDateDiff.index(min(ndviQcDateDiff))]

    # Resource data paths, storing the clipped raw data necessary for regional GWR testing.
    resourceDataPath = regionFolderPath + str(yearDate) + '_Resource\\'
    if not os.path.exists(resourceDataPath):
        os.mkdir(resourceDataPath)

    # Clip AMSRE LST, MODIS LST, NDVI, SRTM Elev, Slp into the extent of testing region.
    amsreLstRegionName = rawAmsreLstName.replace(dayNight, dayNight + '_' + regionName)
    amsreLstRegionPath = resourceDataPath + amsreLstRegionName
    if not arcpy.Exists(amsreLstRegionPath):
        rawAmsreLstPath = amsreLstYearFolderPath + rawAmsreLstName
        arcpy.Clip_management(rawAmsreLstPath, "#", amsreLstRegionPath, regionExtentPath)

    modisLstRegionName = rawModisLstName.replace(dayNight, dayNight + '_' + regionName)
    modisLstRegionPath = resourceDataPath + modisLstRegionName
    if not arcpy.Exists(modisLstRegionPath):
        rawModisLstPath = modisLstYearFolderPath + rawModisLstName
        arcpy.Clip_management(rawModisLstPath, "#", modisLstRegionPath, regionExtentPath)
        lstRegionRaster = arcpy.sa.Times(arcpy.Raster(modisLstRegionPath), 0.02)
        lstRegionRaster = arcpy.sa.SetNull(lstRegionRaster, lstRegionRaster, 'Value = 0')
        lstRegionRaster.save(modisLstRegionPath)

    ndviRegionName = rawNdviName.replace('NDVI', 'NDVI_' + regionName)
    ndviRegionPath = resourceDataPath + ndviRegionName
    if not arcpy.Exists(ndviRegionPath):
        rawNdviPath = modisNdviYearFolderPath + rawNdviName
        outTempNdviName = resourceDataPath + ndviRegionName.replace(regionName, regionName + 'b')
        arcpy.Clip_management(rawNdviPath, "#", outTempNdviName, regionExtentPath)
        regionNdviRaster = arcpy.sa.Divide(outTempNdviName, 10000.0)
        regionNdviRaster.save(ndviRegionPath)
        arcpy.Delete_management(outTempNdviName)

    elevRegionName = 'SRTM_Elev_' + regionName + '.tif'
    elevRegionPath = resourceDataPath + elevRegionName
    if not arcpy.Exists(elevRegionPath):
        arcpy.Clip_management(srtmElevCnPath, "#", elevRegionPath, regionExtentPath)

    slpRegionName = 'SRTM_Slp_' + regionName + '.tif'
    slpRegionPath = resourceDataPath + slpRegionName
    if not arcpy.Exists(slpRegionPath):
        arcpy.Clip_management(srtmSlpCnPath, "#", slpRegionPath, regionExtentPath)

    # GWR data paths, storing the intermediate and output data derived from GWR process.
    regionRasterFolderName = str(yearDate) + '_' + downscalingStrategy + '_Rasters\\'
    regionRasterFolderPath = regionFolderPath + regionRasterFolderName
    if not os.path.exists(regionRasterFolderPath):
        os.mkdir(regionRasterFolderPath)

    regionVarFolderName = str(yearDate) + '_' + downscalingStrategy + '_Variables\\'
    regionVarFolderPath = regionFolderPath + regionVarFolderName
    if not os.path.exists(regionVarFolderPath):
        os.mkdir(regionVarFolderPath)

    # Get the preliminary data and store them in the region folder, including 0.25 degree AMSRE LST,
    #   0.01 degree MODIS LST, NDVI, and 0.01 degree SRTM DTMs.
    arcpy.env.workspace = resourceDataPath
    resourceList = arcpy.ListRasters('*.tif')
    for resource in resourceList:
        if 'AMSRE' in resource:
            replacement = regionName + '_' + cellsizeList[0]
            outResourcePath = regionRasterFolderPath + resource.replace(regionName, replacement)
            if not arcpy.Exists(outResourcePath):
                arcpy.Copy_management(resource, outResourcePath)
        else:
            if 'LST_' + dayNight in resource:
                replacement = regionName + '_' + cellsizeList[-1] + '_ref'
            else:
                replacement = regionName + '_' + cellsizeList[-1]
            outResourcePath = regionRasterFolderPath + resource.replace(regionName, replacement)
            if not arcpy.Exists(outResourcePath):
                arcpy.env.snapRaster = modisLstRegionPath
                arcpy.Resample_management(resource, outResourcePath, cellsizeList[-1], 'BILINEAR')

    # Resample the MODIS LST, NDVI, and SRTM Elev, Slp into the cellsize in the list.
    arcpy.env.workspace = regionRasterFolderPath
    arcpy.env.snapRaster = amsreLstRegionPath
    cellsizeList2 = cellsizeList[::-1]
    for i in range(len(cellsizeList2) - 1):
        smallPixel = cellsizeList2[i]
        largePixel = cellsizeList2[i + 1]
        regionSmallPixel = regionName + '_' + smallPixel
        regionLargePixel = regionName + '_' + largePixel

        lstSmallPixelName = modisLstRegionName.replace(regionName, regionSmallPixel + '_ref')
        lstLargePixelName = modisLstRegionName.replace(regionName, regionLargePixel + '_ref')
        if not arcpy.Exists(lstLargePixelName):
            arcpy.Resample_management(lstSmallPixelName, lstLargePixelName, largePixel, 'BILINEAR')
            outTempModisLstName = lstLargePixelName.replace(largePixel, largePixel + 'b')
            arcpy.Rename_management(lstLargePixelName, outTempModisLstName)
            arcpy.Clip_management(outTempModisLstName, '#', lstLargePixelName, regionExtentPath)
            arcpy.Delete_management(outTempModisLstName)

        ndviSmallPixelName = ndviRegionName.replace(regionName, regionSmallPixel)
        ndviLargePixelName = ndviRegionName.replace(regionName, regionLargePixel)
        if not arcpy.Exists(ndviLargePixelName):
            arcpy.Resample_management(ndviSmallPixelName, ndviLargePixelName, largePixel,
                                      'BILINEAR')
            outTempNdviName = ndviLargePixelName.replace(largePixel, largePixel + 'b')
            arcpy.Rename_management(ndviLargePixelName, outTempNdviName)
            arcpy.Clip_management(outTempNdviName, '#', ndviLargePixelName, regionExtentPath)
            arcpy.Delete_management(outTempNdviName)

        elevSmallPixelName = elevRegionName.replace(regionName, regionSmallPixel)
        elevLargePixelName = elevRegionName.replace(regionName, regionLargePixel)
        if not arcpy.Exists(elevLargePixelName):
            arcpy.Resample_management(elevSmallPixelName, elevLargePixelName, largePixel,
                                      'BILINEAR')
            outTempElevName = elevLargePixelName.replace(largePixel, largePixel + 'b')
            arcpy.Rename_management(elevLargePixelName, outTempElevName)
            arcpy.Clip_management(outTempElevName, '#', elevLargePixelName, regionExtentPath)
            arcpy.Delete_management(outTempElevName)

        slpSmallPixelName = slpRegionName.replace(regionName, regionSmallPixel)
        slpLargePixelName = slpRegionName.replace(regionName, regionLargePixel)
        if not arcpy.Exists(slpLargePixelName):
            arcpy.Resample_management(slpSmallPixelName, slpLargePixelName, largePixel, 'BILINEAR')
            outTempSlpName = slpLargePixelName.replace(largePixel, largePixel + 'b')
            arcpy.Rename_management(slpLargePixelName, outTempSlpName)
            arcpy.Clip_management(outTempSlpName, '#', slpLargePixelName, regionExtentPath)
            arcpy.Delete_management(outTempSlpName)
    arcpy.ResetEnvironments()

    # [Step 2]: GWR process.
    try:
        print('2. {0} GWR process.'.format(downscalingStrategy))
        arcpy.env.workspace = regionRasterFolderPath
        for i in range(0, len(cellsizeList) - 1):
            largePixel = cellsizeList[i]
            smallPixel = cellsizeList[i + 1]
            regionLargePixel = regionName + '_' + largePixel
            regionSmallPixel = regionName + '_' + smallPixel
            print('   From {0} to {1} degree.'.format(largePixel, smallPixel))

            # Create the folder storing the images and features of the variables in each cellsize
            #   level.
            largePixelString = largePixel.replace('.', 'p')
            pixelSizeFolderName = 'r' + largePixelString + '_' + explanatoryVarType + \
                                  'ExplanatoryVar_' + sourceLstType.replace(' ', '') + \
                                  'DependentVar\\'
            pixelSizeFolderPath = regionVarFolderPath + pixelSizeFolderName
            if not os.path.exists(pixelSizeFolderPath):
                os.mkdir(pixelSizeFolderPath)

            # Convert the AMSRE LST image to point feature and extract NDVI, Elev, Slp values into
            #   these points.
            if flg6 == 1:  # resampled MODIS LST.
                if largePixel == cellsizeList[0]:  # 0.25 degree.
                    lstRegionLargePixel = regionLargePixel + '_ref'
                else:
                    replacement = '_est_' + explanatoryVarType + 'ExplanatoryVar_' + \
                                  sourceLstType.replace(' ', '') + 'DependentVar'
                    lstRegionLargePixel = regionLargePixel + [replacement, '_ref'][flg7 - 1]
                lstRegionLargePixelName = modisLstRegionName.replace(regionName,
                                                                     lstRegionLargePixel)
            else:  # retrieved AMSRE LST.
                if largePixel == cellsizeList[0]:  # 0.25 degree.
                    lstRegionLargePixel = regionLargePixel
                else:
                    replacement = '_est_' + explanatoryVarType + 'ExplanatoryVar_' + \
                                  sourceLstType.replace(' ', '') + 'DependentVar'
                    lstRegionLargePixel = regionLargePixel + [replacement, '_ref'][flg7 - 1]
                lstRegionLargePixelName = amsreLstRegionName.replace(regionName,
                                                                     lstRegionLargePixel)

            variablesPointPath = pixelSizeFolderPath + 'Variables_' + largePixelString + '.shp'
            if not arcpy.Exists(variablesPointPath):
                ndviRegionLargePixelName = ndviRegionName.replace(regionName, regionLargePixel)
                elevRegionLargePixelName = elevRegionName.replace(regionName, regionLargePixel)
                slpRegionLargePixelName = slpRegionName.replace(regionName, regionLargePixel)
                inRasterList = [[ndviRegionLargePixelName, 'NDVI'],
                                [elevRegionLargePixelName, 'ELEV'],
                                [slpRegionLargePixelName, 'SLP']]

                arcpy.RasterToPoint_conversion(lstRegionLargePixelName, variablesPointPath)
                arcpy.AddField_management(variablesPointPath, 'LST', 'DOUBLE')
                arcpy.CalculateField_management(variablesPointPath, 'LST', "!grid_code!",
                                                "PYTHON_9.3")
                arcpy.DeleteField_management(variablesPointPath, ["grid_code"])
                arcpy.sa.ExtractMultiValuesToPoints(variablesPointPath, inRasterList)

                # Replace the nodata value (-9999) in the NDVI records with 0.
                cursor = arcpy.da.UpdateCursor(variablesPointPath, ['NDVI'])
                for row in cursor:
                    if row[0] == -9999:
                        row[0] = 0
                        cursor.updateRow(row)

                # Execute GWR process and export the images of regression coefficients, including
                #   the intercept, the slope of each explanatory variable, and the residual.
                gwrPointPath = pixelSizeFolderPath + 'Variables_' + largePixelString + '_gwr.shp'
                gwrSuppPath = pixelSizeFolderPath + 'Variables_' + largePixelString + '_gwr_supp.dbf'
                gwrResidualPath = pixelSizeFolderPath + 'residual'
                aligningImageName = elevRegionName.replace(regionName, regionSmallPixel)

                arcpy.env.snapRaster = aligningImageName
                arcpy.env.extent = aligningImageName
                arcpy.GeographicallyWeightedRegression_stats(variablesPointPath, 'LST',
                                                             explanatoryVarStr, gwrPointPath,
                                                             'ADAPTIVE', 'AICc', '#', '#', '#',
                                                             pixelSizeFolderPath, smallPixel)
                arcpy.EmpiricalBayesianKriging_ga(gwrPointPath, 'Residual', '#', gwrResidualPath,
                                                  smallPixel)

                # Construct the regression model of the explanatory variable between two adjacent
                #   resolution levels, and export the images of adjusted explanatory variable.
                if flg5 == 1:
                    neighborN = arcpy.da.SearchCursor(gwrSuppPath, ['VARIABLE']).next()[0]
                    ndviSmallPixelName = ndviRegionName.replace(regionName, regionSmallPixel)
                    elevSmallPixelName = elevRegionName.replace(regionName, regionSmallPixel)
                    slpSmallPixelName = slpRegionName.replace(regionName, regionSmallPixel)
                    ndviSmallPixelName2 = ndviSmallPixelName.replace(smallPixel, smallPixel + '_fit')
                    elevSmallPixelName2 = elevSmallPixelName.replace(smallPixel, smallPixel + '_fit')
                    slpSmallPixelName2 = slpSmallPixelName.replace(smallPixel, smallPixel + '_fit')
                    if not arcpy.Exists(ndviSmallPixelName2):
                        fn.ScalingRegress(ndviRegionLargePixelName, ndviSmallPixelName,
                                          ndviSmallPixelName2, neighborN)
                    if not arcpy.Exists(elevSmallPixelName2):
                        fn.ScalingRegress(elevRegionLargePixelName, elevSmallPixelName,
                                          elevSmallPixelName2, neighborN)
                    if not arcpy.Exists(slpSmallPixelName2):
                        fn.ScalingRegress(slpRegionLargePixelName, slpSmallPixelName,
                                          slpSmallPixelName2, neighborN)

                # Estimate LST using the explanatory variables and coefficients derived from GWR.
                rastersList = arcpy.ListRasters('*.tif')
                slopeRasterList, variableRasterList = [], []
                for variable in explanatoryVarList:
                    slopeRasterList.append(arcpy.Raster(pixelSizeFolderPath + variable.lower()))
                    if flg5 == 0:
                        patternString = '{0}_{1}_{2}.tif'\
                            .format(variable, regionName, smallPixel).lower()
                    else:
                        patternString = '{0}_{1}_{2}_fit.tif'\
                            .format(variable, regionName, smallPixel).lower()
                    for j in range(len(rastersList)):
                        if patternString in rastersList[j].lower():
                            variableRasterList.append(arcpy.Raster(rastersList[j]))
                interceptRaster = arcpy.Raster(pixelSizeFolderPath + 'intercept')
                residualRaster = arcpy.Raster(pixelSizeFolderPath + 'residual')
                if flg8 == 1:
                    estimatedLstRaster = interceptRaster + residualRaster
                else:
                    estimatedLstRaster = interceptRaster
                for j in range(len(slopeRasterList)):
                    estimatedLstRaster = estimatedLstRaster + slopeRasterList[j] * \
                                         variableRasterList[j]
                replacement = regionSmallPixel + '_est_' + explanatoryVarType + 'ExplanatoryVar_'\
                              + sourceLstType.replace(' ', '') + 'DependentVar'
                if flg6 == 1:
                    lstRegionName = modisLstRegionName.replace(regionName, replacement)
                    lstRegionDiffName = modisLstRegionName.replace(regionName, replacement + '_diff')
                else:
                    lstRegionName = amsreLstRegionName.replace(regionName, replacement)
                    lstRegionDiffName = amsreLstRegionName.replace(regionName, replacement + '_diff')
                estimatedLstRaster.save(lstRegionName)
                referenceLstName = modisLstRegionName.replace(regionName, regionSmallPixel + '_ref')
                referenceLstRaster = arcpy.Raster(referenceLstName)
                LstDifference = estimatedLstRaster - referenceLstRaster
                LstDifference.save(lstRegionDiffName)

        # [Step 3]: Statistics of the estimated LST.
        print('3. Accuracy statistics.')
        replacement = regionName + '_' + cellsizeList[-1] + '_est_' + explanatoryVarType \
            + 'ExplanatoryVar_' + sourceLstType.replace(' ', '') + 'DependentVar'
        if flg6 == 1:
            estimatedLstName = modisLstRegionName.replace(regionName, replacement)
        else:
            estimatedLstName = amsreLstRegionName.replace(regionName, replacement)
        replacement = regionName + '_' + cellsizeList[-1] + '_ref'
        referenceLstName = modisLstRegionName.replace(regionName, replacement)
        ndvi0p01RegionName = ndviRegionName.replace(regionName, regionName + '_0.01')

        estimatedLstPath = regionRasterFolderPath + estimatedLstName
        referenceLstPath = regionRasterFolderPath + referenceLstName

        ndvi0p01RegionPath = regionRasterFolderPath + ndvi0p01RegionName
        estimatedLstRaster = arcpy.Raster(estimatedLstPath)
        referenceLstRaster = arcpy.Raster(referenceLstPath)
        ndviRaster = arcpy.Raster(ndvi0p01RegionPath)
        # Remove the pixels of Water in LST image where the LST is not correctly estimated.
        if regionName in ['HNSX']:
            estimatedLstRaster = arcpy.sa.SetNull(ndviRaster, estimatedLstRaster, 'Value <= 0.1')

        lstDifferenceRaster = estimatedLstRaster - referenceLstRaster
        bias = lstDifferenceRaster.mean
        rmse = arcpy.sa.SquareRoot(arcpy.sa.Power(lstDifferenceRaster, 2).mean)
        del lstDifferenceRaster
        print('   Bias:{0:.3f}, RMSE:{1:.3f}\n'.format(bias, rmse))
        errorRecordList.append('{0}, {1:.3f}, {2:.3f}\n'.format(yearDate, bias, rmse))
    except:
        pass

# Export the statistics of error.
errorFileId = open(regionStaCsv2Path, 'w')
errorFileId.writelines('YearDate, Bias, RMSE\n')
errorFileId.writelines(errorRecordList)
errorFileId.close()
