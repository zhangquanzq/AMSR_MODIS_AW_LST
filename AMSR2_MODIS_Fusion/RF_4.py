import numpy as np
from scipy.ndimage import gaussian_filter
import pandas as pd
from osgeo import gdal
import sklearn.ensemble
import glob
import os


# ------------------------------------ Define functions --------------------------------------------
def writeTiff(im_data, im_width, im_height, im_bands, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape

    # create GDAL .tif file
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    if dataset != None:
        dataset.SetGeoTransform(im_geotrans) # Write the affine transform
        dataset.SetProjection(im_proj) # write the projection
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset


# ------------------------------------ Flags ------------------------------------
# Flag that assigns the region. 1 means [GansuNW], 2 means [HNSX], 3 means [YunNan], 4 means [NMG],
#   5 means [YGGY], 6 means [Heihe], 7 means [Naqu].
flg1 = 7
# Flag that assigns day or night. 1 means [Day], 2 means [Night].
flg2 = 1
# Flag that assigns the downscaling strategy. 1 means [Stepwise], 2 means [Straight].
flg3 = 1
# Flag that assign whether smooth the RF_Regressed LST with Gaussian Filter. 1 means [Yes], 0 means
#   [No].
flg4 = 1

testYear = '2010'
gsFilterSigma = 0.7
regionName = ['GansuNW', 'HNSX', 'YunNan', 'NMG', 'YGGY', 'Heihe', 'Naqu'][flg1 - 1]
dayNight = ['Day', 'Night'][flg2 - 1]
downscalingStrategy = ['Stepwise', 'Straight'][flg3 - 1]

dtm = 'Elev' if regionName in ['HNSX', 'YGGY'] else 'Slp'

# --------------------------------------------- Paths ----------------------------------------------
rootPath = 'F:\\PaperFusionLST\\Fusion\\Data\\'
qcTimeRootPath = 'F:\\PaperFusionLST\\Data\\MYD11A1\\MYD11A1_2_LST_MosaicCN_TIF\\MYD11A1_2010XXXX\\'

regionFolder = 'Region_' + regionName + '_' + dayNight + '\\'
regionPath = rootPath + regionFolder

if flg4 == 1:
    RfFolder = '0_RF_GSFilter_{0}'.format(gsFilterSigma)
else:
    RfFolder = '0_RF'
outputRfPath = regionPath + RfFolder + '\\'
if not os.path.exists(outputRfPath):
    os.mkdir(outputRfPath)

# ------------------------------------------ Procedures --------------------------------------------
qcFileList = glob.glob(qcTimeRootPath + 'MYD11A1*QC*{0}.tif'.format(dayNight))
qcDateList = [qcFile.split('\\')[-1][9:17] for qcFile in qcFileList]
timeFileList = glob.glob(qcTimeRootPath + 'MYD11A1*Time*{0}.tif'.format(dayNight))
timeDateList = [timeFile.split('\\')[-1][9:17] for timeFile in timeFileList]

print('Blending LST in region {0} using Random Forest Regression'.format(regionName))
dateFoldersList = glob.glob(regionPath + '*Stepwise_Rasters')
for dateFolder in dateFoldersList:
    dateString = dateFolder.split('\\')[-1][:8]
    print(dateString)

    outputHalfName = dateFolder.split('\\')[-1][:-8]
    outputRefLstPath = outputRfPath + 'LST_' + outputHalfName + '_' + regionName + '_ModisRef.tif'
    outputRfLstPath = outputRfPath + 'LST_' + outputHalfName + '_' + regionName + '_Rf.tif'
    outputRfSmoothLstPath = outputRfPath + 'LST_' + outputHalfName + '_' + regionName + '_RfSmooth.tif'
    outputFusionLstPath = outputRfPath + 'LST_' + outputHalfName + '_' + regionName + '_Fusion.tif'
    outputLstQcPath = outputRfPath + 'LST' + outputHalfName + regionName + '_ModisTime.tif'
    if flg4 == 1:
        if os.path.exists(outputRfLstPath) and os.path.exists(outputRfSmoothLstPath) and \
                os.path.exists(outputFusionLstPath) and os.path.exists(outputRefLstPath)\
                and os.path.exists(outputLstQcPath):
            continue
    else:
        if os.path.exists(outputRfLstPath) and os.path.exists(outputFusionLstPath) and \
                os.path.exists(outputRefLstPath) and os.path.exists(outputLstQcPath):
            continue

    # Get fullname for each input data
    modisLstQcPath = qcFileList[qcDateList.index(dateString)]
    modisLstTimePath = timeFileList[timeDateList.index(dateString)]
    modisLstPath = glob.glob(dateFolder + '\\MYD11A1.A*_LST_{0}_*_0.01_ref.tif'.format(dayNight))
    amsreLstPath = glob.glob(dateFolder + '\\AMSRE_LST_merged*_0.01_*Var.tif')
    modisNdviPath = glob.glob(dateFolder + '\\MYD13A2.A*NDVI*_0.01.tif')
    srtmDtmPath = glob.glob(dateFolder + '\\SRTM_' + dtm + '*_0.01.tif')
    if len(modisLstPath) == 0 or len(amsreLstPath) == 0 or \
            len(modisNdviPath) == 0 or len(srtmDtmPath) == 0 or len(modisLstQcPath) == 0\
            or len(modisLstTimePath) == 0:
        continue

    # Open each input data and return the corresponding pointer
    modisLstQcObj = gdal.Open(modisLstQcPath)
    modisLstObj = gdal.Open(modisLstPath[0])
    amsreLstObj = gdal.Open(amsreLstPath[0])
    modisNdviObj = gdal.Open(modisNdviPath[0])
    srtmDtmObj = gdal.Open(srtmDtmPath[0])
    modisLstTimeObj = gdal.Open(modisLstTimePath)

    # Get metadata information for each input data
    modisLstColN, modisLstRowN = modisLstObj.RasterXSize, modisLstObj.RasterYSize
    amsreLstColN, amsreLstRowN = amsreLstObj.RasterXSize, amsreLstObj.RasterYSize
    modisNdviColN, modisNdviRowN = modisNdviObj.RasterXSize, modisNdviObj.RasterYSize
    srtmDtmColN, srtmDtmRowN = srtmDtmObj.RasterXSize, srtmDtmObj.RasterYSize
    modisLstQcColN, modisLstQcRowN = modisLstQcObj.RasterXSize, modisLstQcObj.RasterYSize
    modisLstTimeColN, modisLstTimeRowN = modisLstTimeObj.RasterXSize, modisLstTimeObj.RasterYSize

    modisLstGeotrans = modisLstObj.GetGeoTransform()
    amsreLstGeotrans = amsreLstObj.GetGeoTransform()
    modisNdviGeotrans = modisNdviObj.GetGeoTransform()
    srtmDtmGeotrans = srtmDtmObj.GetGeoTransform()
    modisLstQcGeotrans = modisLstQcObj.GetGeoTransform()
    modisLstTimeGeotrans = modisLstTimeObj.GetGeoTransform()

    # The left-top geographical coordinates
    modisLstXMin, modisLstYMax = modisLstGeotrans[0], modisLstGeotrans[3]
    amsreLstXMin, amsreLstYMax = amsreLstGeotrans[0], amsreLstGeotrans[3]
    modisNdviXMin, modisNdviYMax = modisNdviGeotrans[0], modisNdviGeotrans[3]
    srtmDtmXMin, srtmDtmYMax = srtmDtmGeotrans[0], srtmDtmGeotrans[3]
    modisLstQcXMin, modisLstQcYMax = modisLstQcGeotrans[0], modisLstQcGeotrans[3]
    modisLstTimeXMin, modisLstTimeYMax = modisLstTimeGeotrans[0], modisLstTimeGeotrans[3]

    # The right-bottom geographical coordinates
    modisLstXMax = modisLstGeotrans[0] + modisLstColN * modisLstGeotrans[1]
    modisLstYMin = modisLstGeotrans[3] + modisLstRowN * modisLstGeotrans[5]
    amsreLstXMax = amsreLstGeotrans[0] + amsreLstColN * amsreLstGeotrans[1]
    amsreLstYMin = amsreLstGeotrans[3] + amsreLstRowN * amsreLstGeotrans[5]
    modisNdviXMax = modisNdviGeotrans[0] + modisNdviColN * modisNdviGeotrans[1]
    modisNdviYMin = modisNdviGeotrans[3] + modisNdviRowN * modisNdviGeotrans[5]
    srtmDtmXMax = srtmDtmGeotrans[0] + srtmDtmColN * srtmDtmGeotrans[1]
    srtmDtmYMin = srtmDtmGeotrans[3] + srtmDtmRowN * srtmDtmGeotrans[5]
    modisLstQcXMax = modisLstQcGeotrans[0] + modisLstQcColN * modisLstQcGeotrans[1]
    modisLstQcYMin = modisLstQcGeotrans[3] + modisLstQcRowN * modisLstQcGeotrans[5]
    modisLstTimeXMax = modisLstTimeGeotrans[0] + modisLstTimeColN * modisLstTimeGeotrans[1]
    modisLstTimeYMin = modisLstTimeGeotrans[3] + modisLstTimeRowN * modisLstTimeGeotrans[5]

    # Calculate how much each input image is cropped
    xMinVector = np.array([modisLstXMin, amsreLstXMin, modisNdviXMin, srtmDtmXMin, modisLstQcXMin,
                           modisLstTimeXMin])
    yMaxVector = np.array([modisLstYMax, amsreLstYMax, modisNdviYMax, srtmDtmYMax, modisLstQcYMax,
                           modisLstTimeYMax])
    xMaxVector = np.array([modisLstXMax, amsreLstXMax, modisNdviXMax, srtmDtmXMax, modisLstQcXMax,
                           modisLstTimeXMax])
    yMinVector = np.array([modisLstYMin, amsreLstYMin, modisNdviYMin, srtmDtmYMin, modisLstQcYMin,
                           modisLstTimeYMin])
    outputXMin, outputXMax = xMaxVector.min(), xMinVector.max()
    outputYMin, outputYMax = yMaxVector.min(), yMinVector.max()
    xMinDiffVector, yMaxDiffVector = outputXMax - xMinVector, yMaxVector - outputYMin
    xMaxDiffVector, yMinDiffVector = xMaxVector - outputXMin, outputYMax - yMinVector
    xMinClipColNVector = np.int16(np.rint(xMinDiffVector / 0.01))
    xMaxClipColNVector = np.int16(np.rint(xMaxDiffVector / 0.01))
    yMaxClipRowNVector = np.int16(np.rint(yMaxDiffVector / 0.01))
    yMinClipRowNVector = np.int16(np.rint(yMinDiffVector / 0.01))

    # Get matching each input data values
    modisLstArray = modisLstObj.ReadAsArray()
    amsreLstArray = amsreLstObj.ReadAsArray()
    modisNdviArray = modisNdviObj.ReadAsArray()
    srtmDtmArray = srtmDtmObj.ReadAsArray()
    modisLstQcArray = modisLstQcObj.ReadAsArray()
    modisLstTimeArray = modisLstTimeObj.ReadAsArray()

    modisLstArray = modisLstArray[
                    yMaxClipRowNVector[0]:(modisLstRowN - yMinClipRowNVector[0]),
                    xMinClipColNVector[0]:(modisLstColN - xMaxClipColNVector[0])]
    amsreLstArray = amsreLstArray[
                    yMaxClipRowNVector[1]:(amsreLstRowN - yMinClipRowNVector[1]),
                    xMinClipColNVector[1]: (amsreLstColN - xMaxClipColNVector[1])]
    modisNdviArray = modisNdviArray[
                     yMaxClipRowNVector[2]:(modisNdviRowN - yMinClipRowNVector[2]),
                     xMinClipColNVector[2]: (modisNdviColN - xMaxClipColNVector[2])]
    srtmDtmArray = srtmDtmArray[
                   yMaxClipRowNVector[3]:(srtmDtmRowN - yMinClipRowNVector[3]),
                   xMinClipColNVector[3]:(srtmDtmColN - xMaxClipColNVector[3])]
    modisLstQcArray = modisLstQcArray[
                      yMaxClipRowNVector[4]:(modisLstQcRowN - yMinClipRowNVector[4]),
                      xMinClipColNVector[4]:(modisLstQcColN - xMaxClipColNVector[4])]
    modisLstTimeArray = modisLstTimeArray[
                        yMaxClipRowNVector[5]:(modisLstTimeRowN - yMinClipRowNVector[5]),
                        xMinClipColNVector[5]:(modisLstTimeColN - xMaxClipColNVector[5])]
    modisLstArray2 = np.zeros_like(modisLstArray)
    modisLstArray2[modisLstQcArray == 0] = modisLstArray[modisLstQcArray == 0]

    # Reshape each input data into one column
    modisLstVector = modisLstArray2.reshape(-1, 1)
    amsreLstVector = amsreLstArray.reshape(-1, 1)
    modisNdviVector = modisNdviArray.reshape(-1, 1)
    srtmDtmVector = srtmDtmArray.reshape(-1, 1)

    # Get training data
    trainData = np.hstack([modisLstVector, amsreLstVector, modisNdviVector, srtmDtmVector])
    trainData = pd.DataFrame(trainData, columns=["modisLst", "amsreLst", "modisNdvi", "srtmDtm"])
    trainData = trainData[trainData['modisLst'] > 0]
    trainData = trainData[trainData['amsreLst'] >= 0]
    trainData = trainData[trainData['modisNdvi'] >= -1]
    trainData = trainData[trainData['srtmDtm'] >= 0]
    XMatrix = trainData.loc[:, trainData.columns != 'modisLst'].values
    yVector = trainData.loc[:, trainData.columns == 'modisLst'].values.flatten()

    # Get predicting data
    predictData = np.hstack([amsreLstVector, modisNdviVector, srtmDtmVector])
    predictData = pd.DataFrame(predictData, columns=["amsreLst", "modisNdvi", "srtmDtm"])
    N_selected = len(predictData)
    predictData = predictData[predictData['amsreLst'] >= 0]
    predictData = predictData[predictData['modisNdvi'] >= -1]
    predictData = predictData[predictData['srtmDtm'] >= 0]
    indexArray = predictData.index

    # Random Forest Regressor
    rfRegModel = sklearn.ensemble.RandomForestRegressor(n_estimators=100, oob_score=True)
    rfRegModel.fit(XMatrix, yVector)
    pedictResult = rfRegModel.predict(predictData)

    # Export predicted result
    outputGeotrans = tuple([outputXMax, 0.01, 0.0, outputYMin, 0.0, -0.01])
    outputColN = int(modisLstColN - xMinClipColNVector[0] - xMaxClipColNVector[0])
    outputRowN = int(modisLstRowN - yMaxClipRowNVector[0] - yMinClipRowNVector[0])

    modisLstProj = modisLstObj.GetProjection()
    modisLstBandN = modisLstObj.RasterCount

    predictArray = np.zeros((N_selected, 1), dtype=np.float64) * np.nan
    predictArray[indexArray] = pedictResult.reshape(len(pedictResult), 1)
    predictArray = predictArray.reshape(outputRowN, outputColN)
    predictArray[np.bitwise_not(predictArray > 0)] = np.nanmean(predictArray)

    writeTiff(predictArray, outputColN, outputRowN, modisLstBandN, outputGeotrans, modisLstProj,
              outputRfLstPath)
    writeTiff(modisLstArray2, outputColN, outputRowN, modisLstBandN, outputGeotrans, modisLstProj,
              outputRefLstPath)
    writeTiff(modisLstTimeArray, outputColN, outputRowN, modisLstBandN, outputGeotrans,
              modisLstProj, outputLstQcPath)
    if flg4 == 1:
        predictSmoothArray = gaussian_filter(predictArray, gsFilterSigma)
        writeTiff(predictSmoothArray, outputColN, outputRowN, modisLstBandN, outputGeotrans,
                  modisLstProj, outputRfSmoothLstPath)

        fusionArray = modisLstArray2
        fusionArray[modisLstArray2 <= 0] = predictSmoothArray[modisLstArray2 <= 0]
        writeTiff(fusionArray, outputColN, outputRowN, modisLstBandN, outputGeotrans, modisLstProj,
                  outputFusionLstPath)
    else:
        fusionArray = modisLstArray2
        fusionArray[modisLstArray2 <= 0] = predictArray[modisLstArray2 <= 0]
        writeTiff(fusionArray, outputColN, outputRowN, modisLstBandN, outputGeotrans, modisLstProj,
                  outputFusionLstPath)
