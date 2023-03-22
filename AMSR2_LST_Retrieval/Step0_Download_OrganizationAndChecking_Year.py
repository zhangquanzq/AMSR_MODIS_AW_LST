# coding=utf-8
# 组织和检查按年份存储的数据, 涉及MODIS地表覆盖类型和积雪覆盖数据.
# 按数据类型, 年份分文件夹组织下载的MODIS数据, 并检查数据是否损坏, 缺失, 重复.
import arcpy
import numpy as np
import os
from os import path
from glob import glob

# 指定MODIS数据类型的标记. 1表示MCD12Q1, 2表示MYD10C1, 3表示MYD13A2.
flg1 = 3

# 年份列表.
yearList = range(2022, 2023)
yearRange = '{0}-{1}'.format(yearList[0], yearList[-1])

# MODIS数据类型和区域.
modisType = ['MCD12Q1', 'MYD10C1', 'MYD13A2'][flg1 - 1]
region = 'CN'

# 路径.
rootPath = r'H:\AMSR_LST_IceSheet'
downloadPath = path.join(rootPath, 'Download')

urlTxtFolderPath = path.join(rootPath, 'URL', region)
urlTxtPath = path.join(urlTxtFolderPath, '{0}_{1}{2}.txt'.format(modisType, yearRange, region))
urlRestTxtPath = path.join(urlTxtFolderPath, 'Rest_{0}_{1}{2}.txt'.format(modisType, yearRange, region))

modisFolderPath = path.join(rootPath, 'Data', '{0}_1_Tile{1}_HDF'.format(modisType, region))
if not path.exists(modisFolderPath):
    os.makedirs(modisFolderPath)

for yearNum in yearList:
    modisYearFolderPath = path.join(modisFolderPath, '{0}_{1}XXX_HDF'.format(modisType, yearNum))
    if not path.exists(modisYearFolderPath):
        os.makedirs(modisYearFolderPath)

    # 将下载的MODIS数据划分为有效和错误HDF文件列表, 并获取MODIS数据不重复日期字符串列表.
    print(u'检查下载的{0}年{1} {2}数据的有效性, 并删除无效数据.'.format(yearNum, region, modisType))
    modisValidPathList, validYearList = [], []
    for modisPath in glob(path.join(downloadPath, '{0}.A{1}*.hdf'.format(modisType, yearNum))):
        try:
            arcpy.GetRasterProperties_management(modisPath, 'CELLSIZEY')
            print(u'有效HDF文件：{0}'.format(modisPath))
            modisValidPathList.append(modisPath)
            modisName = path.basename(modisPath)
            modisYear = modisName.split('.')[1][1:5]
            if modisYear == str(yearNum):
                if modisYear not in validYearList:
                    validYearList.append(modisYear)
        except:
            print(u'错误HDF文件：{0}'.format(modisPath))
            os.remove(modisPath)

    # 将下载的MODIS数据移动到对应年份的文件夹中.
    print(u'移动{0}年{1}有效的MODIS数据.'.format(yearNum, region))
    for validYear in validYearList:
        # 将同一年份的MODIS数据移动到该年份的文件夹.
        for modisPath in modisValidPathList:
            modisName = path.basename(modisPath)
            modisYear = modisName.split('.')[1][1:5]
            if modisYear == validYear:
                modisNewPath = path.join(modisYearFolderPath, modisName)
                if path.exists(modisNewPath):
                    os.remove(modisNewPath)
                os.rename(modisPath, modisNewPath)
                print(u'移动HDF文件：{0}'.format(modisPath))

    print(u'获取{0}年没有下载或无效的{1} MODIS数据链接.'.format(yearNum, region))
    # 读取移动后的MODIS数据文件名列表.
    modisPathList = glob(path.join(modisYearFolderPath, '{0}*.hdf'.format(modisType)))
    modisNewNameList = [path.basename(modisPath) for modisPath in modisPathList]
    modisNewNameListN = len(modisNewNameList)

    # 获取指定年份下载链接文件中所有MODIS HDF文件列表.
    with open(urlTxtPath, 'r') as f1:
        modisUrlPathList = np.unique(f1.readlines())
    modisUrlNameList = []
    for modisUrl in modisUrlPathList:
        modisName = path.basename(modisUrl.strip())
        modisYear = modisName.split('.')[1][1:5]
        if modisYear == str(yearNum):
            modisUrlNameList.append(modisName)
    modisUrlNameListN = len(modisUrlNameList)

    # 已下载数据和URL个数.
    print(u'{0}年, 应下载文件个数: {1}, 已下载文件个数: {2}.\n'.format(yearNum, modisUrlNameListN, modisNewNameListN))

    # 将没有下载到或不能正常打开的MODIS HDF数据URL导出为txt文档.
    restModisUrlList = []
    for i in range(modisUrlNameListN):
        if modisUrlNameList[i] not in modisNewNameList:
            restModisUrlList.append(modisUrlPathList[i])
    if path.exists(urlRestTxtPath):
        os.remove(urlRestTxtPath)
    if len(restModisUrlList) > 0:
        with open(urlRestTxtPath, 'w') as f2:
            f2.writelines(restModisUrlList)

    # 显示当前年份URL文件里不存在的MODIS数据文件.
    modisExtraNameList = list(set(modisNewNameList) - set(modisUrlNameList))
    if len(modisExtraNameList) > 0:
        print(modisExtraNameList)
