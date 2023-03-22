
dataDir = 'J:\AMSR_MODIS_AW_LST\AMSR_LST_Retrieval\Data\AMSRE_4_LSTCN_TIF\AMSRE_2011XXXX';
dataList = {dir(fullfile(dataDir, 'AMSRE*.tif')).name}';
for i = 1: length(dataList)
    fileName = dataList{i}; newFile = replace(dataList{i}, '_merged_', '_');
    dataPath = fullfile(dataDir, fileName);
    newFilePath = fullfile(dataDir, newFile);
    movefile(dataPath, newFilePath)
end
