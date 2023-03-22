
% 控制昼夜的标记. 1表示白天, 2表示晚上.
flg1 = 1;

dayNight = {'Day', 'Night'};
dayNight = dayNight{flg1};


dataPath = 'I:\AMSR_LST_IceSheet\Data\MYD11A1_2_MosaicAntarctic_TIF\MYD11A1_2013XXX_TIF';
modisLstPathList = dir(fullfile(dataPath, sprintf('MYD11A1*LST_%s.tif', dayNight)));
modisLstPathList = {modisLstPathList(1:8).name}';
modisLstInfo = geotiffinfo(fullfile(dataPath, modisLstPathList{1}));

rowN = modisLstInfo.Height;
colN = modisLstInfo.Width;
ref = modisLstInfo.SpatialRef;

modisLstArray = zeros(rowN, colN, 8, 'single');
for i = 1: 8
    modisLstArray(:, :, i) = readgeoraster(fullfile(dataPath, modisLstPathList(i)));
end
modisLstArray(modisLstArray==0) = nan;
meanLstLayer = mean(modisLstArray, 3, "omitnan");
geotiffwrite('Antarctic2.tif', meanLstLayer, ref, 'GeoKeyDirectoryTag', modisLstInfo.GeoTIFFTags.GeoKeyDirectoryTag)


