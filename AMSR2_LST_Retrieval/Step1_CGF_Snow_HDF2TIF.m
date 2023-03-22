%% 将CGF积雪范围数据由HDF格式转为TIF格式.

%% 预设参数.
yearList = 2012: 2020;
fillValue = 5;

%% 路径.
rootDir = 'E:\AMSR2_MODIS_AW_LST';
dataDir = fullfile(rootDir, 'AMSR2_LST_Retrieval\Data');
cgfSnowHdfDir = fullfile(dataDir, 'CGF_Snow_1_CN_HDF');
cgfSnowTifDir = fullfile(dataDir, 'CGF_Snow_2_CN_TIF');
if ~exist(cgfSnowTifDir, 'dir')
    mkdir(cgfSnowTifDir)
end

% 参考范围栅格数据属性.
refInfo = georasterinfo(fullfile(dataDir, 'Zones\ExtentCN_0d1.tif')).RasterReference;
refLatLim = refInfo.LatitudeLimits; refLonLim = refInfo.LongitudeLimits;

%% 分年份批处理.
for i = 1: length(yearList)
    yearNum = yearList(i);

    % 创建输出文件的年文件夹.
    cgfSnowTifYearDir = fullfile(cgfSnowTifDir, num2str(yearNum));
    if ~exist(cgfSnowTifYearDir, 'dir')
        mkdir(cgfSnowTifYearDir)
    end

    % 读取每年所有的积雪HDF格式数据.
    cgfSnowHdfYearDir = fullfile(cgfSnowHdfDir, num2str(yearNum));
    snowHdfNameList = dir(fullfile(cgfSnowHdfYearDir, 'NIEER*.hdf'));
    snowHdfNameList = {snowHdfNameList.name}';
    for j = 1: length(snowHdfNameList)
        snowTifName = replace(snowHdfNameList{j}, '.hdf', '.tif');
        snowTifPath = fullfile(cgfSnowTifYearDir, snowTifName);
        if exist(snowTifPath, 'file')
            continue
        end

        fprintf('输出: %s\n', snowTifName)
        snowHdfPath = fullfile(cgfSnowHdfYearDir, snowHdfNameList{j});
        snowInfo = hdfinfo(snowHdfPath);

        snowAttNameList = {snowInfo.Attributes.Name}';
        snowAttValueList = {snowInfo.Attributes.Value}';
        snowLatMin = double(snowAttValueList{strcmp(snowAttNameList, 'Latitude_Min')});
        snowLatMax = double(snowAttValueList{strcmp(snowAttNameList, 'Latitude_Max')});
        sonwLonMin = double(snowAttValueList{strcmp(snowAttNameList, 'Longitude_Min')});
        snowLonMax = double(snowAttValueList{strcmp(snowAttNameList, 'Longitude_Max')});
        cellsizeLat = round(snowAttValueList{strcmp(snowAttNameList, 'Latitude_Resolution')}, 3);
        cellsizeLon = round(snowAttValueList{strcmp(snowAttNameList, 'Longitude_Resolution')}, 3);

        skipLeftColN = double((sonwLonMin - refLonLim(1)) / cellsizeLon);
        skipRightColN = double((snowLonMax - refLonLim(2)) / cellsizeLon);
        skipTopRowN = double((snowLatMax - refLatLim(2)) / cellsizeLat);
        skipBottomRowN = double((snowLatMin - refLatLim(1)) / cellsizeLat);

        snowLayer = hdfread(snowHdfPath, snowInfo.SDS.Name);  % 'Day_Snow_Cover_Extent'
        snowLayer(snowLayer == 255) = fillValue;
        if skipLeftColN > 0
            snowLayer = padarray(snowLayer, [0, skipLeftColN], fillValue, 'pre');
        else
            snowLayer = snowLayer(:, 1 + skipLeftColN: end);
        end
        if skipRightColN > 0
            snowLayer = snowLayer(:, 1: end - skipRightColN);
        else
            snowLayer = padarray(snowLayer, [0, skipLeftColN], fillValue, 'post');
        end
        if skipTopRowN > 0
            snowLayer = snowLayer(skipTopRowN + 1: end, :);
        else
            snowLayer = padarray(snowLayer, [skipTopRowN, 0], fillValue, 'pre');
        end
        if skipBottomRowN > 0
            snowLayer = padarray(snowLayer, [skipBottomRowN, 0], fillValue, 'post');
        else
            snowLayer = snowLayer(1: end + skipBottomRowN, :);
        end

        cgfSnowRef = georefcells(refLatLim, refLonLim, size(snowLayer), ColumnsStartFrom='north');
        geotiffwrite(snowTifPath, snowLayer, cgfSnowRef, CoordRefSysCode=4326, ...
            TiffTags=struct('Compression','LZW'))
    end
end
