function resamplePixelSeries(rasterBufferPath, regionExtentPath, resampleDir, cellsizeList, ...
    region, downscale, snapRasterPath, nodata)
% 限定downscale参数的取值范围.
if ~ismember(downscale, ["initial", "previous"])
    error('downscale参数的取值必须为字符串 "initial" 或 "previous" 之一.')
end

% 按分辨率序列重采样.
[rasterBufferDir, name, ext] = fileparts(rasterBufferPath);
rasterBufferName = [name, ext];
rasterRegionName = replace(rasterBufferName, 'Buffer', region);
rasterTempPathList = strings(length(cellsizeList) - 1, 1);
for i = 1: length(cellsizeList) - 1
    cellsizeNow = cellsizeList{i};
    cellsizeNext = cellsizeList{i + 1};
    
    rasterBufferNext = replace(rasterBufferName, '0.01', cellsizeNext);
    rasterBufferResamplePath = fullfile(rasterBufferDir, rasterBufferNext);

    rasterNext = replace(rasterRegionName, '0.01', cellsizeNext);
    rasterResamplePath = fullfile(resampleDir, rasterNext);

    % 两种重采样方式. 1, 上一步重采样结果作为下一步输入数据, 2, 原始MODIS数据生成所有分辨率序列中的数据.
    if strcmp(downscale, 'initial')
        rasterPath = rasterBufferPath;
    elseif strcmp(downscale, 'previous')
        if cellsizeNow == cellsizeList(1)
            rasterPath = rasterBufferPath;
        else
            rasterPath = replace(rasterBufferPath, '0.01', cellsizeNow);
        end
    end
    resampleRaster(rasterPath, rasterBufferResamplePath, cellsizeNext, snapRasterPath, nodata)
    clipRaster(rasterBufferResamplePath, regionExtentPath, rasterResamplePath)
    rasterTempPathList(i) = rasterBufferResamplePath;
end

% 删除Buffer过程数据.
for i = 1: length(cellsizeList) - 1
    delete(rasterTempPathList(i))
end
end
