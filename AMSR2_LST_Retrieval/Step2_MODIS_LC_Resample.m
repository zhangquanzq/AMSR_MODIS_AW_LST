%% 将MODIS LC升尺度到AMSR2 BT数据的0.1度分辨率.

%% 路径.
% 数据根目录.
rootDir = 'E:\AMSR2_MODIS_AW_LST\';
dataDir = fullfile(rootDir, 'AMSR2_LST_Retrieval\Data');
addpath(fullfile(rootDir, 'Code\Functions'));

% MODIS LC路径.
modisLcDir = fullfile(dataDir, 'MCD12Q1_2_MosaicCN_TIF');

% 中国范围栅格路径.
extentCnPath = fullfile(dataDir, 'Zones', 'ExtentCN_0d1.tif');

%% 升尺度MODIS LC.
% 17个分类.
modisLcGcsList = dir(fullfile(modisLcDir, 'MCD12Q1*gcs.tif'));
modisLcGcsList = {modisLcGcsList.name}';
for i = 1: length(modisLcGcsList)
    modisLcName = modisLcGcsList{i};
    modisLcUpscaledName = replace(modisLcName, 'gcs.tif', 'gcs_upscaled.tif');
    modisLcUpscaledPath = fullfile(modisLcDir, modisLcUpscaledName); 
    if exist(modisLcUpscaledPath, 'file')
        continue
    end

    fprintf('升尺度MODIS LC: %s\n', modisLcName)
    modisLcFilePath = fullfile(modisLcDir, modisLcName);
    upscalingRaster(modisLcFilePath, modisLcUpscaledPath, extentCnPath)
end

% 5个分类.
modisLcRcList = dir(fullfile(modisLcDir, 'MCD12Q1*gcs_reclsfy.tif'));
modisLcRcList = {modisLcRcList.name}';
for i = 1: length(modisLcRcList)
    modisLcName = modisLcRcList{i};
    modisLcUpscaledName = replace(modisLcName, 'reclsfy.tif', 'reclsfy_upscaled.tif');
    modisLcUpscaledPath = fullfile(modisLcDir, modisLcUpscaledName); 
    if exist(modisLcUpscaledPath, 'file')
        continue
    end

    fprintf('升尺度MODIS LC: %s\n', modisLcName)
    modisLcFilePath = fullfile(modisLcDir, modisLcName);
    upscalingRaster(modisLcFilePath, modisLcUpscaledPath, extentCnPath)
end

% 为湖泊和冰川创建缓冲区.
waterLcCode = [3, 4]; bufferWidth = 2;
modisLcRcUpList = dir(fullfile(modisLcDir, 'MCD12Q1*reclsfy_upscaled.tif'));
modisLcRcUpList = {modisLcRcUpList.name}';
for i = 1: length(modisLcRcUpList)
    modisLcName = modisLcRcUpList{i};
    modisLcBufferName = replace(modisLcName, 'upscaled.tif', 'upscaled_waterBuffer.tif');
    modisLcBufferPath = fullfile(modisLcDir, modisLcBufferName);
    if exist(modisLcBufferPath, 'file')
        continue
    end

    fprintf('为MODIS LC创建缓冲区: %s\n', modisLcName)
    modisLcFilePath = fullfile(modisLcDir, modisLcName);
    bufferRaster(modisLcFilePath, modisLcBufferPath, waterLcCode, bufferWidth)
end

%% 自定义函数.
% 升尺度MODIS LC数据.
function upscalingRaster(rasterPath, upscaledPath, extentPath)
% 数据图层与属性.
[rasterLayer, rasterRef] = readgeoraster(rasterPath);
rasterRowN = rasterRef.RasterSize(1); rasterColN = rasterRef.RasterSize(2);

% 升尺度范围和像元大小属性.
[extentLayer, extentRef] = readgeoraster(extentPath);
extentRowN = extentRef.RasterSize(1); extentColN = extentRef.RasterSize(2);

% 升尺度过程.
[lcBlockBdy, lcBlockSize] = getStartBlockRowCol(rasterRef, extentRef);
upscaledLayer = zeros(extentRowN, extentColN, 'uint8');
for ii = 1 : extentRowN
    for jj = 1 : extentColN
        % 跳过非研究区像元.
        if extentLayer(ii, jj) ~= 1
            continue
        end
        % 定位每一个LC滑动窗口.
        blockTopRow = lcBlockBdy(1) + lcBlockSize(1) * (ii - 1);
        blockBottomRow = lcBlockBdy(2) + lcBlockSize(1) * (ii - 1);
        blockLeftCol = lcBlockBdy(3) + lcBlockSize(2) * (jj - 1);
        blockRightCol = lcBlockBdy(4) + lcBlockSize(2) * (jj - 1);
        if blockRightCol > rasterColN || blockBottomRow > rasterRowN
            continue
        end
        % 窗口计算.
        lcBlock = rasterLayer(blockTopRow: blockBottomRow, blockLeftCol: blockRightCol);
        upscaledLayer(ii, jj) = mode(lcBlock, 'all');
    end
end

% 保存升尺度后数据.
geotiffwrite(upscaledPath, upscaledLayer, extentRef, TiffTags=struct('Compression', 'LZW'));
end


% 为栅格数据中的特定值创建缓冲区.
function bufferRaster(rasterPath, bufferPath, bufferValues, bufferWidth)
[rasterLayer, rasterRef] = readgeoraster(rasterPath);
rasterRowN = rasterRef.RasterSize(1); rasterColN = rasterRef.RasterSize(2);
bufferLayer = zeros(rasterRowN, rasterColN, 'uint8');
for ii = 1: rasterRowN
    for jj = 1: rasterColN
        rasterPixel = rasterLayer(ii, jj);
        if ismember(rasterPixel, bufferValues)
            for m = -1*bufferWidth: 1*bufferWidth
                for n = -1*bufferWidth: 1*bufferWidth
                    xLocal = ii + m; yLocal = jj + n;
                    if xLocal >= 1 || yLocal >= 1
                        bufferLayer(xLocal, yLocal) = rasterPixel;
                    end
                end
            end
        end
    end
end
geotiffwrite(bufferPath, bufferLayer, rasterRef, TiffTags=struct('Compression', 'LZW'));
end
