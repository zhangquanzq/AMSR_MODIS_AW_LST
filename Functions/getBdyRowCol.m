function [startRow, endRow, startCol, endCol] = getBdyRowCol(clipRef, baseRef)
% 获取待裁剪影像中与参考影像边界对齐的起始和结束行列号. 输入参数为两个影像的空间参考对象.

clipCST = clipRef.CoordinateSystemType; baseCST = baseRef.CoordinateSystemType;
if strcmp(clipCST, 'geographic') && strcmp(baseCST, 'geographic')
    clipCellsizeX = clipRef.CellExtentInLongitude; clipCellsizeY = clipRef.CellExtentInLatitude;

    clipXMin = clipRef.LongitudeLimits(1); clipXMax = clipRef.LongitudeLimits(2);
    clipYMin = clipRef.LatitudeLimits(1); clipYMax = clipRef.LatitudeLimits(2);

    baseXMin = baseRef.LongitudeLimits(1); baseXMax = baseRef.LongitudeLimits(2);
    baseYMax = baseRef.LatitudeLimits(2); baseYMin = baseRef.LatitudeLimits(1);
elseif strcmp(clipCST, 'planar') && strcmp(baseCST, 'planar')
    clipCellsizeX = clipRef.CellExtentInWorldX; clipCellsizeY = clipRef.CellExtentInWorldY;

    clipXMin = clipRef.XWorldLimits(1); clipXMax = clipRef.XWorldLimits(2);
    clipYMin = clipRef.YWorldLimits(1); clipYMax = clipRef.YWorldLimits(2);

    baseXMin = baseRef.XWorldLimits(1); baseXMax = baseRef.XWorldLimits(2);
    baseYMax = baseRef.YWorldLimits(2); baseYMin = baseRef.YWorldLimits(1);
else
   error('两个输入数据的坐标系统类型不一致, 请检查.')
end

clipRowN = clipRef.RasterSize(1); clipColN = clipRef.RasterSize(2);
if clipYMax > baseYMax
    startRow = 1 + ceil((clipYMax - baseYMax) / clipCellsizeY);
else
    startRow = 1;
end

if clipYMin < baseYMin
    endRow = clipRowN - ceil((baseYMin - clipYMin) / clipCellsizeY);
else
    endRow = clipRowN;
end

if baseXMin > clipXMin
    startCol = 1 + ceil((baseXMin - clipXMin) / clipCellsizeX);
else
    startCol = 1;
end

if baseXMax < clipXMax
    endCol = clipColN - ceil((clipXMax - baseXMax) / clipCellsizeX);
else
    endCol = clipColN;
end

end