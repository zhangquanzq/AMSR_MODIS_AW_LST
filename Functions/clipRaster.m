function clipRaster(rasterPath, extentPath, clipRasterPath)
%CLIPRASTER 裁剪栅格数据.
%   使用范围
[rasterLayer, rasterRef] = readgeoraster(rasterPath);
extentBox = shapeinfo(extentPath).BoundingBox;

rasterLonMin = round(rasterRef.LongitudeLimits(1), 10);
rasterLatMax = round(rasterRef.LatitudeLimits(2), 10);
rasterCellsizeLon = rasterRef.CellExtentInLongitude;
rasterCellsizeLat = rasterRef.CellExtentInLatitude;

extentLonMin = round(extentBox(1), 10); extentLonMax = round(extentBox(2), 10);
extentLatMin = round(extentBox(3), 10); extentLatMax = round(extentBox(4), 10);

startRow = floor((rasterLatMax - extentLatMax) / rasterCellsizeLat) + 1;
endRow = ceil((rasterLatMax - extentLatMin) / rasterCellsizeLat);
startCol = floor((extentLonMin - rasterLonMin) / rasterCellsizeLon) + 1;
endCol = ceil((extentLonMax - rasterLonMin) / rasterCellsizeLon);
clipLayer = rasterLayer(startRow: endRow, startCol: endCol);

clipLatMax = rasterLatMax - rasterCellsizeLat * (startRow - 1);
clipLatMin = rasterLatMax - rasterCellsizeLat * endRow;
clipLonMin = rasterLonMin + rasterCellsizeLon * (startCol - 1);
clipLonMax = rasterLonMin + rasterCellsizeLon * endCol;
clipLatlim = [clipLatMin, clipLatMax];
clipLonlim = [clipLonMin, clipLonMax];

clipRef = georefcells(clipLatlim, clipLonlim, size(clipLayer), 'ColumnsStartFrom','north');
geotiffwrite(clipRasterPath, clipLayer, clipRef, 'CoordRefSysCode', 4326);

end

