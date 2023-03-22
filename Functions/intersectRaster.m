function [raster1Layer,raster2Layer] = intersectRaster(raster1Path, raster2Path)
%INTERSECTRASTER 此处显示有关此函数的摘要
%   此处显示详细说明
[raster1Layer, raster1Ref] = readgeoraster(raster1Path);
[raster2Layer, raster2Ref] = readgeoraster(raster2Path);
cellsize = raster2Ref.CellExtentInLongitude;

raster1LonMin = raster1Ref.LongitudeLimits(1);
raster1LatMin = raster1Ref.LatitudeLimits(1);
raster1LonMax = raster1Ref.LongitudeLimits(2);
raster1LatMax = raster1Ref.LatitudeLimits(2);

raster2LonMin = raster2Ref.LongitudeLimits(1);
raster2LatMin = raster2Ref.LatitudeLimits(1);
raster2LonMax = raster2Ref.LongitudeLimits(2);
raster2LatMax = raster2Ref.LatitudeLimits(2);

lonMin = max(raster1LonMin, raster2LonMin);
latMax = min(raster1LatMax, raster2LatMax);

raster1StartCol = round((lonMin - raster1LonMin) / cellsize + 1);
raster1EndCol = round((raster1LonMax - lonMin) / cellsize);
raster1StartRow = round((raster1LatMax - latMax) / cellsize + 1);
raster1EndRow = round((latMax - raster1LatMin) / cellsize);

raster2StartCol = round((lonMin - raster2LonMin) / cellsize + 1);
raster2EndCol = round((raster2LonMax - lonMin) / cellsize);
raster2StartRow = round((raster2LatMax - latMax) / cellsize + 1);
raster2EndRow = round((latMax - raster2LatMin) / cellsize);

raster1Layer = raster1Layer(raster1StartRow: raster1EndRow, raster1StartCol: raster1EndCol);
raster2Layer = raster2Layer(raster2StartRow: raster2EndRow, raster2StartCol: raster2EndCol);

end

