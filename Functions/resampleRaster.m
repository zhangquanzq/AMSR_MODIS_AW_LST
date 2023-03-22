function resampleRaster(rasterPath, resampleRasterPath, cellsize, snapRasterPath, nodata)
%RESAMPLERASTER 栅格数据重采样.

% 捕捉栅格属性.
snapRasterRef = georasterinfo(snapRasterPath).RasterReference;
snapCellsizeLat = snapRasterRef.CellExtentInLatitude;
snapCellsizeLon = snapRasterRef.CellExtentInLongitude;
snapLatMin = snapRasterRef.LatitudeLimits(1);
snapLonMin = snapRasterRef.LongitudeLimits(1);

% 待重采样栅格属性.
[rasterLayer, rasterRef] = readgeoraster(rasterPath);
%rasterNodata = georasterinfo(rasterPath).MissingDataIndicator;
%if ~isempty(rasterNodata)
%    rasterLayer(rasterLayer == rasterNodata) = 0;
%end
rasterLayer = double(rasterLayer);
rasterLayer(rasterLayer == nodata) = nan;

cellsizeLat = rasterRef.CellExtentInLatitude;
cellsizeLon = rasterRef.CellExtentInLongitude;
rasterLatMin = rasterRef.LatitudeLimits(1);
rasterLatMax = rasterRef.LatitudeLimits(2);
rasterLonMin = rasterRef.LongitudeLimits(1);
rasterLonMax = rasterRef.LongitudeLimits(2);

cellLatVector1 = (rasterLatMax - cellsizeLat/2): -cellsizeLat: (rasterLatMin + cellsizeLat/2);
cellLonVector1 = (rasterLonMin + cellsizeLon/2): cellsizeLon: (rasterLonMax - cellsizeLon/2);
[lonLayer1, latLayer1] = meshgrid(cellLonVector1, cellLatVector1);

% 确定捕捉后左下角像元的起始位置.
if snapLatMin > rasterLatMin
    snapCellsizeLat = -snapCellsizeLat;
end
snapLatVector = snapLatMin: snapCellsizeLat: rasterLatMin;
if abs(snapLatVector(end) - rasterLatMin) < abs(snapCellsizeLat) / 2
    shiftLatMin = snapLatVector(end);
else
    shiftLatMin = snapLatVector(end) + snapCellsizeLat;
end

if snapLonMin > rasterLonMin
    snapCellsizeLon = -snapCellsizeLon;
end
snapLonVector = snapLonMin: snapCellsizeLat: rasterLonMin;
if abs(snapLonVector(end) - rasterLonMin) < abs(snapCellsizeLon) / 2
    shiftLonMin = snapLonVector(end);
else
    shiftLonMin = snapLonVector(end) + snapCellsizeLat;
end

% 插值重采样后的栅格像元.
cellsize = str2double(cellsize);
cellLatVector2 = shiftLatMin + cellsize/2: cellsize: rasterLatMax;
cellLonVector2 = shiftLonMin + cellsize/2: cellsize: rasterLonMax;
[lonLayer2, latLayer2] = meshgrid(cellLonVector2, fliplr(cellLatVector2));
resampleLayer = interp2(lonLayer1, latLayer1, double(rasterLayer), lonLayer2, latLayer2, 'linear');

% 输出重采样结果.
resampleLatlim = [shiftLatMin, cellLatVector2(end) + cellsize/2];
resampleLonlim = [shiftLonMin, cellLonVector2(end) + cellsize/2];
resampleRef = georefcells(resampleLatlim, resampleLonlim, size(resampleLayer),...
    ColumnsStartFrom=north);
geotiffwrite(resampleRasterPath, resampleLayer, resampleRef, CoordRefSysCode=4326);

end

