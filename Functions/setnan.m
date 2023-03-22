function array2 = setnan(array1,nanarray)
%SETNAN 将矩阵特定位置的值设为nan, 并赋值给新矩阵.
%  现有Matlab函数 直接处理原矩阵, 而非赋值给新矩阵.
array2 = array1;
array2(nanarray) = nan;
end