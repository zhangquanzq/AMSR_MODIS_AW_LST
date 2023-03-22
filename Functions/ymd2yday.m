function yday = ymd2yday(ymd)
% 将YearMonthDay转换为YearDayofyear. 输入数据参数均为字符串.
% 输入参数格式: 'yyyymmdd'. 例如: '20100503'.
% 输出参数格式: 'yyyyday'. 例如: '2010123'.
year = ymd(1:4);
month = str2double(ymd(5:6));
day = str2double(ymd(7:8));

monthday = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
if leapyear(str2double(year))
    monthday(2) = 29;
end

monthsum = sum(monthday(1:month)) - monthday(month);
dday = monthsum + day;

yday = [year, num2str(dday, '%03d')];

end