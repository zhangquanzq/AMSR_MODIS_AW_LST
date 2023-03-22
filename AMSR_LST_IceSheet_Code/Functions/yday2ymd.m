function ymd = yday2ymd(yday)
% Convert YearDayofyear to YearMonthDay. Both input and output are strings.
% Format of input: 'yyyyday'. For example: '2010123'
% Format of output: 'yyyymmdd'. For example: '20100503'
year = yday(1:4);
day = str2double(yday(5:7));
monthday = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
if leapyear(str2double(year))
    monthday(2) = 29;
end
i = 1;
while day > 0
    day = day - monthday(i);
    i = i + 1;
end
day = day + monthday(i-1);
ymd = [year, num2str(i-1, '%02d'), num2str(day, '%02d')];

end