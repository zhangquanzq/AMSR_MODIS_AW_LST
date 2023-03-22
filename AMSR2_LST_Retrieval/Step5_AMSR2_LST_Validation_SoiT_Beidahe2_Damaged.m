%% ä½¿ç”¨åŒ—å¤§æ²³æµåŸŸç«™ç‚¹è§‚æµ‹çš„æ¸©åº¦æ•°æ®éªŒè¯AMSR2åœ°è¡¨æ¸©åº¦.
% æ­¤ç¨‹åºå°†åŒä¸€ç«™ç‚¹çš„æ˜¼å¤œæ•°æ®æ”¾åˆ°åŒä¸€å¹…æ•£ç‚¹å›¾ä¸Š.

%% åŠŸèƒ½æ ‡è®°å’Œé¢„è®¾å‚æ•°.
% æŒ‡å®šç«™ç‚¹ç‰‡åŒºçš„æ ‡è®°. 1è¡¨ç¤ºçŽ‰é—¨ä¸œ, 2è¡¨ç¤ºåŠè¾¾å‚æ²Ÿ, 3è¡¨ç¤ºä¸ƒä¸€å†°å·.
flg1 = 1;

% ç«™ç‚¹ç‰‡åŒºåç§°.
siteGroup = {'Yumendong', 'Diaodabangou', 'QiyiGlacier'};
siteGroup = siteGroup{flg1};

% æ˜¼å¤œ, è¿‡å¢ƒæ—¶é—´.
daynightTypes = {'Day', 'Night'};
daynightTypesN = length(daynightTypes);
transitTypes = {'1330', '0130'};

%% è·¯å¾„.
% æ ¹ç›®å½•.
rootDir = 'I:\AMSR2_MODIS_AW_LST';
addpath(fullfile(rootDir, 'Code/Functions/'))
retrievalDir = fullfile(rootDir, 'AMSR2_LST_Retrieval');
dataPath = fullfile(retrievalDir, 'Data');
figPath = fullfile(retrievalDir, 'Figures');

% è¾“å…¥æ•°æ®è·¯å¾„.
featureDir = fullfile(dataPath, 'Feature');
siteTrDir = fullfile(dataPath, 'SiteData\Beidahe\SoilTRecords');
amsr2LstDir = fullfile(dataPath, 'AMSR2_4_LSTCN_TIF');
modisUpscaleDir = fullfile(dataPath, 'MYD11A1_3_UpscalingCn_TIF');

% è¾“å‡ºæ•°æ®è·¯å¾„.
siteMatDir = fullfile(dataPath, 'SiteSoilT_Matlab');
if ~exist(siteMatDir, 'dir')
    mkdir(siteMatDir)
end
siteFigureDir = fullfile(figPath, 'SiteSoilT_DaynightCombine');
if ~exist(siteFigureDir, 'dir')
    mkdir(siteFigureDir)
end

%% æ•´ç†ç«™ç‚¹ç‰‡åŒºçš„åœŸå£¤æ¸©åº¦è§‚æµ‹æ•°æ®, å¹¶ä¿å­˜ä¸ºMatæ–‡ä»¶.
% èŽ·å–ç«™ç‚¹å±žæ€§ä¿¡æ¯.
siteStruct = shaperead(fullfile(featureDir, 'Sites_Location_Beidahe.shp'));
siteLocationList = [[siteStruct.X]; [siteStruct.Y]]';
siteNameList = {siteStruct.Name}';

% èŽ·å–ç‰‡åŒºç«™ç‚¹æ–‡ä»¶åç§°åˆ—è¡¨.
siteTrXlsxList = dir(fullfile(siteTrDir, '*.xlsx'));
siteTrXlsxList = {siteTrXlsxList.name}';
siteTrXlsxList = siteTrXlsxList(contains(siteTrXlsxList, siteGroup));
siteTrXlsxN = length(siteTrXlsxList);

% åˆ†æ˜¼å¤œèŽ·å–MODISè¿‡å¢ƒæ—¶åˆ»çš„ç«™ç‚¹è§‚æµ‹æ¸©åº¦, å¹¶å­˜å‚¨ä¸ºMatæ–‡ä»¶.
for i = 1: daynightTypesN
    daynight = daynightTypes{i};

    % åˆ¤æ–­Matæ–‡ä»¶æ˜¯å¦å­˜åœ¨.
    siteMatPath = fullfile(siteMatDir, sprintf('SiteSoilT_%s_%s.mat', siteGroup, daynight));
    if exist(siteMatPath, 'file')
        continue
    end

    % åˆ›å»ºå­˜å‚¨æ•°æ®çš„è¡¨, è¡¨åŒ…æ‹¬4ä¸ªå­—æ®µ, åˆ†åˆ«æ˜¯: ç«™ç‚¹å, ç«™ç‚¹ä½ç½®, æ—¶é—´åˆ—è¡¨, æ¸©åº¦åˆ—è¡¨.
    siteDataCell = cell(siteTrXlsxN, 4);
    for j = 1: siteTrXlsxN
        % èŽ·å–ç«™ç‚¹çš„è§‚æµ‹æ—¶åˆ»ä¸Žæ¸©åº¦è®°å½•.
        siteTrXlsx = siteTrXlsxList{j};
        siteTrTable = readtable(fullfile(siteTrDir, siteTrXlsx));
        siteDatetimeRecords = siteTrTable.Time;
        siteSoilTRecords = siteTrTable.T_C;

        % èŽ·å–æœ‰ç«™ç‚¹è§‚æµ‹çš„æ—¥æœŸçš„MODISè¿‡å¢ƒæ—¶åˆ»åˆ—è¡¨.
        siteDateRecords = datetime(string(siteDatetimeRecords, 'yyyMMdd'), InputFormat='yyyyMMdd');
        modisDatetimeList = strcat(string(unique(siteDateRecords), 'yyyyMMdd'), transitTypes{i});
        modisDatetimeList = datetime(modisDatetimeList, InputFormat='yyyyMMddHHmm');

        % èŽ·å–ä¸ŽMODISè¿‡å¢ƒæ—¶åˆ»æœ€è¿‘çš„ç«™ç‚¹è§‚æµ‹æ—¶é—´å’Œæ¸©åº¦åˆ—è¡¨.
        modisDatetimeN = length(modisDatetimeList);
        siteDatetimeList = strings(modisDatetimeN, 1);
        siteSoilTList = zeros(modisDatetimeN, 1) * nan;
        siteModisDurationIndex = true(modisDatetimeN, 1);
        for k = 1: modisDatetimeN
            datetimeDiff = abs(siteDatetimeRecords - modisDatetimeList(k));
            if min(datetimeDiff) > minutes(2.5)
                siteModisDurationIndex(k) = false;
                continue
            end
            datetimeDiffMinIndex = find(datetimeDiff == min(datetimeDiff), 1);
            siteDatetimeList(k) = siteDatetimeRecords(datetimeDiffMinIndex);
            siteSoilTList(k) = siteSoilTRecords(datetimeDiffMinIndex);
        end
        modisDatetimeList = modisDatetimeList(siteModisDurationIndex);
        siteDatetimeList = siteDatetimeList(siteModisDurationIndex);
        siteDatetimeList = datetime(siteDatetimeList, InputFormat='yyyy-MM-dd HH:mm:ss');
        siteSoilTList = siteSoilTList(siteModisDurationIndex);

        % æ’å€¼MODISè¿‡å¢ƒæ—¶åˆ»çš„ç«™ç‚¹æ¸©åº¦.
        siteModisSoilTList = interp1(siteDatetimeRecords, siteSoilTRecords, modisDatetimeList, ...
            'spline');

        % æ•´ç†ç«™ç‚¹è§‚æµ‹æ•°æ®.
        [~, siteName] = fileparts(siteTrXlsx);
        fprintf('æ•´ç†ç«™ç‚¹%s %sçš„è§‚æµ‹æ•°æ®.\n', siteName, daynight)
        siteDataCell{j, 1} = siteName;
        for k = 1: length(siteNameList)
            if contains(siteName, siteNameList{k})
                siteDataCell{j, 2} = siteLocationList(k, :);
                break
            end
        end
        siteDataCell{j, 3} = modisDatetimeList;
        siteDataCell{j, 4} = siteModisSoilTList;
    end

    % è¾“å‡ºç«™ç‚¹è§‚æµ‹è®°å½•è¡¨åˆ°Matæ–‡ä»¶.
    tableVarStr = sprintf('site%sDataTable', daynight);
    siteDataTable = cell2table(siteDataCell, ...
        VariableNames=["SiteName" "Location" "Datetime" "SoilTemperature"]);
    assignin('base', tableVarStr, siteDataTable)
    save(siteMatPath, tableVarStr)
end

%% æ ¡æ­£ç«™ç‚¹è§‚æµ‹æ•°æ®, å¹¶éªŒè¯åæ¼”çš„AMSR2æ¸©åº¦.
% èŽ·å–AMSR2 LSTå½±åƒæ¯ä¸ªåƒå…ƒçš„ç»çº¬åº¦åæ ‡çŸ©é˜µ, å‡å°ºåº¦åŽçš„MODIS LSTä¸ŽAMSR2 LSTçš„ä¿¡æ¯ä¸€æ ·.
amsr2LstPath = fullfile(amsr2LstDir, 'AMSR2_LST_2012XXXX_TIF', 'AMSR2_LST_Day_20120703.tif');
amsr2Ref = geotiffinfo(amsr2LstPath).SpatialRef;
lonMin = amsr2Ref.LongitudeLimits(1);
lonMax = amsr2Ref.LongitudeLimits(2);
latMin = amsr2Ref.LatitudeLimits(1);
latMax = amsr2Ref.LatitudeLimits(2);
cellsizeX = amsr2Ref.CellExtentInLongitude;
cellsizeY = amsr2Ref.CellExtentInLatitude;
lonVector = lonMin + cellsizeX/2: cellsizeX: lonMax - cellsizeX/2;
latVector = latMax - cellsizeY/2: -cellsizeY: latMin + cellsizeY/2;

% èŽ·å–ç«™ç‚¹ç‰‡åŒºå†…å„ç«™ç‚¹çš„åç§°, ä½ç½®, ä»¥åŠæ˜¼å¤œçš„æ—¶é—´, æ¸©åº¦æ•°æ®åˆ—è¡¨.
siteDayMatPath = fullfile(siteMatDir, sprintf('SiteSoilT_%s_Day.mat', siteGroup));
load(siteDayMatPath, 'siteDayDataTable');
siteNameList = siteDayDataTable.SiteName;
siteLocationList = siteDayDataTable.Location;

[siteDatetimeCell, siteSoilTCell] = deal(cell(daynightTypesN, 1));
for i = 1: daynightTypesN
    daynight = daynightTypes{i};
    siteMatPath = fullfile(siteMatDir, sprintf('SiteSoilT_%s_%s.mat', siteGroup, daynight));
    siteVarStr = sprintf('site%sDataTable', daynight);
    load(siteMatPath, siteVarStr); siteDataTable = eval(siteVarStr);
    siteDatetimeCell{i} = siteDataTable.Datetime;
    siteSoilTCell{i} = siteDataTable.SoilTemperature;
end

% æŒ‰ç«™ç‚¹éªŒè¯AMSR2 LST.
for i = 1: length(siteNameList)
    % ç«™ç‚¹åç§°, ä½ç½®.
    siteName = siteNameList{i}; siteName2 = replace(siteName, '_', ' ');
    siteLocation = siteLocationList(i, :);

    % èŽ·å–ç«™ç‚¹ä½ç½®åœ¨AMSR2 LSTæ•°æ®ä¸Šçš„è¡Œåˆ—å·.
    lonDiffVector = abs(lonVector - siteLocation(1));
    latDiffVector = abs(latVector - siteLocation(2));
    lstCol = find(lonDiffVector == min(lonDiffVector), 1);
    lstRow = find(latDiffVector == min(latDiffVector), 1);

    % åˆ†å¹´åº¦éªŒè¯AMSR2 LST.
    siteYearTypes = unique(siteDatetimeCell{1}{i}.Year);
    for j = 1: length(siteYearTypes)
        siteYear = siteYearTypes(j);

        % åˆ¤æ–­æ˜¯å¦æœ‰ç«™ç‚¹è§‚æµ‹å¹´ä»½çš„AMSR2 LST.
        amsr2LstYearDir = fullfile(amsr2LstDir, sprintf('AMSR2_LST_%dXXXX_TIF', siteYear));
        if ~exist(amsr2LstYearDir, 'dir')
            continue
        end

        % èŽ·å–æ ¡æ­£ç«™ç‚¹æ¸©åº¦çš„ç³»æ•°, ç«™ç‚¹è§‚æµ‹æ¸©åº¦, ä»¥åŠMODIS LST.
        modisLstYearDir = fullfile(modisUpscaleDir, sprintf('MYD11A1_%dXXX_TIF', siteYear));
        [pCell, siteSoilTInYearCell, modisLstCell] = deal(cell(daynightTypesN, 1));
        for k = 1: daynightTypesN
            % èŽ·å–å½“å‰å¹´ä»½ç«™ç‚¹è§‚æµ‹æ—¥æœŸåˆ—è¡¨.
            siteDatetimeList = siteDatetimeCell{k}{i};
            siteYearIndex = (siteDatetimeList.Year == siteYear);
            siteDateInYearList = string(siteDatetimeList(siteYearIndex), 'yyyyMMdd');

            % èŽ·å–å½“å‰å¹´ä»½å‡å°ºåº¦åŽMODIS LSTçš„æ—¥æœŸåˆ—è¡¨.
            modisLstName = sprintf('MYD11A1*_%s.tif', daynII*     þ             ^      ¾                                      =       B    €   C    €   D    Ú   E    Â   S       ¤    ò       ‹  ¤  ­  \   ¶    ÷   ‚  &
  Ó  /  å  -128 xœí–Ýnã FßÄŠ”4NcãÆæb¥\ôÙxôÕÿÇ[Úå«Ç­ÚsæÜ
ÑÒÒÒÒÒÒÒÒÒÒÒÒÒòoåëH6æ¿åÈ¿*øA
WÈÈ÷öuP|Pp-àû=ì&Ð÷{ô½w‰Ôä‡ÙµwlO¡:ÿ|>Ãç¾.]æ1ø:×šßãK	¿„óõ*½êâíüª…€¾7Üûê»?2~]þ /·U|Ð5ü[ææh—Ù•Á™­ø7«à=‚œ­GÜÁ0Ã;çà×YÄk‹µù#T|5ù0·zx
¯¿:üAç./oÿÑž»úRà®<ƒËårñ®øÆã~7¨x4ð‡IÔ€Ð ú¾?A”ƒ,âtªÈW˜í ¬ÜTâ›çÞŽðqŠ„?è#lGLE¾øî¯vŸ`HeÔ|ËÀ=g±øö´Þ _R]°\ôá;V:9ß…uÁwÐ7cNçä|uÎÓuŒ±®ëÆq>«ÅŸ&9ùÈÆqdð2ŽÀíð®¬«×ÿ4yC£‚œÛà]þ„åÜjf,Ý˜¯À…˜¯é82íñ<	ó-2¾„Oã¨,þu~7.¿P‚W
°×që|Ï…§ó'ËG03øÕjêwüxÉsã‹ÎQ •œÌ·Ë-oé%É¡{³g”žÅÿ	ž1]9;ÿƒ±§]þp9¾KÁ#ßls‚áSçsÐJùs×u9/ðÇËèY%]ÿñxøÇ|!~¬E	~öóûû)ßk!{ñçýùl¥øÄþ=äþÅgóÓggó–MOäÇØX
 ¯äLJÀÿ{«ì7³§Ïÿâà6þÿ³Ïà¿Ä7¤ÚñÔüX¾‚ã²oÑç?Tü ¿	¦˜	ù‰ì<¾X ~©ô,¾VÐt×aù•N¿¨–}ç—YÜ ?^È÷–%þ|>)ùé3)?™Îy¿¸ÎùŸpn®úí<ó™œ¿" é ÀÁAÞqr|È˜V‘p^ïèªÖ•˜kÌï
D A­ÀìÐ·ÐX Ö~C |þøCHo¯Í”ó×Ÿ õùË²ö€¾Ãü±í¯×v—ùßð¥ƒ¼V[ÿ mðê©W»ÿIßÕø<OÈÏ<¯{>:äÒ‰ø4|!Žæ£A†_üeòµÆÁ|q4?Eá/@KxœíÖËrâ:à„Eˆæf¨"Ø$uö”ŒàÙòè§ºukI-Ûa ,â.IU¾_-ÙÌ×W—.]ºtéÒ¥K—.]ºty|ÖÏòzýÔëçúëßí¯Ÿë¯·¿þåþ³o}]ºtY.Ÿ:ƒåsýåsýÅSýÅâYþBfùaù:&=Þõ×`ÁféùjPÃ;þCmÎD…°¾\þ÷œÔÏ²ÈÏÏÏÏŸ› ieÙ§Îƒ|þü‹™ˆíü”®}Uâîþ¢!ÊÞª)|||ÀËâµŸm±Î××=Z,Úû­óïxš¦é"¯¾ï×5Øˆ4¬|Ñá6?Ë¾² ­(D¿ôüôf?Ûl65xÚ*«UºZÝˆ›8º³¼zu‹¿©ÏbÊP“€~o*âñ“‰SCê÷õ§ÓÙ,¥³’¶Ñ¾Öïê?KÓD®Ôqý„¾Í¯Y>ºðjnÁ“¹xo*yþ»Ç8·ã6<l¼x9—™à,$O²h‘½ëxä§R3­+Øü|Ž:ãË½ÞêÝöÙUËY4û–MãáãÉxÜƒŸÙ½ô–:ËûKc<_V¸\àž/²E;;´ú:Þ_ÿÅIÞÞŸN|Ýø.kUÈ7›ívÛÈó{Ÿ0¼Ò{½Ýn÷þ¾«ÓEƒ<ÏkÀÕÇêI¢Œx·Û]¯×Zÿ|»Ía:[›Ÿ…ü$säãÂ¿ŠíË#Øàƒ§Ë{/ÇÏÑüÈÓ{=‰×PÈs<­xX¿òGb a?Ð@óÐ Ç9¸«G?À+Ÿ™½ç_/×†¸Û.{÷y_í=ÎŸOox®Æ£èþF_n€ï;šyÁ–\øïà9óÉÀøÎ®Íþ–oqfúÚ3j’X¾Ò½úú÷mùî,$y.žf¢Þœ»÷ßßó’÷E‡œz–Žˆoî?¼ïŸCýê\ÚªÆ©Îø2R\ÿÆwOLy)KZ þ¤ž«ÔÞÊ¦¥ˆ³ñÉ>Ã÷ûeI¨";—·*œÛ‰êM¬æï]ÿý~¸“ß@ÍcGü²<—%þÉ†
ê¤‹—q<Šãðýçtbx
|‘G|L8›.wLîø‰áÕðÏßé$'`öB¡¦Á‘$ÜŠ}ðÂ~ÿ 3ø+c¾,+Ë?Ú>£–ß#æãço¢p¯DUÑX¢ãÜúÄúÝT^…JÄñiüã‘KÃG®~ÇqçŸ*·Áß*Ø@VLÇã°Îø•ï× z‘	hN¿ôÙ®-ð½d½ñÕÚ±Aa]ÃNÁèÌØ‡ƒ¥GQDà‘ø	²QÜú»¢(|ßi`ýFá›§ÔûX¶€Ìüµî¨jyð£(:DLàJº(Š‚"û½iàÔ"ãÇ
-8ÿ…IUnªÎ¿Kâ›¢€nö£Ö>b/­|5á»±/€¶8ú///ßòåþ}þZóp¾³Ö±Üï-¿Õ­Ýú±Á^r¡‡ã-«o·þ¢($ìÀRäöcCooµ¸í·ý½²á[,^<‘6û½ ‰ÿ¦Eâ+\À«`áˆÁ£`¥ðG××¼èÀa€¿ ¸\´Úy½f|gñ6.§0ôüØø cûMço~UYþ+ë¿‡Ãa4¤%‘^ûÀißl€‘¸½¯¯¯¯/`ÕBºðD¼¼ßZª íòþ@ÔuÕBùj? —¼3j÷(öÙû&+W1<ñÜl†,p³¯YÏhÁ.]?I¶jÚ{Kç}sôúÅ¥ï\•ïWÍ¾Å‡,¯N™?@>Ñô^ 6>|ƒ5¼9ÿõaq*Á‡
´àéA,¼ü±
ÈÆWÓðyÇ×'OÜ˜õÿñ€ÇŽÜz‚>£[<¹ñÈ÷ÿèø¦xœíØŽœ0àò¸@»ÀÜ•£Wv^Ø‰“tÔ‰Ô®ºeôývB³m9Ç¯Ç‘•ÿúßþôú[+ù+e}lŸ	"CðýÕ;Ú¾__×¶m«õ¿ ï¯å¬_µ ªßðC+°@ý<7 ×,==rúÁÊóûáÎÇñ¡ÎÀò×7‰Ïæ¿ßïuU…ô¬¾¥¿iv ²Ø® Ú¾ô>Ð¾²ß
ú ð;ÌpŒ; -DŒoL‹'7"ÊGå]'€1rô#€ÕiÿIgMB¢ß<V/î?–OiÀ~U”ÿŒ?7à¸ŠQÿ~!¹øpý×5ŒþWÊïžüfÛ\Dõïì^¾3àÎ‘ë‡3wãàða€€}þ[–¥[4ëòï¡êcÏ¿;¿Íw]ceðúÇz`û?‡¿\0|Ü}óÅÙvøç˜÷ùxòÕóýÇ¿m@£çO³ýFïAÔþXÿR•ë€a¯>¾o”Zà\çê0p.;P¼'ÿ¢];ÐÕu2„‹pKô;sñ;°n½±»x¦oôÞ=óg£ŒÖÁ–ì/Ïþ9ó ºé|ãÎÇw€cÝÁÀbÔOç­Y÷Áˆþ]¼íûŸ¾àB¾m»—ŸÉ'ùË£vÇÔ‡|Ðyž_„âëO½¹ëƒî£Ý'¡ÿ!Ýzêƒµ'äû[oùø¾ÃÛo´ÿ;T½öï?ì_ÿ<Ïócû[ÏúÛw5'I¾—¦CW¿ÔoIõËì´vãÓÞÿ:¼í[>æÛzýs×ÍžâËNÐWœ|IGúÖÛ®;€¼•ï<êÃõÆÀô÷W°xbÙ‘¾:èú¶Ü¾ÕýDï+^}r²|¾Sÿu‚ž0£s|“ßtšß«qê“É‡žn~ÇýAtÈ.äƒÏéGþtõ#¿¨óöK³¹>ú ˆNó§m³Ú<ÉðTßÒ·Äyçø2ÒçúÖÜö ¬ï¸ý*ø:€hvý“l/è>Ðg„Qÿ¾
ûŠ¾1¦bþËÆdªÿz½úýOæïü+Oëÿ9\v7¤ø¨~IœÖ4ûÂ<¥~@Z'ÖŸñD©?×Üý«þ|ìù¿ŠŸ#Ï—OÀõ¥WÁçùSe¿ÿßýâÏÿœ¾~Dÿ«¼ÿüS¾X‚Øõ' ¶ÿ•ý¾Ü÷_cß÷CÅùÇÑåëÿ8Žƒ5î¿[¿ˆé0}Aï« Sê·þ	þ0LjÔó‡#ƒÌë—ÛþP(–xœíÐÁ€0Áô”]:âŸùGX3ìé"¶˜Y{òùû©[û‰	Ýûë3
ûqŸò-©/æUóù×€Ã}     €Ö       Àø·}‘3âxœí•ën¤0Fû+È@Ÿ¦¤y;?úÊv¹2¬Pw}F¥z>;NøùI±ö•°mÛöÞÞï÷æX„ï˜µÈ´NÆ3÷ä\ï¥£þ©ä7¿µöEŸ\_ðÇ	Êú8@¦ý–üöe‹þ-×/çö8ÀIÿÉù·-é]€†»àxÊ/YÌä±~9S'ˆ&¢`G‚ç€nÚ®ë©¹ß¦~D*ŸýË.&wÒ€=A3Ë”Õï®¦fß!-}Ä]³“¾Òè 8Áý²6ôc‘EüRîwÖúr$;ÐÏ!wƒ—ºig¿ë}ªrTÇašŽì;oß†¾î—¾×íqLºîg/ðã‰ü²ü#ú•ªRÄ~îÿ)Üþáò³ó':‡ûü’`ÀŸ,?û®ú‡—¿púÆ¯ÁÎú[¯ÓoÊÞKþ^È—„à/oû¿k}O²&ÐÛ_xfÄß]|Aë^yQ~»?è>À
 âOœò¸Ë?^:Ðäb€¿äº.ú•u~ÄŸ&:ñÕ¾rë¥ØÈçÿ’\·øùÅo9vþ0‚¿åÇ óçý/œ8@+à=‡ ?07øk‡þ±ÔGÍQº=õ/]!Òòõw5ËZøi ^0ólæù¤þî)ˆýáòï…‡WŠ0÷øƒ  OæŠþJø¿×üs°3¨÷ö³ ˆ˜l¾r .Øß÷û›	äéæ¯ø)ÀlÜý‡zôn‡ŸûºÙ
ˆWJ/ø÷­é›v.ñRÓ‹þêwšvé8^¯<Õ×ýØx–•žeèÀÃÖþ¹Ÿ®ïx3z=¼”~G wiB:JÂ³NþÏÍiˆ>néú<kÞêÿPýÏÀÿyÙ=ÖwúÌÏÚò?¬Ç‡õê´ûn´+ÊïàKQEQEQEQEQEQEQEQåëwó×#xœí–Áƒ =	“¯Î§whmz(8Ï0mw½xa–—($­;iM)•3s	ËGÒKþ@(ïóo(ê=½…ù7ù¡eWÿgÙzYzýu|¬éze¿—ßcû_§2î·¹~­þËü&îþ	¿Oõ‹«?Üµ~Ì¯?â7ÓÇÊAüA¿\?VÙ­ÂEñ»ýVžã²œCýÍÄ—‘þfàÔé{üoÏÁù¿Rßáo‡ýï¯VHõ=þzAŽö7þÜ/ÖŸÍ?Ùïóü.ÕŸðûl¿Ïö+ÁOýùþøÿ88¹¸™?˜¿˜?™¿               –ŸáaÌ}ì                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        iteDateInYearList);

            % èŽ·å–å…±åŒæ—¥æœŸçš„ç«™ç‚¹æ¸©åº¦åˆ—è¡¨, å¹¶æ ¡æ­£.
            siteAllSoilTMeanList = siteAllSoilTMeanCell{k};
            siteSoilTInYearList = siteAllSoilTMeanList(siteYearIndex);
            siteSoilTInYearList = siteSoilTInYearList(siteDateIndex) + 273.15;
            siteLstInYearList = polyval(pCell{k}, siteSoilTInYearList);

            % èŽ·å–å…±åŒæ—¥æœŸAMSR2 LSTå½±åƒä¸­ç«™ç‚¹æ‰€åœ¨åƒå…ƒçš„æ¸©åº¦åˆ—è¡¨.
            amsr2LstNameList = amsr2LstNameList(amsr2DateIndex);
            amsr2LstNameN = length(amsr2LstNameList);
            amsr2LstList = zeros(amsr2LstNameN, 1)