
% load any preprocessed file to get EEG.chanlocs
load('gip_ZAB_SR5_EEG.mat');

% ZuCo 2
zucoDir = "/Volumes/methlab/NLP/Ce_ETH/2019/FirstLevel_V2/";
subjects = {'YAC','YAG', 'YAK', 'YDG','YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YMS', 'YRH', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL'}; 
mean_all = 0;
for k = 1:length(subjects)
    nr_baseFileName = strcat('results', subjects(k),'_NR.mat');
    nr_fullFileName = fullfile(zucoDir, nr_baseFileName);
    fprintf(1, 'Now reading %s\n', nr_fullFileName);
    
    load(nr_fullFileName);
    
    nr_m=0;
    for x=1:length(sentenceData)
        %m = m+sentenceData(x).mean_t1;
        nr_m = nansum(nr_m+sentenceData(x).mean_g1, 1);
    end
    nr_m=nr_m/length(sentenceData);
    
    tsr_baseFileName = strcat('results', subjects(k),'_TSR.mat');
    tsr_fullFileName = fullfile(zucoDir, tsr_baseFileName);
    fprintf(1, 'Now reading %s\n', tsr_fullFileName);
    
    load(tsr_fullFileName);
    
    tsr_m=0;
    for x=1:length(sentenceData)
        %m = m+sentenceData(x).mean_t1;
        tsr_m = nansum(tsr_m+sentenceData(x).mean_g1, 1);
    end
    tsr_m=tsr_m/length(sentenceData);
    
    diff_g1 = nr_m - tsr_m;
    mean_all = nansum(mean_all+diff_g1, 1);
    strcat(string(subjects(k)), ',', string(min(diff_g1)), ',', string(max(diff_g1)))
    % before: -1, 1
    fig = topoplot(diff_g1, EEG.chanlocs, 'electrodes', 'labels', 'style', 'both', 'maplimits', [min(diff_g1),max(diff_g1)]);
    saveas(fig, strcat(string(subjects(k)),'_gamma1_diff_scaled.png'));
    close;
end
mean_all = mean_all/length(subjects);
strcat('all,', string(min(mean_all)), ',', string(max(mean_all)))
fig_all = topoplot(mean_all, EEG.chanlocs, 'electrodes', 'labels', 'style', 'both', 'maplimits', [min(mean_all),max(mean_all)]);
saveas(fig_all, 'allSUbjects_gamma1_diff_scaled.png');
