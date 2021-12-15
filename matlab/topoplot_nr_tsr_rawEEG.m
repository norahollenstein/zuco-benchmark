
% load any preprocessed file to get EEG.chanlocs
load('/Volumes/methlab/NLP/Ce_ETH/2019/Preprocessed_V2/YTL/gip_YTL_TSR4_EEG.mat');

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
        averageReference = mean(sentenceData(x).rawData, 1);
        new_sent_data = sentenceData(x).rawData - repmat(averageReference, size(sentenceData(x).rawData, 1),1);
        new_sent_data_avg = mean(new_sent_data, 2);

        nr_m = nansum(nr_m+new_sent_data_avg, 2);
    end
    nr_m=nr_m/length(sentenceData);
    
    tsr_baseFileName = strcat('results', subjects(k),'_TSR.mat');
    tsr_fullFileName = fullfile(zucoDir, tsr_baseFileName);
    fprintf(1, 'Now reading %s\n', tsr_fullFileName);
    
    load(tsr_fullFileName);
    
    tsr_m=0;

    for x=1:length(sentenceData)
        averageReference = mean(sentenceData(x).rawData, 1);
        new_sent_data = sentenceData(x).rawData - repmat(averageReference, size(sentenceData(x).rawData, 1),1);
        new_sent_data_avg = mean(new_sent_data, 2);

        tsr_m = nansum(tsr_m+new_sent_data_avg, 2);
    end
    tsr_m=tsr_m/length(sentenceData);
    
    diff = nr_m - tsr_m;
    mean_all = nansum(mean_all+diff, 2);
    strcat(string(subjects(k)), ',', string(min(diff)), ',', string(max(diff)))
    % before: scale -1, 1
    fig = topoplot(diff, EEG.chanlocs, 'electrodes', 'labels', 'style', 'both', 'maplimits', [min(diff),max(diff)]);
    saveas(fig, strcat(string(subjects(k)),'_rawSentEEG_diff_scaled.png'));
    close;
end
mean_all_final = mean_all/length(subjects);
strcat('all,', string(min(mean_all_final)), ',', string(max(mean_all_final)))
fig_all = topoplot(mean_all_final, EEG.chanlocs, 'electrodes', 'labels', 'style', 'both', 'maplimits', [min(mean_all_final),max(mean_all_final)]);
saveas(fig_all, 'allSUbjects_rawSentEEG_diff_scaled.png');



