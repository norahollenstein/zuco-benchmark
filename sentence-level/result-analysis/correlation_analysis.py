import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

#stats = pd.read_excel("/Users/norahollenstein/Desktop/PhD/eth/passive_supervision/task_classification/zuco2_stats.xlsx")
#print(stats)

#stats = pd.read_excel(open("/Users/norahollenstein/Desktop/PhD/eth/passive_supervision/task_classification/zuco2_stats.xlsx", 'rb'))



#print(stats)



def get_subject_avg(results, dataset, feature):

    if dataset == "zuco2":
        subjects = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL',
                'YTL']
        stats = pd.read_csv(
        "/Users/norahollenstein/Desktop/PhD/eth/passive_supervision/task_classification/zuco2_stats.csv",
        )
    if dataset == "zuco1":
        subjects = ["ZDN", "ZPH", "ZJN", "ZAB", "ZJM", "ZKB", "ZKH", "ZMG", "ZGW", "ZKW", "ZDM"]
        stats = pd.read_csv(
            "/Users/norahollenstein/Desktop/PhD/eth/passive_supervision/task_classification/zuco1_stats.csv",
        )


    subj_accs = pd.DataFrame(columns=["ID","acc"])
    for s in subjects:
        subj_results = results[results['subject'] == s]
        subj_results_feat = subj_results[subj_results['feature_set'] == feature]
        acc = subj_results_feat["accuracy"].values[0]
        #print(s, acc)
        new = pd.DataFrame([[s, acc]], columns=['ID', 'acc'])
        subj_accs = subj_accs.append(new, ignore_index=True)
    #print(subj_accs)

    merged_df = stats.merge(subj_accs)
    print(merged_df)
    comparions = ["Score TSR", "Score NR", "Speed NR", "Speed TSR", "LexTALE"]
    for comp in comparions:
        pearsons_r = np.corrcoef(merged_df['acc'], merged_df[comp])
        #print(comp, pearsons_r[0][1])
        spearman = scipy.stats.spearmanr(merged_df['acc'], merged_df[comp])
        print(comp, spearman[0])
        sns.scatterplot(x='acc', y=comp, data=merged_df, hue='ID', palette="flare")
        #sns.regplot(x='acc', y='Score NR', data=merged_df)


        z = np.polyfit(merged_df['acc'], merged_df[comp], 1)
        p = np.poly1d(z)
        plt.plot(merged_df['acc'], p(merged_df['acc']), ":", color="darkblue")
        plt.savefig("plots/correlation_"+feature+"_"+comp+"_"+dataset+".pdf")
        #plt.show()





def main():
    #result_file = "../results/2021-07-09_svm_results_tasks_zuco2_randomFalse_linear.csv" #  for sent_gaze_sacc
    result_file = "../results/2021-04-15_svm_results_tasks_zuco2_randomFalse_linear.csv" # eeg_means & electrode_features_all
    #result_file = "../results/2021-07-09_svm_results_tasks_zuco1_randomFalse_linear.csv"  # for sent_gaze_sacc
    #result_file = "../results/2021-04-15_svm_results_tasks_zuco1_randomFalse_linear.csv" # eeg_means & electrode_features_all
    results = pd.read_csv(result_file, delimiter=" ", names=["subject", "feature_set", "accuracy", "std", "features", "samples", "runs"])
    #print(results.head())
    dataset = result_file.split("_")[4]
    get_subject_avg(results, dataset, "electrode_features_all")




if __name__ == '__main__':
    main()
