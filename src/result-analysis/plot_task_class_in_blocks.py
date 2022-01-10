import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_results_blocks(results, dataset):
    features = results.feature_set.unique()
    print(features)
    print(len(results))

    ax = sns.lineplot(x="blocks", y="accuracy", data=results, hue="feature_set")#, style="feature_set", markers=True)#, palette=sns.color_palette("Spectral", len(results["blocks"].unique())))
    #ax.set_xticklabels(results["feature_set"], rotation=90)
    plt.title("Task classification with increasing blocks")
    plt.xlabel("blocks per task")
    plt.tight_layout()
    plt.savefig("plots/task_class_blocks"+dataset+".pdf")
    plt.show()


def main():

    results_all = pd.DataFrame()
    for n in list(range(1,7)):
        result_file = "../results/2021-07-21_svm_results_blocks-in-sets_zuco2_randomFalse_linear_"+str(n)+".csv"
        results = pd.read_csv(result_file, delimiter=" ", names=["subject", "feature_set", "accuracy", "std", "features", "samples", "runs"])
        results["blocks"] = n
        #print(results)
        results_all = results_all.append(results)
        dataset = "zuco2"
    print(results_all)
    plot_results_blocks(results_all, dataset)




if __name__ == '__main__':
    main()
