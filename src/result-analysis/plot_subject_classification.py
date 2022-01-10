import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_results_detailed(results, dataset):
    features = results.feature_set.unique()
    print(features)
    print(len(results))
    results = results.drop_duplicates()
    print(len(results))

    if dataset == "zuco1":
        random_baseline = 0.0909
    if dataset == "zuco2":
        random_baseline = 0.0625

    ax = sns.barplot(x=results["feature_set"], y=results["accuracy"], palette=sns.color_palette("Spectral", len(features)))
    ax.set_xticklabels(results["feature_set"], rotation=90)
    ax.axhline(random_baseline, ls='-.', color="gray", label="random")
    plt.title("Subject classification")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/subj_class_"+dataset+".pdf")
    plt.show()




def main():
    result_file = "../results/2021-06-16_svm_averaged_results_subjects_zuco2_randomFalse_linear.csv"
    results = pd.read_csv(result_file, delimiter=" ", names=["subject", "feature_set", "accuracy", "std"])
    dataset = result_file.split("_")[5]
    plot_results_detailed(results, dataset)




if __name__ == '__main__':
    main()
