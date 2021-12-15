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

    random_baseline = 0.0714

    ax = sns.barplot(x=results["feature_set"], y=results["accuracy"], palette=sns.color_palette("Spectral", len(features)))
    ax.set_xticklabels(results["feature_set"], rotation=90)
    ax.axhline(random_baseline, ls='-.', color="gray", label="random")
    plt.title("Block classification")
    plt.tight_layout()
    plt.legend()
    plt.savefig("plots/block_class_"+dataset+".pdf")
    plt.show()


def main():
    result_file = "../results/2021-07-13_svm_results_blocks_zuco2_randomFalse_linear.csv"
    results = pd.read_csv(result_file, delimiter=" ", names=["subject", "feature_set", "accuracy", "std", "features", "samples", "runs"])
    print(results.head())
    dataset = result_file.split("_")[4]
    plot_results_detailed(results, dataset)




if __name__ == '__main__':
    main()
