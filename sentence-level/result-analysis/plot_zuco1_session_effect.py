import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def plot_results_compared(results_sess, results_tasks):
    features = results_sess.feature_set.unique()
    #print(features)
    #print(len(results_sess))
    results_sess = results_sess.drop_duplicates()
    #print(len(results_sess))

    #print(len(results_tasks))
    results_tasks = results_tasks.drop_duplicates()
    #print(len(results_tasks))

    sesss = []
    tasks = []

    for f in features:
        feature_results_sess = results_sess[results_sess['feature_set'] == f]
        feature_results_sess = feature_results_sess.sort_values(by=['accuracy'])

        print("Mean accuracy:", f, np.mean(feature_results_sess['accuracy']))
        sesss.append(np.mean(feature_results_sess['accuracy']))

        feature_results_tasks = results_tasks[results_tasks['feature_set'] == f]
        feature_results_tasks = feature_results_tasks.sort_values(by=['accuracy'])
        tasks.append(np.mean(feature_results_tasks['accuracy']))

        print("Mean accuracy:", f, np.mean(feature_results_tasks['accuracy']))

    fig, ax = plt.subplots()

    print([t-s for t,s in zip(tasks,sesss)])
    ax.bar(features, sesss, label='Additional sents.', color=sns.color_palette("Spectral", len(features)))
    ax.bar(features,[t-s for t,s in zip(tasks,sesss)], bottom=sesss,
           label='NR/TSR', color=sns.color_palette("Spectral", len(features)), alpha=0.4)

    plt.ylim(0.49, 1.03)
    ax.set_xticklabels(features, rotation=90)
    #plt.title("Session classification")
    plt.tight_layout()
    plt.legend()
    plt.savefig("plots/session_effect_zuco1.pdf")
    plt.show()



def main():
    result_file_sentiment = "../results/2021-07-26_svm_results_sessions_zuco1sr-only-balanced_randomFalse_linear.csv"
    results_senti = pd.read_csv(result_file_sentiment, delimiter=" ", names=["subject", "feature_set", "accuracy", "std", "features", "samples", "runs"])
    print(results_senti.head())
    dataset = result_file_sentiment.split("_")[4]

    result_file_nrtsr = "../results/2021-07-26_svm_results_sessions_zuco1-balanced_randomFalse_linear.csv"
    results_nrtsr = pd.read_csv(result_file_nrtsr, delimiter=" ", names=["subject", "feature_set", "accuracy", "std", "features", "samples", "runs"])
    print(results_nrtsr.head())
    dataset = result_file_nrtsr.split("_")[4]

    plot_results_compared(results_senti, results_nrtsr)




if __name__ == '__main__':
    main()
