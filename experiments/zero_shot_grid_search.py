import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def plot_percentile():

    results_df = pd.read_csv("retrieval_results_BP.csv")

    results_df[["Sequence", "Structure", "Text", "Interpro"]] = results_df["config"].str.split("_", expand=True)
    results_df[["Sequence", "Structure", "Text", "Interpro"]] = results_df[["Sequence", "Structure", "Text", "Interpro"]].astype(float)
    results_df = results_df.drop(columns=["config", "run_index", "group_index"])

    grouped = results_df.groupby(['Sequence', 'Structure', 'Text', 'Interpro'])\
                        [['Ret@1', 'Ret@3', 'Ret@5', 'MRR']].agg(['mean', 'max']).reset_index()


    grouped.columns = ['{}_{}'.format(col[0], col[1]) if col[1] else col[0] for col in grouped.columns.values]

    grouped = grouped[['Sequence', 'Structure', 'Text', 'Interpro', 'Ret@1_mean']]
    print(grouped.describe())

    sorted_ret1 = grouped['Ret@1_mean'].sort_values(ascending=True)

    percentiles = [0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95]
    percentiles = [0.50, 0.80, 0.90, 0.95]
    percentile_values = sorted_ret1.quantile(percentiles)

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(sorted_ret1) + 1), sorted_ret1.values)

    for p, value in percentile_values.items():
        plt.axhline(value, color='r', linestyle=':', linewidth=0.8, alpha=1.0, label=f'{value:.4f}' if p in [0.05, 0.50, 0.95] else "")

    # Add labels and title
    plt.xlabel("Ranked Weight Combinations (Highest to Lowest Ret@1_mean)")
    plt.ylabel("Average Ret@1")
    plt.title("Distribution of Average Ret@1 Scores Across Weight Combinations")
    plt.legend(fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("ret1_distribution_BP.png", dpi=300)


def determine_weights():

    results_df = pd.read_csv("retrieval_results.csv")

    #results_df[["Sequence", "Structure", "Text", "Interpro"]] = results_df["config"].str.split("_", expand=True)
    #results_df[["Sequence", "Structure", "Text", "Interpro"]] = results_df[["Sequence", "Structure", "Text", "Interpro"]].astype(float)
    # results_df = results_df.drop(columns=["config", "run_index", "group_index"])




    grouped = results_df.groupby(['config'])[['Ret@1', 'Ret@3', 'Ret@5', 'MRR']].agg(['mean', 'max']).reset_index()
    grouped.columns = ['{}_{}'.format(col[0], col[1]) if col[1] else col[0] for col in grouped.columns.values]

    print(grouped.columns)


    criteria = "Ret@1_mean"
    top_groups = grouped.sort_values(by=criteria, ascending=False)



    print(top_groups[['config', criteria]].head(20))
    exit()

    grouped = results_df.groupby(['Sequence', 'Structure', 'Text', 'Interpro'])\
                        [['Ret@1', 'Ret@3', 'Ret@5', 'MRR']].agg(['mean', 'max']).reset_index()


    grouped.columns = ['{}_{}'.format(col[0], col[1]) if col[1] else col[0] for col in grouped.columns.values]





    exit()

    grouped = grouped[['Sequence', 'Structure', 'Text', 'Interpro', 'Ret@1_mean']]


    criteria = "Ret@1_mean"

    threshold = grouped[criteria].quantile(0.75)
    # threshold = 0.7
    # CC- 0.7; MF- 0.75; BP- 0.75; ALL- 0.85
    top_groups = grouped[grouped[criteria] >= threshold]

    top_groups = top_groups.sort_values(by=criteria, ascending=False)

    pd.set_option('display.max_columns', None)
    print(top_groups.shape)

    average_of_each_column = top_groups.mean(numeric_only=True)
    print("\nAverage of each numeric column:")
    print(average_of_each_column)


determine_weights()