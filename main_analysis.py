import argparse
import copy
import os
from pathlib import Path

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

from tabulate import tabulate
from crffortcpci.statistics.analysis import Analisys

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


def run_complete_analysis(ana, dataset, IS_TUNNING, RUN_WITH_DETERMINISTIC):
    if not IS_TUNNING:
        # Accumulative
        ana.plot_accumulative("ACC_NAPFD")
        ana.plot_accumulative("ACC_APFDc", 'cost')  # APFDc
        ana.plot_accumulative("ACC_Reward", 'rewards')  # Rewards

        # # Variation Visualization along the CI Cycles
        ana.plot_lines("NAPFD_Variation")
        ana.plot_lines("APFDc_Variation", 'cost')  # APFDc

        # Unique boxplot
        # ana.visualize_boxplot_unique()  # NAPFD
        # ana.visualize_boxplot_unique('cost')  # APFDc
        # ana.visualize_boxplot_unique('ttf')  # RFTC
        # ana.visualize_boxplot_unique('prioritization_time')  # Prioritization Time

    # Normalized Time Reduction
    df_stats = ana.visualize_ntr()

    # Apply the Kruskal-Wallis Test in the Data
    df_stats = df_stats.append(ana.statistical_test_kruskal())  # NAPFD
    df_stats = df_stats.append(ana.statistical_test_kruskal('cost'))  # APFDc
    df_stats = df_stats.append(ana.statistical_test_kruskal('ttf'))  # RFTC
    df_stats = df_stats.append(ana.statistical_test_kruskal('prioritization_time'))  # Prioritization Time
    df_stats = df_stats.append(ana.statistical_test_kruskal('ttf_duration'))  # Time Spent to Fail

    # Test Set Size
    #df_stats = df_stats.append(ana.visualize_test_set_size('scheduled_testcases'))
    #df_stats = df_stats.append(ana.visualize_test_set_size('unscheduled_testcases'))
    df_stats = df_stats.append(ana.visualize_test_set_size('testcaseset_percentage_scheduled'))

    # Faults Detection-Effectiveness
    df_stats = df_stats.append(ana.accumulative_fde())

    # Update the current dataset used
    df_stats['Dataset'] = dataset

    if RUN_WITH_DETERMINISTIC:
        # RMSE
        df_distances = ana.rmse_calculation()
        df_distances = df_distances.append(ana.rmse_calculation('cost'))

        # Jaccard
        df_distances = df_distances.append(
            ana.jaccard_similarity_calculation())

        # Update the current dataset used
        df_distances['Dataset'] = dataset

        return df_stats, df_distances

    return df_stats, None


def export_df(df, filename, caption):
    # print(f"Exporting {filename}")
    with open(f'{filename}.txt', 'w') as tf:
        tf.write(tabulate(df, headers='keys', tablefmt='psql', showindex=False))

    latex = df.to_latex(index=False)

    # Remove special characters provided by pandas
    latex = latex.replace("\\textbackslash ", "\\").replace(
        "\$", "$").replace("\{", "{").replace("\}", "}")

    # split lines into a list
    latex_list = latex.splitlines()

    # Insert new LaTeX commands
    latex_list.insert(0, '\\begin{table*}[!ht]')
    latex_list.insert(1, f'\\caption{{{caption}}}')
    latex_list.insert(2, '\\resizebox{\\linewidth}{!}{')
    latex_list.append('}')
    latex_list.append('\\end{table*}')

    # join split lines to get the modified latex output string
    latex_new = '\n'.join(latex_list)

    # Save in a file
    with open(f'{filename}.tex', 'w') as tf:
        tf.write(latex_new)


def print_dataset(dataset):
    print(f"====================================================\n\t\t{dataset}\n"
          f"====================================================")


def get_best_equiv(df, column):
    equiv = len(df[df[column].str.contains("cellgray")])
    best = len(df[df[column].str.contains("cellbold")])

    return best, equiv


def get_magnitude(df, column):
    very_near = len(df[df[column].str.contains("bigstar")])
    near = len(df[df[column].str.contains("blacktriangledown")])
    far = len(df[df[column].str.contains("vartriangle")])

    df_temp = df[~df[column].str.contains("blacktriangledown")]

    reasonable = len(df_temp[df_temp[column].str.contains("triangledown")])
    very_far = len(df_temp[df_temp[column].str.contains("blacktriangle$")])

    return very_near, near, reasonable, far, very_far


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Analysis')

    ap.add_argument('--is_tunning', default=False,
                    type=lambda x: (str(x).lower() == 'true'))
    ap.add_argument('--results_dir', default="results/group_selection")
    ap.add_argument('--dataset_dir', default='context_datasets')
    ap.add_argument('--datasets', nargs='+', default=[
        "alibaba@druid",
        "alibaba@fastjson",
        "DSpace@DSpace",
        "square@okhttp",
        "square@retrofit",
        "zxing@zxing"
    ],
                    help='Datasets to analyse. Ex: \'deeplearning4j@deeplearning4j\'')

    args = ap.parse_args()

    IS_TUNNING = args.is_tunning
    table_results_dir_main = f"{args.results_dir}_plots{os.sep}tables"
    Path(table_results_dir_main).mkdir(parents=True, exist_ok=True)

    if IS_TUNNING:
        replace_names = {
            'FRRMAB (C=0.3, D=1, SW=100)': 'FRRMAB',
            'LinUCB (Alpha=0.5)_Time_Execution': 'LinUCB',
            'SWLinUCB (Alpha=0.5, SW=100)_Time_Execution': 'SWLincUCB',
            "LSTM-2-layers-iofrol": "2L",
            "LSTM": "LSTM",
            "GRU-1": "GRU",
            "RF_Naive": "RF_Naive",
            "RF_SW_100": "RF_SW_100",
            "RF_SW_1": "RF_SW_1",
            "RF_SW_4_estimators_500": "RF",
            "RF_Vertical_SW_estimators_500": "RF_Vertical_SW_fill_0",
            "RF_Vertical_SW_fill_1_estimators_500": "RF_Vertical_SW_fill_1",
            "RF_Vertical_SW_fill_MEAN_estimators_500": "RF_Vertical_SW_fill_MEAN",
            "XGBoost_Vertical_estimators_500": "XGBoost_Vertical",
            "XGBoost_Vertical_SW_historical_4_estimators_500": "XGBoost_Vertical_SW_historical_500e",
            "XGBoost_Vertical_SW_historical_4_estimators_100": "XGBoost_Vertical_SW_historical_100e",
            "XGBoost_Vertical_SW_fill_MEAN_4_estimators_500": "XGBoost_Vertical_SW_fill_MEAN_500e",
            "XGBoost_Vertical_SW_fill_MEAN_4_estimators_100": "XGBoost_Vertical_SW_fill_MEAN_100e",
            "RF_Vertical_SW_history_based_4_estimators_500": "RF_Vertical_SW_history_based_500e",
            "RF_Vertical_SW_history_based_4_estimators_100": "RF_Vertical_SW_history_based_100e",
            "RF_Vertical_SW_history_based_40_estimators_500": "CRF",
            "RF_Vertical_SW_history_based_40_estimators_500_All_Features": "CRF_AF",
            "RF_Vertical_SW_history_based_40_estimators_500_Time_Execution": "CRF_TE",
            "RF_Vertical_SW_history_based_40_estimators_500_Program_Size": "CRF_PS",
            "RF_Vertical_SW_history_based_40_estimators_500_TC_Complexity": "CRF_TCC",
            "RF_Vertical_SW_history_based_40_estimators_500_TC_Evolution": "CRF_TCE",
            "RF_Vertical_SW_history_based_40_estimators_500_Feature_Selection": "CRF_FS",
            "XGBoost_Vertical_SW_historical_40_estimators_500": "XGBoost"
        }
    else:
        if 'merge' in args.results_dir:
            replace_names = {
                'FRRMAB (C=0.3, D=1, SW=100)': 'FRRMAB',
                'LinUCB (Alpha=0.5)_Time_Execution': 'LinUCB',
                'SWLinUCB (Alpha=0.5, SW=100)_Time_Execution': 'SWLincUCB',
                "LSTM-2-layers-iofrol": "2L",
                "LSTM": "LSTM",
                "GRU-1": "GRU",
                "RF_Naive": "RF_Naive",
                "RF_SW_100": "RF_SW_100",
                "RF_SW_1": "RF_SW_1",
                "RF_SW_4_estimators_500": "RF",
                "RF_Vertical_SW_estimators_500": "RF_Vertical_SW_fill_0",
                "RF_Vertical_SW_fill_1_estimators_500": "RF_Vertical_SW_fill_1",
                "RF_Vertical_SW_fill_MEAN_estimators_500": "RF_Vertical_SW_fill_MEAN",
                "XGBoost_Vertical_estimators_500": "XGBoost_Vertical",
                "XGBoost_Vertical_SW_historical_4_estimators_500": "XGBoost_Vertical_SW_historical_500e",
                "XGBoost_Vertical_SW_historical_4_estimators_100": "XGBoost_Vertical_SW_historical_100e",
                "XGBoost_Vertical_SW_fill_MEAN_4_estimators_500": "XGBoost_Vertical_SW_fill_MEAN_500e",
                "XGBoost_Vertical_SW_fill_MEAN_4_estimators_100": "XGBoost_Vertical_SW_fill_MEAN_100e",
                "RF_Vertical_SW_history_based_4_estimators_500": "RF_Vertical_SW_history_based_500e",
                "RF_Vertical_SW_history_based_4_estimators_100": "RF_Vertical_SW_history_based_100e",
                "RF_Vertical_SW_history_based_40_estimators_500": "CRF",
                "RF_Vertical_SW_history_based_40_estimators_500_All_Features": "CRF_AF",
                "RF_Vertical_SW_history_based_40_estimators_500_Time_Execution": "CRF_TE",
                "RF_Vertical_SW_history_based_40_estimators_500_Program_Size": "CRF_PS",
                "RF_Vertical_SW_history_based_40_estimators_500_TC_Complexity": "CRF_TCC",
                "RF_Vertical_SW_history_based_40_estimators_500_TC_Evolution": "CRF_TCE",
                "RF_Vertical_SW_history_based_40_estimators_500_Feature_Selection": "CRF_FS",
                "XGBoost_Vertical_SW_historical_40_estimators_500": "XGBoost"
            }
        else:
            replace_names = {
                'FRRMAB (C=0.3, D=1, SW=100)': 'FRRMAB',
                'LinUCB (Alpha=0.5)_Time_Execution': 'LinUCB',
                'SWLinUCB (Alpha=0.5, SW=100)_Time_Execution': 'SWLincUCB',
                "LSTM-2-layers-iofrol": "2L",
                "LSTM": "LSTM",
                "GRU-1": "GRU",
                "RF_Naive": "RF_Naive",
                "RF_SW_100": "RF_SW_100",
                "RF_SW_1": "RF_SW_1",
                "RF_SW_4_estimators_500": "RF",
                "RF_Vertical_SW_estimators_500": "RF_Vertical_SW_fill_0",
                "RF_Vertical_SW_fill_1_estimators_500": "RF_Vertical_SW_fill_1",
                "RF_Vertical_SW_fill_MEAN_estimators_500": "RF_Vertical_SW_fill_MEAN",
                "XGBoost_Vertical_estimators_500": "XGBoost_Vertical",
                "XGBoost_Vertical_SW_historical_4_estimators_500": "XGBoost_Vertical_SW_historical_500e",
                "XGBoost_Vertical_SW_historical_4_estimators_100": "XGBoost_Vertical_SW_historical_100e",
                "XGBoost_Vertical_SW_fill_MEAN_4_estimators_500": "XGBoost_Vertical_SW_fill_MEAN_500e",
                "XGBoost_Vertical_SW_fill_MEAN_4_estimators_100": "XGBoost_Vertical_SW_fill_MEAN_100e",
                "RF_Vertical_SW_history_based_4_estimators_500": "RF_Vertical_SW_history_based_500e",
                "RF_Vertical_SW_history_based_4_estimators_100": "RF_Vertical_SW_history_based_100e",
                "RF_Vertical_SW_history_based_40_estimators_500": "CRF",
                "RF_Vertical_SW_history_based_40_estimators_500_All_Features": "CRF_AF",
                "RF_Vertical_SW_history_based_40_estimators_500_Time_Execution": "CRF_TE",
                "RF_Vertical_SW_history_based_40_estimators_500_Program_Size": "CRF_PS",
                "RF_Vertical_SW_history_based_40_estimators_500_TC_Complexity": "CRF_TCC",
                "RF_Vertical_SW_history_based_40_estimators_500_TC_Evolution": "CRF_TCE",
                "RF_Vertical_SW_history_based_40_estimators_500_Feature_Selection": "CRF_FS",
                "XGBoost_Vertical_SW_historical_40_estimators_500": "XGBoost"
            }

    print_dataset(args.datasets[0])

    ana = Analisys(f"{args.dataset_dir}/{args.datasets[0]}", args.results_dir, replace_names=replace_names)

    policies = list(ana._get_policies())

    RUN_WITH_DETERMINISTIC = 'Deterministic' in policies

    df_stats_main = pd.DataFrame(columns=['Dataset', 'Metric', 'TimeBudget'] + policies)

    if RUN_WITH_DETERMINISTIC:
        policies.remove('Deterministic')

    df_distances_main = pd.DataFrame(
        columns=['Dataset', 'Metric', 'TimeBudget'] + policies)

    df_stats, df_distances = run_complete_analysis(
        ana, args.datasets[0], IS_TUNNING, RUN_WITH_DETERMINISTIC)
    df_stats_main = df_stats_main.append(df_stats)

    if RUN_WITH_DETERMINISTIC:
        df_distances_main = df_distances_main.append(df_distances)

    for dataset in args.datasets[1:]:
        print_dataset(dataset)

        ana.update_project(f"{args.dataset_dir}/{dataset}", replace_names=replace_names)

        df_stats, df_distances = run_complete_analysis(
            ana, dataset, IS_TUNNING, RUN_WITH_DETERMINISTIC)
        df_stats_main = df_stats_main.append(df_stats)

        if RUN_WITH_DETERMINISTIC:
            df_distances_main = df_distances_main.append(df_distances)

    print("\n\n\n\n\n\n===========================================================")
    print(f"\t\tSummary Results ")
    print("===========================================================")
    if RUN_WITH_DETERMINISTIC:
        df_distances_main.sort_values(
            by=['TimeBudget', 'Dataset'], inplace=True)
        metrics_dist = df_distances_main['Metric'].unique()
        df_distances_main['TimeBudget'] = pd.to_numeric(
            df_distances_main['TimeBudget'])

        print("\n\n\n== Distance")
        for tr in [10, 50, 80]:
            print(f"\n\nTime Budget {tr}%")
            for m in metrics_dist:
                print("\n\n~~", m)
                df_temp = df_distances_main[
                    (df_distances_main.TimeBudget == tr) & (df_distances_main.Metric == m)]

                df_temp.sort_values(by=['Dataset'], inplace=True)

                del df_temp['TimeBudget']
                del df_temp['Metric']
                policies = list(df_temp.columns)

                filename = f"{table_results_dir_main}{os.sep}{m}_{tr}"
                caption = f"{m} values - Time Budget {tr}\\%"

                export_df(df_temp, filename, caption)

                policies.remove('Dataset')
                for pol in policies:
                    best, equiv = get_best_equiv(df_temp, pol)
                    print(f"{pol}: {best} ({equiv}) ")

                if 'RMSE' in m or 'Jaccard' in m:
                    print(f"\n{m} Magnitudes:")

                    for pol in policies:
                        print("\n", pol)
                        very_near, near, reasonable, far, very_far = get_magnitude(
                            df_temp, pol)

                        print(f"Very Near: {very_near}({round(very_near * 100 / len(args.datasets))})")
                        print(f"Near: {near}({round(near * 100 / len(args.datasets))})")
                        print(f"Reasonable: {reasonable}({round(reasonable * 100 / len(args.datasets))})")
                        print(f"Far: {far}({round(far * 100 / len(args.datasets))})")
                        print(f"Very Far: {very_far}({round(very_far * 100 / len(args.datasets))})")

    df_stats_main.sort_values(by=['TimeBudget', 'Dataset'], inplace=True)
    metrics = df_stats_main['Metric'].unique()
    df_stats_main['TimeBudget'] = pd.to_numeric(df_stats_main['TimeBudget'])

    print("\n\n\n== Each Metric")
    for tr in [10, 50, 80]:
        print(f"\n\nTime Budget {tr}%")
        for m in metrics:
            print("\n~~", m)
            df_temp = df_stats_main[
                (df_stats_main.TimeBudget == tr) & (df_stats_main.Metric == m)]

            del df_temp['TimeBudget']
            del df_temp['Metric']

            df_temp.sort_values(by=['Dataset'], inplace=True)

            policies = list(df_temp.columns)

            filename = f"{table_results_dir_main}{os.sep}NTR_{tr}" if "NTR" in m \
                else f"{table_results_dir_main}{os.sep}stats_{tr}_{m}"

            caption = f"NTR values - Time budget {tr}\\%" if "NTR" in m else f"{m} values - Time Budget {tr}\\%"
            export_df(df_temp, filename, caption)

            policies.remove('Dataset')
            print(' & '.join(policies))
            for i, col in enumerate(policies):
                best, equiv = get_best_equiv(df_temp, col)

                if i == len(policies) - 1:
                    print(f"{best} ({equiv})")
                else:
                    print(f"{best} ({equiv}) & ", end="")
