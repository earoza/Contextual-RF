import csv
import os
import pandas as pd

from pathlib import Path

# extinctions want to search
extensions = ('.csv')

path_results = f'results{os.sep}results4merge{os.sep}'
#deterministic_dir = f"results{os.sep}experiments_deterministic"
coleman_dir = f"results{os.sep}merge_constantine_coleman{os.sep}merge_experiments_constantine_coleman"
rf_dir = f"results{os.sep}new_rf"
save_dir = f'results{os.sep}all_approaches'

cols = ['experiment',
        'step',
        'policy',
        'reward_function',
        'sched_time',
        'sched_time_duration',
        'total_build_duration',
        'prioritization_time',
        'detected',
        'missed',
        'tests_ran',
        'scheduled_testcases',
        'tests_not_ran',
        'unscheduled_testcases',
        'ttf',
        'ttf_duration',
        'time_reduction',
        'fitness',
        'cost',
        'rewards',
        'avg_precision',
        'prioritization_order'
        ]


def save_df(fpaths, savedir, df):
    merge_dir = f'{os.sep}'.join(fpaths[2:-1])

    merge_file = f"{savedir}{os.sep}{target_file}"
    merge_dir = f"{savedir}{os.sep}{merge_dir}"

    # Create the folder if necessary
    Path(merge_dir).mkdir(parents=True, exist_ok=True)

    # To improve the view and observe if the data is correct
    df.sort_values(by=['experiment', 'step', 'policy'], inplace=True)

    print("Saving:", merge_file)

    # Save the data filtered
    df.to_csv(merge_file, sep=';', na_rep='[]',
              header=True, index=False,
              quoting=csv.QUOTE_NONE)


# this loop though directories recursively
for dname, dirs, files in os.walk(path_results):
    # exclude directory if in exclude list
    for fname in files:
        if (fname.lower().endswith(extensions)):
            # this generate full directory path for file that we use to read
            # the main data
            full_path = os.path.join(dname, fname)
            
            print("\n\nReading:", full_path)

            fpaths = full_path.split(os.sep)

            target_file = f'{os.sep}'.join(fpaths[2:])                    
            coleman_file = f'{coleman_dir}{os.sep}{target_file}'
            rf_file = f'{rf_dir}{os.sep}{target_file}'

            print("The COLEMAN data is in:", coleman_file)

            # Load base data
            print("Loading base data")
            df = pd.DataFrame
            df_base = pd.read_csv(full_path, sep=';', thousands=',')
            # df_base = pd.read_csv(full_path, sep=';', thousands=',', error_bad_lines=False) # in case of bug in lines,
            # find said lines and correct them
            # df_x = df_base[df_base.policy == 'XGBoost_Vertical_SW_historical_40_estimators_500_Time_Execution']
            df_x = df_base[df_base.policy == 'inexistente']
            df_rf = df_base[df_base.policy == 'RF_Vertical_SW_history_based_40_estimators_500_Time_Execution']
            # df_rf = df_base[df_base.policy == 'inexistente']
            df = df.append(df_x, df_rf)
            df = df[cols]

            # Load COLEMAN data
            print("Loading COLEMAN data")
            df_lin = pd.DataFrame
            df_col = pd.read_csv(coleman_file, sep=';', thousands=',')
            # df_col_ucb = df_col[df_col.policy == 'LinUCB (Alpha=0.5)_Time_Execution']
            # df_col_ucb = df_col[df_col.policy == 'inexistente']
            df_col_ucb = df_col[df_col.policy == 'FRRMAB (C=0.3, D=1, SW=100)']
            df_col_sw = df_col[df_col.policy == 'SWLinUCB (Alpha=0.5, SW=100)_Time_Execution']
            # df_col_sw = df_col[df_col.policy == 'inexistente']
            # df_col = df_col[df_col.experiment < 11]
            df_lin = df_lin.append(df_col_ucb, df_col_sw)
            df_lin = df_lin[cols]

            # Load Random Forest data
            print("Loading Random Forest data")
            df_col_rf = pd.read_csv(rf_file, sep=';', thousands=',', error_bad_lines=False)
            df_col_rf = df_col_rf.sort_values(['experiment', 'step'])
            df_col_rf = df_col_rf[df_col_rf.experiment < 11]
            df_col_rf = df_col_rf[cols]

            # First merge results only with COLEMAN
            # df = df.append(df_col)
            df = df.append(df_lin)
            # Then, merge results with Random Forest and save
            df = df.append(df_col_rf)
            df.drop_duplicates(inplace=True)
            save_df(fpaths, save_dir, df)
