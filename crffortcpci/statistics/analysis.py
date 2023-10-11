import os
import textwrap
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# rpy2
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from roses.effect_size import vargha_delaney
from roses.statistical_test.kruskal_wallis import kruskal_wallis

MAX_XTICK_WIDTH = 10

# For a beautiful plots
plt.style.use('ggplot')
sns.set_style("whitegrid")
sns.set(palette="pastel")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


def exception_to_string(excp):
    stack = traceback.extract_stack(
    )[:-3] + traceback.extract_tb(excp.__traceback__)  # add limit=??
    pretty = traceback.format_list(stack)
    return ''.join(pretty) + '\n  {} {}'.format(excp.__class__, excp)


class Analisys(object):

    def __init__(self, project_dir, results_dir, font_size_plots=20, sched_time_ratio=[0.1, 0.5, 0.8],
                 replace_names=[]):
        self.project_dir = project_dir
        self.project = project_dir.split('/')[-1]
        self.results_dir = results_dir

        self.update_figure_dir(f"{self.results_dir}_plots{os.sep}{self.project}")

        self.sched_time_ratio = sched_time_ratio
        self.sched_time_ratio_names = [
            str(int(tr * 100)) for tr in sched_time_ratio]

        # Load the information about the system
        self.df_system = self._get_df_system()

        # Load the results from system
        self.datasets = {}
        self._load_datasets(replace_names)

        # TODO: get information from reward.py
        self.reward_names = {
            'Time-ranked Reward': 'timerank',
            'Reward Based on Failures': 'RNFail',
            'Reward Based on Rank': 'RRank',
            'NA': 'LSTM',
            'LSTM': 'LSTM',
            'tahn': 'tahn',
            'relu': 'relu',
            'bagging': 'Bagging',
            'boosting': 'Boosting'
        }

        self.font_size_plots = font_size_plots
        self._update_rc_params()

    def update_figure_dir(self, path):
        self.figure_dir = path
        Path(self.figure_dir).mkdir(parents=True, exist_ok=True)

    def update_project(self, project_dir, replace_names=[]):
        self.project_dir = project_dir
        self.project = project_dir.split('/')[-1]

        self.figure_dir = f"{self.results_dir}_plots/{self.project}"

        Path(self.figure_dir).mkdir(parents=True, exist_ok=True)

        # Load the information about the system
        self.df_system = self._get_df_system()

        # Load the results from system
        self.datasets = {}
        self._load_datasets(replace_names)

    def update_font_size(self, font_size_plots):
        self.font_size_plots = font_size_plots
        self._update_rc_params()

    def _update_rc_params(self):
        plt.rcParams.update({
            'font.size': self.font_size_plots,
            'xtick.labelsize': self.font_size_plots,
            'ytick.labelsize': self.font_size_plots,
            'legend.fontsize': self.font_size_plots,
            'axes.titlesize': self.font_size_plots,
            'axes.labelsize': self.font_size_plots,
            'figure.max_open_warning': 0,
            'pdf.fonttype': 42
        })

    def _get_df_system(self):
        # Dataset Info
        df = pd.read_csv(f'{self.project_dir}/features-engineered.csv', sep=';', thousands=',')
        df = df.groupby(['BuildId'], as_index=False).agg(
            {'Duration': np.sum, 'NumErrors': np.sum})
        df.rename(columns={'BuildId': 'step',
                           'Duration': 'duration',
                           'NumErrors': 'faults'}, inplace=True)

        return df

    def _load_datasets(self, replace_names=[]):
        for tr in self.sched_time_ratio_names:
            df_path = f"{self.results_dir}/time_ratio_{tr}"

            df = pd.read_csv(f'{df_path}/{self.project}.csv', sep=';', thousands=',', low_memory=False)

            # df = df[['experiment', 'step', 'policy', 'reward_function', 'prioritization_time', 'time_reduction', 'ttf',
            #          'fitness', 'avg_precision', 'cost', 'rewards']]

            df['policy'] = df['policy'].apply(
                lambda x: replace_names[x] if x in replace_names.keys() else x)

            policies = list(df['policy'].unique())

            if 'Deterministic' in policies:
                # df['name'] = df['policy']

                n_builds = len(df['step'].unique())

                # Find the deterministic
                dt = df[df['policy'] == 'Deterministic']

                # As we have only one experiment run (deterministic), we increase to have 30 independent runs
                # This allow us to calculate the values without problems :D
                dt = dt.append([dt] * 29, ignore_index=True)
                dt['experiment'] = np.repeat(list(range(1, 31)), n_builds)

                # Clean
                df = df[df['policy'] != 'Deterministic']

                df = df.append(dt)

                # df.sort_values(by=['name'], inplace=True)

                df.drop_duplicates(inplace=True)

            df.sort_values(by=['experiment', 'step', 'policy'], inplace=True)

            self.datasets[tr] = df

    def replace_names(self, names: dict, column='policy'):
        for key in self.datasets.keys():
            self.datasets[key][column] = self.datasets[key][column].apply(
                lambda x: names[x] if x in names.keys() else x)

    def _get_metric_ylabel(self, column, rw=None):
        metric = 'NAPFD'
        ylabel = metric
        if column == 'cost':
            metric = 'APFDc'
            ylabel = metric
        elif column == 'ttf':
            metric = 'RFTC'
            ylabel = 'Rank of the Failing Test Cases'
        elif column == 'prioritization_time':
            metric = 'PrioritizationTime'
            ylabel = 'Prioritization Time (sec.)'
        elif  column == 'ttf_duration':
            metric = 'TimeSpentFail'
            ylabel = 'Time Spent to Fail'
        elif column == "rewards":
            metric = rw
            ylabel = rw
        elif column == "scheduled_testcases":
            metric = 'ScheduledTestCases'
            ylabel = 'Scheduled Test Cases'
        elif column == "unscheduled_testcases":
            metric = 'UnscheduledTestCases'
            ylabel = 'Unscheduled Test Cases'
        elif column == "prioritized_test_set":
            metric = 'PrioritizationOrder'
            ylabel = 'Prioritization Order'
        elif column == "testcaseset_percentage_scheduled":
            metric = 'PercentageScheduled'
            ylabel = 'Percentage of Test Case Scheduled'
        elif column == 'fde':
            metric = 'FDE'
            label = 'Faults Detection-Effectiveness'

        return metric, ylabel

    def _get_rewards(self):
        if len(self.datasets.keys()) > 0:
            return self.datasets[list(self.datasets.keys())[0]]['reward_function'].unique()
        else:
            return []

    def _get_policies(self):
        if len(self.datasets.keys()) > 0:
            return self.datasets[list(self.datasets.keys())[0]]['policy'].unique()
        else:
            return []

    def print_mean(self, df, column, direction='max'):
        mean = df.groupby(['policy'], as_index=False).agg(
            {column: ['mean', 'std', 'max', 'min']})
        mean.columns = ['policy', 'mean', 'std', 'max', 'min']

        # sort_df(mean)

        # Round values (to be used in the article)
        mean = mean.round({'mean': 4, 'std': 3, 'max': 4, 'min': 4})
        mean = mean.infer_objects()

        bestp = mean.loc[mean['mean'].idxmax() if direction ==
                         'max' else mean['mean'].idxmin()]

        val = 'Highest' if direction == 'max' else 'Lowest'

        # mean.sort_values(by=['mean'], inplace=True, ascending=[False])

        print(f"\n{val} Value found by {bestp['policy']}: {bestp['mean']:.4f}")
        print("\nMeans:")
        print(mean)

        return mean, bestp['policy']

    def print_mean_latex(self, x, column):
        policies = self._get_policies()

        print(*policies, sep="\t")
        cont = len(policies)

        for policy in policies:
            df_temp = x[x.policy == policy]
            print(f"{df_temp[column].mean():.4f} $\pm$ {df_temp[column].std():.3f} ", end="")

            cont -= 1
            if cont != 0:
                print("& ", end="")
        print()

    def _transpose_df(self, df, column='policy'):
        df_tras = df.copy()
        df_tras.index = df_tras[column]
        return df_tras.transpose()

    def _define_axies(self, ax, rw, tr, column, ylabel=None, force_ylabel=False):
        metric, ylabel_temp = self._get_metric_ylabel(column, rw)

        ax.set_xlabel('CI Cycle', fontsize=self.font_size_plots)
        if force_ylabel:
            ax.set_ylabel(ylabel if ylabel is not None else ylabel_temp,
                          fontsize=self.font_size_plots)
        else:
            ax.set_ylabel(ylabel + " " + metric if ylabel is not None else metric,
                          fontsize=self.font_size_plots)
        ax.set_title(f"Time Budget: {tr}%", fontsize=self.font_size_plots)

    def _plot_accumulative(self, df, rw, ax, tr, column='fitness'):
        df = df[['step', 'policy', column]] if rw is None else df[
            ['step', 'policy', column]][df.reward_function == rw]
        df.groupby(['step', 'policy']).mean()[
            column].unstack().cumsum().plot(ax=ax, linewidth=2.5)

        self._define_axies(ax, rw, tr, column, ylabel='Accumulative')

    def plot_accumulative(self, figname, column='fitness'):
        rewards = self._get_rewards()
        policies = self._get_policies()

        if 'Deterministic' in list(policies):
            fig, axes = plt.subplots(ncols=len(self.datasets.keys()), sharex=True, sharey=True,
                                     figsize=(int(4.3 * 3 * (len(policies) / 3)), 20))
            # Todo try a generic way
            (ax1, ax2, ax3) = axes

            for df_k, tr, ax in zip(self.datasets.keys(), self.sched_time_ratio_names, [ax1, ax2, ax3]):
                self._plot_accumulative(
                    self.datasets[df_k], None, ax, tr, column)

            handles, labels = ax1.get_legend_handles_labels()
            # lgd = fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.1), loc='lower right')
            lgd = ax1.legend(handles, labels, bbox_to_anchor=(-0.02, 1.05, 1, 0.2), loc='lower left',
                             ncol=len(policies))

            # ax1.get_legend().remove()
            ax2.get_legend().remove()
            ax3.get_legend().remove()

            plt.tight_layout()
            plt.savefig(f"{self.figure_dir}/{figname}.pdf", bbox_inches='tight')
            plt.cla()
            plt.close(fig)
        else:
            for rw in rewards:
                fig, axes = plt.subplots(ncols=len(self.datasets.keys()), sharex=True, sharey=True,
                                         figsize=(int(4.3 * 3 * (len(policies) / 3)), 20))
                # Todo try a generic way
                (ax1, ax2, ax3) = axes

                for df_k, tr, ax in zip(self.datasets.keys(), self.sched_time_ratio_names, [ax1, ax2, ax3]):
                    self._plot_accumulative(
                        self.datasets[df_k], rw, ax, tr, column)

                handles, labels = ax1.get_legend_handles_labels()
                # lgd = fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.1), loc='lower right')
                lgd = ax1.legend(handles, labels, bbox_to_anchor=(-0.02, 1.05, 1, 0.2), loc='lower left',
                                 ncol=len(policies))

                # ax1.get_legend().remove()
                ax2.get_legend().remove()
                ax3.get_legend().remove()

                plt.tight_layout()
                plt.savefig(f"{self.figure_dir}/{figname}_{self.reward_names[rw]}.pdf", bbox_inches='tight')
                plt.cla()
                plt.close(fig)

    def _visualize_boxplot_unique(self, df, ax, tr, rewardfuns, policies, ylabel, column='fitness'):
        if column == 'ttf':
            df = df[df.ttf > 0]

        columns = ['experiment', 'reward_function'] + list(policies)
        df_temp = pd.DataFrame(columns=columns)

        # prepare the data
        for rw in rewardfuns:
            # Select the values
            x = df[['policy', 'experiment', column]][df.reward_function == rw]

            # Get the mean of fitness (or cost) in each experiment
            # Then, we transform the policy values to columns
            x = x.groupby(['experiment', 'policy'], as_index=False).agg(
                {column: np.mean}).pivot(index='experiment', columns='policy', values=[column])

            # Get only the main columns and discard the "ugly" names
            x.columns = [j for i, j in x.columns]
            x.reset_index(inplace=True)

            # Update with the actual reward function
            x['reward_function'] = rw

            # Merge the data
            df_temp = df_temp.append(x, sort=False)

        # policies = get_policy_order(policies)

        # transform the data
        dd = pd.melt(df_temp, id_vars=[
            'reward_function'], value_vars=policies, var_name='ci')

        b = sns.boxplot(x='reward_function', y='value',
                        data=dd, hue='ci', ax=ax)
        b.tick_params(labelsize=self.font_size_plots)

        # Line separating the reward functions
        for i in range(len(df['reward_function'].unique()) - 1):
            ax.vlines(i + .5,
                      df_temp[policies].min(axis=1),
                      df_temp[policies].max(axis=1),
                      linestyles='solid',
                      colors='gray',
                      alpha=0.3)

        ax.set_xlabel('')
        ax.set_ylabel(f'{ylabel}', fontsize=self.font_size_plots)
        ax.set_title(f"Time Budget: {tr}%", fontsize=self.font_size_plots)

    def visualize_boxplot_unique(self, column='fitness'):
        metric, ylabel = self._get_metric_ylabel(column)

        rewards = self._get_rewards()
        policies = self._get_policies()

        fig, axes = plt.subplots(
            ncols=len(self.datasets.keys()), sharex=True, sharey=True, figsize=(35, 15))
        # Todo try a generic way
        (ax1, ax2, ax3) = axes

        for df_k, tr, ax in zip(self.datasets.keys(), self.sched_time_ratio_names, [ax1, ax2, ax3]):
            self._visualize_boxplot_unique(
                self.datasets[df_k], ax, tr, rewards, policies, ylabel, column)

        handles, labels = ax1.get_legend_handles_labels()
        lgd = ax1.legend(handles, labels, bbox_to_anchor=(-0.009,
                                                          1.05, 1, 0.2), loc='lower left', ncol=len(policies))

        ax2.get_legend().remove()
        ax3.get_legend().remove()

        plt.grid(True, ls='--', axis='y')
        plt.tight_layout()
        plt.savefig(f"{self.figure_dir}/{metric}.pdf", bbox_inches='tight')
        plt.cla()
        plt.close(fig)

    def _plot_lines(self, df, rw, ax, tr, column='fitness'):
        df = df[['step', 'policy', column]] if rw is None else df[
            ['step', 'policy', column]][df.reward_function == rw]
        df.groupby(['step', 'policy']).mean()[
            column].unstack().plot(ax=ax, linewidth=2.5)

        self._define_axies(ax, rw, tr, column)

    def plot_lines(self, figname, column='fitness'):
        rewards = self._get_rewards()
        policies = self._get_policies()

        if 'Deterministic' in list(policies):
            fig, axes = plt.subplots(nrows=len(self.datasets.keys()), sharex=True, sharey=True,
                                     figsize=(int(4.3 * 3 * (len(policies) / 3)), 20))
            # Todo try a generic way
            (ax1, ax2, ax3) = axes

            for df_k, tr, ax in zip(self.datasets.keys(), self.sched_time_ratio_names, [ax1, ax2, ax3]):
                self._plot_lines(self.datasets[df_k], None, ax, tr, column)

            handles, labels = ax1.get_legend_handles_labels()
            lgd = ax1.legend(handles, labels, bbox_to_anchor=(-0.005, 1.05, 1, 0.2), loc='lower left',
                             ncol=len(policies))

            ax2.get_legend().remove()
            ax3.get_legend().remove()

            plt.tight_layout()
            plt.savefig(f"{self.figure_dir}/{figname}.pdf", bbox_inches='tight')
            plt.cla()
            plt.close(fig)
        else:
            for rw in rewards:
                fig, axes = plt.subplots(
                    nrows=len(self.datasets.keys()), sharex=True, sharey=True,
                    figsize=(int(4.3 * 3 * (len(policies) / 3)), 20))
                # Todo try a generic way
                (ax1, ax2, ax3) = axes

                for df_k, tr, ax in zip(self.datasets.keys(), self.sched_time_ratio_names, [ax1, ax2, ax3]):
                    self._plot_lines(self.datasets[df_k], rw, ax, tr, column)

                handles, labels = ax1.get_legend_handles_labels()
                lgd = ax1.legend(handles, labels, bbox_to_anchor=(-0.005, 1.05, 1, 0.2), loc='lower left',
                                 ncol=len(policies))

                ax2.get_legend().remove()
                ax3.get_legend().remove()

                plt.tight_layout()
                plt.savefig(f"{self.figure_dir}/{figname}_{self.reward_names[rw]}.pdf", bbox_inches='tight')
                plt.cla()
                plt.close(fig)

    def _visualize_ntr(self, df, tr, ax, total_time_spent):
        policies = self._get_policies()

        # Only the commits which failed
        x = df[['experiment', 'policy', 'time_reduction']
               ][(df.avg_precision == 123)]

        df_ntr = pd.DataFrame(columns=['experiment', 'policy', 'n_reduction'])

        row = [tr]
        means = []
        for policy in policies:
            df_ntr_temp = x[x.policy == policy]

            # sum all differences (time_reduction column) in all cycles for
            # each experiment
            df_ntr_temp = df_ntr_temp.groupby(['experiment'], as_index=False).agg({
                'time_reduction': np.sum})

            # Evaluate for each experiment
            df_ntr_temp['n_reduction'] = df_ntr_temp['time_reduction'].apply(
                lambda x: x / (total_time_spent))

            df_ntr_temp['policy'] = policy

            df_ntr_temp = df_ntr_temp[['experiment', 'policy', 'n_reduction']]

            df_ntr = df_ntr.append(df_ntr_temp)

            means.append(df_ntr_temp['n_reduction'].mean())
            if means[-1] <= 0:
                means[-1] = 0.0
            text = f"{means[-1]:.4f} $\pm$ {df_ntr_temp['n_reduction'].std():.3f}"

            row.append(text)

        if len(df_ntr) > 0:
            df_ntr.sort_values(by=['policy', 'experiment'], inplace=True)
            sns.boxplot(x='policy', y='n_reduction', data=df_ntr, ax=ax)
            ax.set_xlabel('')
            ax.set_ylabel('Normalized Time Reduction' if tr ==
                          '10' else '')  # Workaround
            ax.set_title(f"Required Time: {tr}%")
            ax.set_xticklabels(textwrap.fill(x.get_text(), MAX_XTICK_WIDTH)
                               for x in ax.get_xticklabels())

        best_idx = [i for i, x in enumerate(means) if x == max(means)]

        if len(best_idx) == 1:
            best_i = best_idx[0]
            row[best_i + 1] = f"\\cellbold{{{row[best_i + 1]}}}"
        else:
            for best_i in best_idx:
                row[best_i + 1] = f"\\cellgray{{{row[best_i + 1]}}}"

        return row

    def visualize_ntr(self):
        # Total time spent in each Cycle
        total_time_spent = self.df_system['duration'].sum()
        rewards = self._get_rewards()
        policies = self._get_policies()

        stat_columns = ['TimeBudget'] + list(policies)
        df_stats = pd.DataFrame(columns=stat_columns)

        fig, axes = plt.subplots(
            ncols=len(self.datasets.keys()), sharex=True, sharey=True, figsize=(int(8.3 * 3 * (len(policies) / 3)), 8))
        (ax1, ax2, ax3) = axes

        for df_k, tr, ax in zip(self.datasets.keys(), self.sched_time_ratio_names, [ax1, ax2, ax3]):
            df = self.datasets[df_k]
            row = self._visualize_ntr(df, tr, ax, total_time_spent)
            df_stats = df_stats.append(
                pd.DataFrame([row], columns=stat_columns))

        plt.tight_layout()
        plt.savefig(f"{self.figure_dir}/NTR.pdf", bbox_inches='tight')
        plt.cla()
        plt.close(fig)

        df_stats['Metric'] = 'NTR'

        return df_stats

    def _visualize_ttf_duration(self, df, tr, ax):
        policies = self._get_policies()

        # Only the commits which failed
        x = df[['experiment', 'policy', 'ttf_duration']
               ][(df.avg_precision == 123)]

        df_ntr = pd.DataFrame(columns=['experiment', 'policy', 'diff'])

        row = [tr]
        means = []
        for policy in policies:
            df_ntr_temp = x[x.policy == policy]

            # sum all differences (time_reduction column) in all cycles for
            # each experiment
            df_ntr_temp = df_ntr_temp.groupby(['experiment'], as_index=False).agg({
                'ttf_duration': np.sum})

            df_ntr_temp['policy'] = policy

            df_ntr_temp = df_ntr_temp[['experiment', 'policy', 'ttf_duration']]

            df_ntr = df_ntr.append(df_ntr_temp)

            means.append(df_ntr_temp['ttf_duration'].mean())
            if means[-1] <= 0:
                means[-1] = 0.0
            text = f"{means[-1]:.4f} $\pm$ {df_ntr_temp['ttf_duration'].std():.3f}"

            row.append(text)

        if len(df_ntr) > 0:
            df_ntr.sort_values(by=['policy', 'experiment'], inplace=True)
            sns.boxplot(x='policy', y='ttf_duration', data=df_ntr, ax=ax)
            ax.set_xlabel('')
            ax.set_ylabel('Time Spent to Fail' if tr ==
                          '10' else '')  # Workaround
            ax.set_title(f"Required Time: {tr}%")
            ax.set_xticklabels(textwrap.fill(x.get_text(), MAX_XTICK_WIDTH)
                               for x in ax.get_xticklabels())

        best_idx = [i for i, x in enumerate(means) if x == min(means)]

        if len(best_idx) == 1:
            best_i = best_idx[0]
            row[best_i + 1] = f"\\cellbold{{{row[best_i + 1]}}}"
        else:
            for best_i in best_idx:
                row[best_i + 1] = f"\\cellgray{{{row[best_i + 1]}}}"

        return row

    def visualize_ttf_duration(self):
        policies = self._get_policies()

        stat_columns = ['TimeBudget'] + list(policies)
        df_stats = pd.DataFrame(columns=stat_columns)

        fig, axes = plt.subplots(
            ncols=len(self.datasets.keys()), sharex=True, sharey=True, figsize=(int(8.3 * 3 * (len(policies) / 3)), 8))
        (ax1, ax2, ax3) = axes

        for df_k, tr, ax in zip(self.datasets.keys(), self.sched_time_ratio_names, [ax1, ax2, ax3]):
            df = self.datasets[df_k]
            row = self._visualize_ttf_duration(df, tr, ax)
            df_stats = df_stats.append(
                pd.DataFrame([row], columns=stat_columns))

        plt.tight_layout()
        plt.savefig(f"{self.figure_dir}/TTF_DURATION.pdf", bbox_inches='tight')
        plt.cla()
        plt.close(fig)

        df_stats['Metric'] = 'TTF_DURATION'

        return df_stats

    def _visualize_test_set_size(self, df, tr, ax, column):
        # Only the commits which failed
        if column == 'testcaseset_percentage_scheduled':
            df_size = df[['experiment', 'policy', 'prioritization_order', 'scheduled_testcases']][
                (df.avg_precision == 123)]
            df_size["test_size"] = df_size.apply(
                lambda x: len(x['scheduled_testcases']) * 100 / len(x['prioritization_order']), axis=1)
        else:
            df_size = df[['experiment', 'policy', column]][
                (df.avg_precision == 123)]
            # Calculate the test set size based on the list inside each cell
            df_size["test_size"] = df_size[column].apply(lambda x: len(x))

        df_size_temp = df_size.groupby(
            ['experiment', 'policy'], as_index=False).agg({'test_size': np.mean})

        df_size_temp.sort_values(by=['policy', 'experiment'], inplace=True)
        sns.boxplot(x='policy', y='test_size', data=df_size_temp, ax=ax)

        self._define_axies(ax, None, tr, column, force_ylabel=True,
                           ylabel='Test Set Size' if column != "testcaseset_percentage_Scheduled" else None)

        df_new = df_size.groupby(['policy'], as_index=False).agg(
            {'test_size': [np.mean, np.std]})
        df_new.columns = ['policy', 'test_size_mean', 'test_size_std']

        df_new['output'] = df_new.apply(lambda x: f"{x['test_size_mean']:.4f} $\\pm$ {x['test_size_std']:.3f}", axis=1)

        # The best values is the max number of test cases that I can test
        # during a time budget
        best = df_new['test_size_mean'].max()
        contains_equivalent = len(df_new[df_new['test_size_mean'] == best]) > 1

        def get_config_latex(row, best, contains_equivalent):
            """
            Latex commands used:
            - Best algorithm: \newcommand{\cellgray}[1]{\cellcolor{gray!30}{#1}}
            - Equivalent to the best one: \newcommand{\cellbold}[1]{\cellcolor{gray!30}{\textbf{#1}}}
            """
            if contains_equivalent and row['test_size_mean'] == best:
                return f"\\cellgray{{{row['output']}}}"
            if row['test_size_mean'] == best:
                return f"\\cellbold{{{row['output']}}}"

            return row['output']

        df_new['latex_format'] = df_new.apply(lambda row: get_config_latex(row, best, contains_equivalent),
                                              axis=1)

        # Return only the values
        return self._transpose_df(df_new[['policy', 'latex_format']]).values[1]

    def visualize_test_set_size(self, column):
        metric, ylabel = self._get_metric_ylabel(column, None)

        policies = self._get_policies()

        stat_columns = ['TimeBudget'] + list(policies)
        df_stats = pd.DataFrame(columns=stat_columns)

        fig, axes = plt.subplots(
            ncols=len(self.datasets.keys()), sharex=True, sharey=True, figsize=(int(8.3 * 3 * (len(policies) / 3)), 8))
        (ax1, ax2, ax3) = axes

        for df_k, tr, ax in zip(self.datasets.keys(), self.sched_time_ratio_names, [ax1, ax2, ax3]):
            df = self.datasets[df_k]
            row = [tr] + \
                list(self._visualize_test_set_size(df, tr, ax, column))
            df_stats = df_stats.append(
                pd.DataFrame([row], columns=stat_columns))

        plt.tight_layout()
        plt.savefig(f"{self.figure_dir}/TestSetSize_{metric}.pdf", bbox_inches='tight')
        plt.cla()
        plt.close(fig)

        df_stats['Metric'] = 'TestSetSize' + metric

        return df_stats

    def _visualize_duration(self, df):
        dd = df[['policy', 'prioritization_time']]
        # sort_df(dd)
        self.print_mean(dd, 'prioritization_time', direction='min')
        self.print_mean_latex(dd, 'prioritization_time')

    def visualize_duration(self):
        # print(f"\n\n||||||||||||||||||||||||||||||| PRIORITIZATION DURATION
        # |||||||||||||||||||||||||||||||\n")
        rewards = self._get_rewards()

        for rw in rewards:
            print(f"\n\n======{rw}======")
            for df_k, tr in zip(self.datasets.keys(), self.sched_time_ratio_names):
                print(f"\nTime Budget {tr}%")
                df = self.datasets[df_k]
                self._visualize_duration(df[df.reward_function == rw])

    def _accumulative_fde(self, df, tr, ax):
        """
        Faults Detection-Effectiveness
        :param df:
        :param tr:
        :param ax:
        :return:
        """
        df_system = self.df_system.copy()
        df_system.sort_values(by=['step'], inplace=True)

        x = df[['step', 'policy', 'missed', 'avg_precision']]

        # Only the commits which failed
        x['total_faults'] = x['step'].apply(
            lambda step: df_system[df_system.step == step]['faults'].sum())

        x['detected'] = x['total_faults'] - x['missed']
        x['fde'] = x['detected'] / x['total_faults']
        # We can have ZERO division, so we fill NA values
        x['fde'].fillna(0, inplace=True)

        df_new = x.copy()

        x.groupby(['step', 'policy']).mean()[
            'fde'].unstack().cumsum().plot(ax=ax, linewidth=2.5)
        self._define_axies(ax, None, tr, 'fde', ylabel='Accumulative')

        # Only the commits which failed
        df_new = df_new[['policy', 'fde']][(x.avg_precision == 123)]
        df_new = df_new.groupby(['policy'], as_index=False).agg(
            {'fde': [np.mean, np.std]})
        df_new.columns = ['policy', 'fde_mean', 'fde_std']

        df_new['output'] = df_new.apply(lambda x: f"{x['fde_mean']:.4f} $\\pm$ {x['fde_std']:.3f}", axis=1)

        # The best values is the max number of test cases that I can test
        # during a time budget
        best = df_new['fde_mean'].max()
        contains_equivalent = len(df_new[df_new['fde_mean'] == best]) > 1

        def get_config_latex(row, best, contains_equivalent):
            """
            Latex commands used:
            - Best algorithm: \newcommand{\cellgray}[1]{\cellcolor{gray!30}{#1}}
            - Equivalent to the best one: \newcommand{\cellbold}[1]{\cellcolor{gray!30}{\textbf{#1}}}
            """
            if contains_equivalent and row['fde_mean'] == best:
                return f"\\cellgray{{{row['output']}}}"

            if row['fde_mean'] == best:
                return f"\\cellbold{{{row['output']}}}"

            return row['output']

        df_new['latex_format'] = df_new.apply(lambda row: get_config_latex(row, best, contains_equivalent),
                                              axis=1)

        # Return only the values
        return self._transpose_df(df_new[['policy', 'latex_format']]).values[1]

    def accumulative_fde(self):
        """
        Faults Detection-Effectiveness
        :return:
        """
        policies = self._get_policies()

        stat_columns = ['TimeBudget'] + list(policies)
        df_stats = pd.DataFrame(columns=stat_columns)

        fig, axes = plt.subplots(
            ncols=len(self.datasets.keys()), sharex=True, sharey=True, figsize=(int(8.3 * 3 * (len(policies) / 3)), 8))
        (ax1, ax2, ax3) = axes

        for df_k, tr, ax in zip(self.datasets.keys(), self.sched_time_ratio_names, [ax1, ax2, ax3]):
            row = [tr] + \
                list(self._accumulative_fde(self.datasets[df_k], tr, ax))
            df_stats = df_stats.append(
                pd.DataFrame([row], columns=stat_columns))

        handles, labels = ax1.get_legend_handles_labels()
        # lgd = fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.1), loc='lower right')
        lgd = ax1.legend(handles, labels, bbox_to_anchor=(-0.02, 1.05, 1, 0.2), loc='lower left',
                         ncol=len(policies))

        ax2.get_legend().remove()
        ax3.get_legend().remove()

        plt.tight_layout()
        plt.savefig(f"{self.figure_dir}/FDE.pdf", bbox_inches='tight')
        plt.cla()
        plt.close(fig)

        df_stats['Metric'] = 'FDE'

        return df_stats

    def _rmse_calculation(self, df, column='fitness'):
        def get_rmse_symbol(mean):
            """
            very near     if RMSE < 0.15
            near          if 0.15 <= RMSE < 0.23
            reasonable    if 0.23 <= RMSE < 0.30
            far           if 0.30 <= RMSE < 0.35
            very far      if 0.35 <= RMSE
            """
            if mean < 0.15:
                # very near
                return "$\\bigstar$"
            elif mean < 0.23:
                # near
                return "$\\blacktriangledown$"
            elif mean < 0.30:
                # reasonable
                return "$\\triangledown$"
            elif mean < 0.35:
                # far
                return "$\\vartriangle$"
            else:
                # very far
                return "$\\blacktriangle$"

        def get_mean_std_rmse(df_rmse, column, n_builds):
            df_f = df_rmse.groupby(['experiment'], as_index=False).agg(
                {column: lambda x: np.sqrt(sum(x) / n_builds)})

            # Get column values and provide a beautiful output
            mean, std = round(df_f[column].mean(), 4), round(
                df_f[column].std(), 4)

            return [mean, std, f"{mean:.4f} $\\pm$ {std:.3f} {get_rmse_symbol(mean)}".strip()]

        def get_config_latex(row, best_rmse, contains_equivalent):
            """
            Latex commands used:
            - Best algorithm: \newcommand{\cellgray}[1]{\cellcolor{gray!30}{#1}}
            - Equivalent to the best one: \newcommand{\cellbold}[1]{\cellcolor{gray!30}{\textbf{#1}}}
            """
            if contains_equivalent and row['mean'] == best_rmse:
                return f"\\cellgray{{{row['output']}}}"

            if row['mean'] == best_rmse:
                return f"\\cellbold{{{row['output']}}}"

            return row['output']

        columns = [column, 'experiment', 'step']

        n_builds = len(df['step'].unique())

        # Get only the required columns
        df = df[['experiment', 'step', 'policy', column]]

        # Orderby to guarantee the right value
        df.sort_values(by=['experiment', 'step'], inplace=True)

        df_rmse = pd.DataFrame(
            columns=['experiment', 'step', 'Deterministic'])

        dt = df.loc[df['policy'] == 'Deterministic', columns]

        df_rmse['Deterministic'] = dt[column]
        df_rmse['experiment'] = dt['experiment']
        df_rmse['step'] = dt['step']

        policies = list(self._get_policies())
        policies.remove('Deterministic')

        for pol in policies:
            df_rmse[pol] = df.loc[df['policy']
                                  == pol, columns][column].tolist()

        df_rmse = df_rmse.reset_index()

        for pol in policies:
            df_rmse[f'RMSE_{pol}'] = df_rmse.apply(lambda x: (x[pol] - x['Deterministic']) ** 2, axis=1)

        df_rmse_rows = []
        for pol in policies:
            rmse = get_mean_std_rmse(df_rmse, f'RMSE_{pol}', n_builds)

            df_rmse_rows.append([pol] + rmse)

        df_rmse_results = pd.DataFrame(
            df_rmse_rows, columns=['policy', 'mean', 'std', 'output'])

        best_rmse = df_rmse_results['mean'].min()
        contains_equivalent = len(
            df_rmse_results[df_rmse_results['mean'] == best_rmse]) > 1

        df_rmse_results['latex_format'] = df_rmse_results.apply(
            lambda row: get_config_latex(row, best_rmse, contains_equivalent), axis=1)

        # Select the main information
        rmse = df_rmse_results[['policy', 'latex_format']]

        # Return only the values
        return self._transpose_df(rmse).values[1]

    def rmse_calculation(self, column='fitness'):
        policies = list(self._get_policies())
        policies.remove('Deterministic')

        rmse_cols = ['TimeBudget'] + policies
        df_rmse = pd.DataFrame(columns=rmse_cols)

        for df_k, tr in zip(self.datasets.keys(), self.sched_time_ratio_names):
            df = self.datasets[df_k]
            row = [tr] + list(self._rmse_calculation(df, column))
            df_rmse = df_rmse.append(pd.DataFrame([row], columns=rmse_cols))

        metric, ylabel = self._get_metric_ylabel(column)

        df_rmse['Metric'] = 'RMSE_' + metric

        return df_rmse

    def _jaccard_similarity_calculation(self, df):
        """
        Jaccard Similarity Coeficient between the scheduled test set by a prioritization algorithm
        and a deterministic one
        :param df: Dataframe that contains the results of the algorithms
        :return:
        """

        def jaccard_similarity(list1, list2):
            s1 = set(list1)
            s2 = set(list2)
            intersection = len(s1.intersection(s2))
            union = len(s1.union(s2))
            return float(intersection) / union

        def get_jaccard_symbol(mean):
            """
            very near     if RMSE < 0.15
            near          if 0.15 <= RMSE < 0.23
            reasonable    if 0.23 <= RMSE < 0.30
            far           if 0.30 <= RMSE < 0.35
            very far      if 0.35 <= RMSE
            """
            if mean > 0.90:
                # very near
                return "$\\bigstar$"
            elif mean > 0.80:
                # near
                return "$\\blacktriangledown$"
            elif mean > 0.70:
                # reasonable
                return "$\\triangledown$"
            elif mean > 0.60:
                # far
                return "$\\vartriangle$"
            else:
                # very far
                return "$\\blacktriangle$"

        def get_mean_std_jaccard(df_res, column, n_builds):
            df_f = df_res.groupby(['experiment'], as_index=False).agg(
                {column: lambda x: np.sqrt(sum(x) / n_builds)})

            # Get column values and provide a beautiful output
            mean, std = round(df_f[column].mean(), 4), round(
                df_f[column].std(), 4)

            return [mean, std, f"{mean:.4f} $\\pm$ {std:.3f} {get_jaccard_symbol(mean)}".strip()]

        # Used to calculate the mean across the commits
        n_builds = len(df['step'].unique())

        # Get only the required columns
        df = df[['experiment', 'step', 'policy', 'scheduled_testcases']]

        # Orderby to guarantee the right value
        df.sort_values(by=['experiment', 'step'], inplace=True)

        df_jaccard = pd.DataFrame(
            columns=['experiment', 'step', 'Deterministic'])

        columns = ['scheduled_testcases', 'experiment', 'step']

        # Get values from Deterministic
        dt = df.loc[df['policy'] == 'Deterministic', columns]

        # Here we can define the main values for each column
        df_jaccard['Deterministic'] = dt['scheduled_testcases']
        df_jaccard['experiment'] = dt['experiment']
        df_jaccard['step'] = dt['step']

        policies = list(self._get_policies())
        policies.remove('Deterministic')

        # Now, for each remain policies we fill them column
        for pol in policies:
            df_jaccard[pol] = df.loc[df['policy'] == pol, columns][
                'scheduled_testcases'].tolist()

        # Organize the dataframe
        df_jaccard = df_jaccard.reset_index()

        # Calculate the jaccard similarity score for each policy
        for pol in policies:
            # df["test_size"] = df_size[column].apply(lambda x: len(x))
            df_jaccard[f'Jaccard_{pol}'] = df_jaccard.apply(lambda x: jaccard_similarity(x[pol], x['Deterministic']),
                                                            axis=1)

        rows = [[pol] + get_mean_std_jaccard(df_jaccard, f'Jaccard_{pol}', n_builds) for pol in policies]

        df_results = pd.DataFrame(
            rows, columns=['policy', 'mean', 'std', 'output'])

        best = df_results['mean'].max()
        contains_equivalent = len(df_results[df_results['mean'] == best]) > 1

        def get_config_latex(row, best, contains_equivalent):
            """
            Latex commands used:
            - Best algorithm: \newcommand{\cellgray}[1]{\cellcolor{gray!30}{#1}}
            - Equivalent to the best one: \newcommand{\cellbold}[1]{\cellcolor{gray!30}{\textbf{#1}}}
            """
            if contains_equivalent and row['mean'] == best:
                return f"\\cellgray{{{row['output']}}}"

            if row['mean'] == best:
                return f"\\cellbold{{{row['output']}}}"

            return row['output']

        df_results['latex_format'] = df_results.apply(lambda row: get_config_latex(row, best, contains_equivalent),
                                                      axis=1)

        # Return only the values for the main information
        return self._transpose_df(df_results[['policy', 'latex_format']]).values[1]

    def jaccard_similarity_calculation(self):
        policies = list(self._get_policies())
        policies.remove('Deterministic')

        cols = ['TimeBudget'] + policies
        df_metric = pd.DataFrame(columns=cols)

        for df_k, tr in zip(self.datasets.keys(), self.sched_time_ratio_names):
            df = self.datasets[df_k]
            row = [tr] + list(self._jaccard_similarity_calculation(df))
            df_metric = df_metric.append(pd.DataFrame([row], columns=cols))

        df_metric['Metric'] = 'Jaccard'

        return df_metric

    def _statistical_test_kruskal(self, df, ax, column):
        if column == 'ttf':
            df = df[df.ttf > 0]

        if (len(df)) > 0:
            # Get the mean of fitness in each experiment
            x = df[['experiment', 'policy', column]]

            policies = self._get_policies()
            # Some policies can be not have values (low performance)
            diff_pol = list(set(policies) - set(x['policy'].unique()))

            x = x.groupby(['experiment', 'policy'], as_index=False).agg(
                {column: np.mean})

            # Remove unnecessary columns
            x = x[['policy', column]]

            mean, best = self.print_mean(x, column, 'min' if column in [
                'ttf', 'prioritization_time'] else 'max')
            mean['eff_symbol'] = " "

            posthoc_df = None
            all_equivalent = False

            try:
                k = kruskal_wallis(x, column, 'policy')
                kruskal, posthoc = k.apply(ax)
                print(f"\n{kruskal}")  # Kruskal results

                all_equivalent = 'p-unc' not in kruskal.columns or kruskal[
                    'p-unc'][0] >= 0.05

                if posthoc is not None:
                    print("\n--- POST-HOC TESTS ---")
                    print("\np-values:")
                    print(posthoc[0])

                    # Get the posthoc
                    df_eff = vargha_delaney.reduce(posthoc[1], best)

                    print(df_eff)

                    def get_eff_symbol(x, best, df_eff):
                        if x['policy'] == best:
                            return "$\\bigstar$"
                        elif len(df_eff.loc[df_eff.compared_with == x['policy'], 'effect_size_symbol'].values) > 0:
                            return df_eff.loc[df_eff.compared_with == x['policy'], 'effect_size_symbol'].values[0]
                        else:
                            return df_eff.loc[df_eff.base == x['policy'], 'effect_size_symbol'].values[0]

                    mean['eff_symbol'] = mean.apply(
                        lambda x: get_eff_symbol(x, best, df_eff), axis=1)

                    # Parse the posthoc to a dataframe in R because allows us
                    # to parse to pandas in Py
                    ro.r.assign('posthoc', posthoc[0])
                    ro.r('posthoc_table <- t(as.matrix(posthoc$p.value))')
                    ro.r('df_posthoc <- as.data.frame(t(posthoc_table))')

                    # Convert the dataframe from R to pandas
                    with localconverter(ro.default_converter + pandas2ri.converter):
                        posthoc_df = ro.conversion.rpy2py(ro.r('df_posthoc'))

            except Exception as e:
                print("\nError in statistical test:", exception_to_string(e))

                # Concat the values to a unique columns
            mean['avg_std_effect'] = mean.apply(
                lambda row: f"{row['mean']:.4f} $\\pm$ {row['std']:.4f} {row['eff_symbol']}".strip(), axis=1)

            def get_config_latex(row, best, posthoc_df, all_equivalent):
                """
                Latex commands used:
                - Best algorithm: \newcommand{\cellgray}[1]{\cellcolor{gray!30}{#1}}
                - Equivalent to the best one: \newcommand{\cellbold}[1]{\cellcolor{gray!30}{\textbf{#1}}}
                """
                current_name = row['policy']

                if all_equivalent:
                    return f"\\cellgray{{{row['avg_std_effect']}}}"

                if row['policy'] == best:
                    return f"\\cellbold{{{row['avg_std_effect']}}}"

                is_equivalent = False

                # If the posthoc was applied
                if posthoc_df is not None:
                    if best in posthoc_df.columns and current_name in posthoc_df.index and not np.isnan(
                            posthoc_df.loc[current_name][best]):
                        # They are equivalent
                        is_equivalent = posthoc_df.loc[
                            current_name][best] >= 0.05
                    elif current_name in posthoc_df.columns and best in posthoc_df.index and not np.isnan(
                            posthoc_df.loc[best][current_name]):
                        # They are equivalent
                        is_equivalent = posthoc_df.loc[
                            best][current_name] >= 0.05
                    else:
                        raise Exception(
                            "Problem found when we tried to find the post-hoc p-value")

                if is_equivalent:
                    return f"\\cellgray{{{row['avg_std_effect']}}}"

                return row['avg_std_effect']

            # Insert the latex commands
            mean['latex_format'] = mean.apply(lambda row: get_config_latex(
                row, best, posthoc_df, all_equivalent), axis=1)

            # Select the main information
            mean = mean[['policy', 'latex_format']]

            mean_trans = mean.copy()
            mean_trans.index = mean['policy']
            mean_trans = mean_trans.transpose()

            if len(diff_pol) > 0:
                # We remove the value from the policies that do not have result
                for dp in diff_pol:
                    mean_trans[dp] = '-'

            # Return only the values
            return mean_trans.values[1]
        else:
            return None

    def statistical_test_kruskal(self, column='fitness'):
        metric, ylabel = self._get_metric_ylabel(column)

        print(
            f"\n\n\n\n||||||||||||||||||||||||||||||| STATISTICAL TEST - KRUSKAL WALLIS - {metric} |||||||||||||||||||||||||||||||\n")

        rewards = self._get_rewards()
        policies = self._get_policies()

        stat_columns = ['TimeBudget'] + list(policies)
        df_stats = pd.DataFrame(columns=stat_columns)

        fig, axes = plt.subplots(
            ncols=len(self.datasets.keys()), sharex=True, sharey=True, figsize=(int(8.3 * 3 * (len(policies) / 3)), 8))
        (ax1, ax2, ax3) = axes

        for df_k, tr, ax in zip(self.datasets.keys(), self.sched_time_ratio_names, [ax1, ax2, ax3]):
            print(f"~~~~ Time Budget {tr}% ~~~~")

            row = self._statistical_test_kruskal(
                self.datasets[df_k], ax, column)

            if row is not None:
                row = np.insert(row, 0, tr)
                df_stats = df_stats.append(
                    pd.DataFrame([row], columns=stat_columns))

                ax.set_title(f"Time Budget: {tr}%")
                ax.set_ylabel(ylabel if tr == '10' else '')  # Workaround
                ax.set_xticklabels(textwrap.fill(x.get_text(), MAX_XTICK_WIDTH)
                                   for x in ax.get_xticklabels())

        if len(df_stats) > 0:
            plt.tight_layout()
            plt.savefig(f"{self.figure_dir}/{metric}_Kruskal.pdf", bbox_inches='tight')
            plt.cla()
            plt.close(fig)

        df_stats['Metric'] = metric

        return df_stats