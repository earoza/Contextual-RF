import os

import pandas as pd

from crffortcpci.utils.features_utils import FEATURES_GROUPS, DEFAULT_PREVIOUS_BUILD

INDUSTRIAL_DATASETS = ['iofrol', 'paintcontrol',
                       'gsdtsr', 'lexisnexis', 'libssh@libssh-mirror', 'core@dune-common']

pd.options.mode.chained_assignment = None  # default='warn'

from crffortcpci.utils.experiment_utils import is_industrial_dataset


class VirtualScenario(object):
    def __init__(self, available_time, testcases, history, build_id, total_build_duration):
        self.available_time = available_time
        self.testcases = testcases
        self.history = history
        self.build_id = build_id
        self.total_build_duration = total_build_duration
        self.reset()

    def reset(self):
        # Reset the priorities
        for row in self.get_testcases():
            row['CalcPrio'] = 0

    def get_available_time(self):
        return self.available_time

    def get_total_build_duration(self):
        return self.total_build_duration

    def get_testcases(self):
        """
        :return: a pandas DataFrame
        """
        return self.testcases

    def get_testcases_names(self):
        return [row['Name'] for row in self.get_testcases()]

    def get_history(self):
        return self.history


class IndustrialDatasetScenarioProvider():
    """
    Scenario provider to process CSV files for experimental evaluation.
    Required columns are `self.tc_fieldnames`
    """

    def __init__(self, tcfile, sched_time_ratio=0.5, feature_group=None):

        INDUSTRIAL_DATASETS = ['iofrol', 'paintcontrol', 'gsdtsr', 'lexisnexis']
        self.feature_group = feature_group
        self.name = os.path.split(os.path.dirname(tcfile))[1]

        self.tcdf = pd.read_csv(tcfile, sep=';', parse_dates=['LastRun'])

        # To avoid duplicated test cases in the same commit (If the dataset does not consider 1 Tc per commit)
        columns_id = ['BuildId', 'Name']
        self.tcdf = self.tcdf.drop_duplicates(subset=columns_id, keep="last")

        self.tcdf["Duration"] = self.tcdf["Duration"].apply(
            lambda x: float(x.replace(',', '')) if type(x) == str else x)

        self.build = 0
        self.max_builds = max(self.tcdf.BuildId)
        self.scenario = None
        self.avail_time_ratio = sched_time_ratio
        self.total_build_duration = 0

        if self.name in INDUSTRIAL_DATASETS:
            self.tc_fieldnames = ['BuildId', 'Name', 'Duration', 'CalcPrio', 'LastRun', 'LastResults', 'Verdict']
        else:
            self.tc_fieldnames = ['BuildId', 'Name', 'Duration', 'CalcPrio', 'LastRun', 'NumRan', 'NumErrors',
                                  'Verdict']

        # self.tc_fieldnames = ['BuildId', 'Name', 'Duration', 'CalcPrio', 'LastRun','LastResults', 'Verdict']

    def __str__(self):
        return self.name

    def last_build(self, build):
        self.build = build

    def get_is_industrial_dataset(self):
        return is_industrial_dataset(self.name)

    def get(self):
        """
        This function is called when the __next__ function is called.
        In this function the data is "separated" by builds. Each next build is returned.
        :return:
        """
        self.build += 1

        # Stop when reaches the max build
        if self.build > self.max_builds:
            self.scenario = None
            return None

        # Select the data for the current build with all columns
        builddf = self.tcdf.loc[self.tcdf.BuildId == self.build]
        features = FEATURES_GROUPS[self.feature_group]
        fields = self.tc_fieldnames + features
        # Get only the need fields and convert the solutions to a list of dict
        seltc = builddf[fields]

        # Get the history until previous build
        history = self.tcdf.loc[self.tcdf.BuildId < self.build]
        history = history[fields]

        self.total_build_duration = builddf['Duration'].sum()
        total_time = self.total_build_duration * self.avail_time_ratio

        # This test set is a "scenario" that must be evaluated.
        self.scenario = VirtualScenario(testcases=seltc,
                                        history=history,
                                        available_time=total_time,
                                        build_id=self.build,
                                        total_build_duration=self.total_build_duration)

        return self.scenario

    # Generator functions
    def __iter__(self):
        return self

    def __next__(self):
        sc = self.get()

        if sc is None:
            raise StopIteration()

        return sc


class VirtualContextScenario(VirtualScenario):
    def __init__(self, available_time, testcases, history, build_id, total_build_duration,
                 feature_group, features):
        super().__init__(available_time, testcases, history, build_id, total_build_duration)
        self.feature_group = feature_group
        self.features = features

    def reset(self):
        super().reset()

    def get_feature_group(self):
        return self.feature_group

    def get_features(self):
        return self.features


class IndustrialDatasetContextScenarioProvider(IndustrialDatasetScenarioProvider):
    """
    Scenario provider to process CSV files for experimental evaluation.
    Required columns are `self.tc_fieldnames`
    """

    def __init__(self, tcfile, feature_group, sched_time_ratio=0.5):

        super().__init__(tcfile, sched_time_ratio, feature_group)

        self.feature_group = feature_group

        # List of columns used as features
        self.features = FEATURES_GROUPS[feature_group]

    def __str__(self):
        return self.name

    def last_build(self, build):
        self.build = build

    def get_context(self, builddf):
        non_features = list(set(self.tc_fieldnames) - set(self.features))

        # Now, we will get the features from previous build
        if self.build == 1:
            # If we are in the first build, we do not have previous build
            # We create an empty dataframe
            df = pd.DataFrame(columns=non_features + self.features)

            # Get the information for non features
            df[non_features] = builddf[non_features]

            # So we fill all features to start with a default value equal 1
            df[self.features] = 1

            return df
        else:
            # Features from the current build
            current_build_features = list(set(self.features) - set(DEFAULT_PREVIOUS_BUILD))
            df = builddf[non_features + current_build_features]

            # We have previous build, lets to get all the previous features for each test case
            previous_build = self.tcdf.loc[self.tcdf.BuildId == self.build - 1, non_features + DEFAULT_PREVIOUS_BUILD]

            # Remove possible duplicated values
            previous_build = previous_build.loc[:, ~previous_build.columns.duplicated()]

            # Merge the data from current build with the data which we obtain from previous build
            # (only with features chosen)
            previous_build_features = [x for x in DEFAULT_PREVIOUS_BUILD if x in self.features]

            if len(previous_build_features) > 0:
                # In the case if we do not have an of then chosen
                df = df.merge(previous_build[non_features + previous_build_features], on=non_features, how='left')

            # Fill the NA values for each feature
            # This is done when we have a new test case and the tests were not execute yet
            # (If the feature chosen is one of the features that we have information only after the tests are executed)
            for feature in previous_build_features:
                # We fill it with the mean from previous build
                df[feature].fillna((previous_build[feature].mean()), inplace=True)

            return df

    def get(self):
        """
        This function is called when the __next__ function is called.
        In this function the data is "separated" by builds. Each next build is returned.
        :return:
        """
        self.build += 1

        # Stop when reaches the max build
        if self.build > self.max_builds:
            self.scenario = None
            return None

        # Get the history until current build along with context features
        history = self.get_context(self.tcdf.loc[self.tcdf.BuildId < self.build])

        # Select the data for the current build with all columns
        builddf = self.tcdf.loc[self.tcdf.BuildId == self.build]

        # Get only the need fields and convert the solutions to a list of dict
        seltc = builddf[self.tc_fieldnames].to_dict(orient='record')

        self.total_build_duration = builddf['Duration'].sum()
        total_time = self.total_build_duration * self.avail_time_ratio

        # This test set is a "scenario" that must be evaluated.
        self.scenario = VirtualContextScenario(testcases=seltc,
                                               available_time=total_time,
                                               build_id=self.build,
                                               history=history,
                                               total_build_duration=self.total_build_duration,
                                               feature_group=self.feature_group,
                                               features=self.features)

        return self.scenario

    # Generator functions
    def __iter__(self):
        return self

    def __next__(self):
        sc = self.get()

        if sc is None:
            raise StopIteration()

        return sc
