import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from crffortcpci.agent import Agent


class RandomForest(object):
    def __init__(self, strategyName="RF_SW", history_size=100, n_estimators=500,
                 random_state=None, criterion='mse'):
        self.strategyName = strategyName
        self.history_size = history_size
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.criterion = criterion

    def __str__(self):
        description = ""
        if self.history_size == 0:
            description = f"RF_Naive_estimators_{self.n_estimators}"
        else:
            description = f"{self.strategyName}_{self.history_size}_estimators_{self.n_estimators}"

        return description

    def _get_rf_model(self):
        model = RandomForestRegressor(n_estimators=self.n_estimators,
                                      criterion=self.criterion,
                                      random_state=self.random_state)

        return model

    # def _split_sequences(self, sequences):
    #     """
    #     split a multivariate sequence into samples
    #     :param sequences: data sequence
    #     :return: X input sequence, y forecast sequence
    #     """
    #     df = DataFrame(sequences)
    #     X = df.shift(periods=1)
    #     X.dropna(inplace=True)
    #     X = X.values
    #
    #     y = df.shift(periods=-1)
    #     y.dropna(inplace=True)
    #     y = y.values
    #
    #     # Return sequences
    #     return np.array(X), np.array(y)

    def _split_contextual_(self, sequences, num_features):
        """
        split a multivariate sequence into samples
        :param sequences: data sequence
        :param num_features: Number of features
        :return: X input sequence, y forecast sequence
        """
        num_features += 1  # we add +1 to consider the verdict (not consider as a feature by the group)
        # sequences.drop(['Features', 'BuildId'], axis=1, inplace=True)
        sequences = sequences.values  # select values from the dataframe as array

        X, y = sequences[:-num_features, :], sequences[num_features:, :]

        # Return sequences
        return np.array(X), np.array(y)

    def choose_all(self, agent: Agent, is_industrial_dataset=False):
        """
        The function prioritizes a test set
        :param agent:
        :param is_industrial_dataset: If the test set is from a industrial dataset, the field is different
        :return: Prioritized test set
        """

        total_builds = len(agent.history['BuildId'].unique())

        if total_builds <= 2:
            # There is no enough history
            return np.arange(len(agent.actions))

        history = agent.history

        if self.history_size > 0:
            history = agent.history[agent.history.BuildId >= total_builds - self.history_size]

        bid = history.BuildId.values
        history_tc_names = history.Name.values
        features = agent.features
        num_features = len(features)

        colnames = list(np.array(sorted(list(set(history_tc_names)))))
        dataset_base = pd.DataFrame(columns=colnames)
        for build_id in list(set(bid)):
            df_build = agent.history[agent.history.BuildId == build_id][['Name', 'Verdict'] + features]
            dataset_t = df_build.set_index('Name').T
            # dataset_t['BuildId'] = build_id
            # dataset_t = dataset_t.reset_index()
            # dataset_t.columns = ['index'] + colnames + ['BuildId']
            dataset_base = dataset_base.append(dataset_t)

        dataset_base = dataset_base.reset_index(drop=True)
        dataset_base = dataset_base.fillna(0)
        # dataset_base.rename({'index': 'Features'}, axis=1, inplace=True)
        # convert into input/output

        X, y = self._split_contextual_(dataset_base, num_features)
        model = self._get_rf_model()

        model.fit(X, y)
        n_test_cases = X.shape[-1]
        # Create input for the prediction with the last commit from the history
        pred_input = np.zeros(X[-1].shape)
        # Select only the verdict for the input
        pred_input[:-1] = X[-(num_features + 1)][1:]

        # demonstrate prediction
        x_input = pred_input.reshape((1, n_test_cases))
        pred = model.predict(x_input)[0]

        #Current Test Case set Available, which we want to predict
        current_tc_set = agent.actions.Name.values.astype(np.int)
        mapping = np.zeros(len(agent.actions))

        for i, n in enumerate(current_tc_set):
            if n in colnames:
                n = colnames.index(n),
                mapping[i] = pred[n] if isinstance(pred, np.ndarray) else pred

        # Sort ignoring predicted duration
        ind_pred = np.array([0]) if len(mapping) == 1 else np.max(mapping) - mapping

        agent.actions['CalcPrio'] = ind_pred
        agent.actions.sort_values(by=['CalcPrio'], inplace=True)

        return agent.actions['Name'].tolist()
