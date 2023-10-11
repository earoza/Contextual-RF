import numpy as np
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor

from crffortcpci.agent import Agent


class RandomForestHistorical(object):
    def __init__(self, strategyName="RF_Vertical_SW_history_based", history_size=100, n_estimators=500,
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

    def choose_all(self, agent: Agent):
        """
        The function prioritizes a test set
        :param agent:
        :return: Prioritized test set
        """

        total_builds = len(agent.history['BuildId'].unique())

        if total_builds <= 2:
            # There is no enough history
            return np.arange(len(agent.actions))

        history = agent.history

        if self.history_size > 0:
            history = agent.history[agent.history.BuildId >= total_builds - self.history_size]

        unused_cols = ['Verdict', 'BuildId', 'LastRun', 'Name', 'CalcPrio']
        # Drop unused cols for training
        historical_set = history.drop(unused_cols, axis=1)
        X = np.array(historical_set.values)
        # Select Verdict collumn for training
        y = np.array(history['Verdict'].values)
        model = self._get_rf_model()
        model.fit(X, y)

        # Extract the number of TC from the last historical Build
        last_historical_tc = history[history.BuildId == total_builds].shape[0]
        x_predic = X[-last_historical_tc:]

        # Feed the extracted TC to the predictor
        pred = model.predict(x_predic)

        last_history_names = list(history[history.BuildId == total_builds].Name.values)

        # Current Test Case set Available, which we want to predict
        current_tc_set = agent.actions.Name.values.astype(np.int)
        mapping = np.zeros(len(agent.actions))

        for i, n in enumerate(current_tc_set):
            if n in last_history_names:
                n = last_history_names.index(n),
                mapping[i] = pred[n] if isinstance(pred, np.ndarray) else pred

        # Sort ignoring predicted duration
        ind_pred = np.array([0]) if len(mapping) == 1 else np.max(mapping) - mapping

        agent.actions['CalcPrio'] = ind_pred

        agent.actions['CalcPrio'] = ind_pred
        agent.actions.sort_values(by=['CalcPrio'], inplace=True)

        return agent.actions['Name'].tolist()
