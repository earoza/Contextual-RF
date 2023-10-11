import random
from typing import List

import pandas as pd

from crffortcpci.evaluation import EvaluationMetric

from crffortcpci.utils.features_utils import FEATURES_GROUPS


class Agent(object):
    """
    An Agent is able to take one of a set of actions at each time step.
    The action is chosen using a strategy based on the history of prior actions and outcome observations.
    """

    def __init__(self, strategy=None, history=None, feature_group=None):
        """
        :param strategy: Strategy used to prioritize the test set available in each commit (CI Cycle)
        :param history: historical test data (updated in each step)
        """
        self.strategy = strategy

        # Name | Unique numeric identifier of the test case
        # Duration | Approximated runtime of the test case
        # CalcPrio | Priority of the test case, calculated by the prioritization algorithm(output column, initially 0)
        # LastRun | Previous last execution of the test case as date - time - string(Format: `YYYY - MM - DD HH: ii`)
        # NumRan | Test runs
        # NumErrors | Test errors revealed
        
        self.tc_fieldnames = ['BuildId', 'Name', 'Duration', 'CalcPrio', 'LastRun', 'NumRan', 'NumErrors', 'Verdict']
        #self.tc_fieldnames = ['BuildId', 'Name', 'Duration', 'CalcPrio', 'LastRun','LastResults', 'Verdict']
        self.features = FEATURES_GROUPS[feature_group]
        self.fields = list(set(self.tc_fieldnames + self.features))

        # List of features
        self.context_features = None

        self.reset()

    def __str__(self):
        return f'{str(self.strategy)}'

    def reset(self):
        """
        Resets the agent's memory to an initial state.
        """
        self.actions = pd.DataFrame(columns=self.tc_fieldnames)
        self.history = pd.DataFrame(columns=self.fields)

        # Last action (TC) choosen
        self.last_prioritization = None

        # Time of usage
        self.t = 0

    def update_actions(self, actions):
        self.actions = pd.DataFrame(actions, columns=self.fields)

    def update_history(self, history):
        """
        We update the Historical Test Data at the current commit
        :param action: Historical test data
        """
        self.history = pd.DataFrame(history, columns=self.fields)

    def update_priority(self, action):
        """
        We update the Priority column with the priorities
        :param action: List of test cases in order of prioritization
        """
        self.actions['CalcPrio'] = self.actions['Name'].apply(lambda x: action.index(x) + 1)

    def choose(self) -> List[str]:
        """
        The agent choose an action.
        An action is the Priorized Test Suite
        :return: List of Test Cases in ascendent order of priority
        """
        self.last_prioritization = []
        self.t += 1

        # If is the first time that the agent is been used, we don't have a "history" (rewards).
        # So, I we can choose randomly
        if self.t == 1:
            actions = self.actions['Name'].tolist()
            random.shuffle(actions)
            self.last_prioritization = actions

            self.update_priority(self.last_prioritization)

            # After, we must to order the test cases based on the priorities
            # Sort tc by Prio ASC (for backwards scheduling)
            self.actions = self.actions.sort_values(by=['CalcPrio'])

        else:
            self.last_prioritization = self.strategy.choose_all(self)

        # Return the Priorized Test Set
        return self.last_prioritization

    def pull(self, evaluation_metric):
        """
        Submit prioritized test set for evaluation step the environment and get new measurementss
        :return: The result ("state") of an evaluation by Evaluation Metric
        """
        evaluation_metric.evaluate(self.actions.to_dict(orient='record'))

    def update_features(self, features):
        self.features = features
