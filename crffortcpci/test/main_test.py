import unittest
from unittest import TestCase

import gc
import os
import time
from pathlib import Path

from crffortcpci.agent import Agent
from crffortcpci.environment import Environment
from crffortcpci.evaluation import NAPFDMetric
from crffortcpci.evaluation import NAPFDVerdictMetric
from crffortcpci.scenarios import IndustrialDatasetScenarioProvider
from crffortcpci.strategy.rnn import RNN


class TestRunLSTM(TestCase):
    def setUp(self):
        self.datasets = ["alibaba@druid"]
        self.dataset_dir = "../../data"
        self.output_dir = "test_results/lstm"
        # self.sched_time_ratio = [0.1, 0.5, 0.8]
        self.sched_time_ratio = [0.5]
        self.ITERATIONS = 1
        self.INDUSTRIAL_DATASETS = ['iofrol', 'paintcontrol', 'gsdtsr', 'lexisnexis']

    def _get_agents(self):
        """
        This function prepares the possible LSTM architectures to be evaluated
        :return:
        """
        agents = []

        # LSTM standard
        agents.append(Agent(RNN()))

        return agents

    def test_env(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        for tr in self.sched_time_ratio:
            experiment_dir = os.path.join(self.output_dir, f"time_ratio_{int(tr * 100)}/")
            Path(experiment_dir).mkdir(parents=True, exist_ok=True)

            for dataset in self.datasets:
                metric = NAPFDVerdictMetric() if dataset in self.INDUSTRIAL_DATASETS else NAPFDMetric()

                # Get scenario
                scenario_provider = IndustrialDatasetScenarioProvider(
                    f"{self.dataset_dir}/{dataset}/features-engineered.csv",
                    tr)

                # Stop conditional
                trials = 50 #scenario_provider.max_builds

                env = Environment(self._get_agents(), scenario_provider, metric)

                # create a file with a unique header for the scenario (workaround)
                env.monitor.create_file(f"{experiment_dir}{str(env.scenario_provider)}.csv")

                # Compute time
                start = time.time()

                env.run_single(1, trials)
                env.store_experiment(f"{experiment_dir}{str(env.scenario_provider)}.csv")

                end = time.time()

                print(f"Time expend to run the experiments: {end - start}")

                gc.collect()
