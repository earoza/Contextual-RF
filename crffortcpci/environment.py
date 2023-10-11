import gc
import os
import pickle
import time
import warnings
from pathlib import Path

import pandas as pd

from crffortcpci.scenarios import IndustrialDatasetScenarioProvider
from crffortcpci.utils.monitor import MonitorCollector

pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings('ignore')
Path("backup").mkdir(parents=True, exist_ok=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



class Environment(object):
    """
    The environment class run the simulation.
    """

    def __init__(self, agents, scenario_provider: IndustrialDatasetScenarioProvider, evaluation_metric):
        self.agents = agents
        self.scenario_provider = scenario_provider
        self.evaluation_metric = evaluation_metric
        self.fits = []
        self.reset()

    def reset(self):
        """
        Reset all variables (for a new simulation)
        :return:
        """
        self.reset_agents_memory()
        # Monitor saves the feedback during the process
        self.monitor = MonitorCollector()

    def store_experiment(self, name):
        self.monitor.save(name)

    def reset_agents_memory(self):
        """
        # Resets the agent's memory to an initial state.
        :return:
        """
        for agent in self.agents:
            agent.reset()

    def run_single(self, experiment, trials=100, restore=True):
        """
        Execute a simulation
        :param experiment: Current Experiment
        :param trials: The max number of scenarios that will be analyzed
        :param restore: restore the experiment if fail (i.e., energy down)
        :return:
        """
       

        # The agent must to learn from the beginning for each experiment
        self.reset_agents_memory()

        is_industrial_dataset = self.scenario_provider.get_is_industrial_dataset()

        # restore to step
        r_t = 1

        if restore:
            # restore the experiment if fail (i.e., energy down)
            r_t, self.agents, self.monitor = self.load_experiment(experiment)
            self.scenario_provider.last_build(r_t)
            r_t += 1  # start 1 step after the last build

        # For each "virtual scenario (vsc)" I must analyse it and evaluate it
        for (t, vsc) in enumerate(self.scenario_provider, start=r_t):
            # The max number of scenarios that will be analyzed
            if t > trials:
                break

            test_set_available = vsc.get_testcases()

            # Update the time budget available
            self.evaluation_metric.update_available_time(vsc.get_available_time())

            # If there is only a test case, then it is impossible that it execute with
            # reduced maximum time. Then, the fitness is 0. So, we consider the total time
            # TODO: Implementing?
            #if len(test_set_available) == 1:
            #    self.evaluation_metric.update_available_time(vsc.get_total_build_duration())

            # I can analyse the same "moment/scenario t" for "i agents"
            for i, agent in enumerate(self.agents):
                # Update the agent with the current information available
                agent_time_start = time.time()
                agent.update_actions(test_set_available)  # Test set available (actions)
                agent.update_history(vsc.get_history())  # Historical test data + Context Features
                agent_time_end = time.time()
                agent_time_duration = agent_time_end - agent_time_start

                feature_group = str(vsc.get_feature_group())
                exp_name = f"{str(agent)}_{feature_group}"

                # Compute time
                start = time.time()

                # Choose action (Prioritized Test Suite List) from agent
                action = agent.choose()
                strategy_type = 'boosting' if agent.strategy.strategyName.count('XGBoost') else 'bagging'
                # Pick up reward from bandit for chosen action
                # Submit prioritized test cases for evaluation step the environment and get new measurements
                agent.pull(self.evaluation_metric)
            
                end = time.time()
                duration = end - start
                print(f"Exp {experiment} - Ep {t} - Agent {str(agent)} - Feature Group {feature_group} -NAPFD/APFDc: ", end="")
                print(f"{self.evaluation_metric.fitness:.4f}/{self.evaluation_metric.cost:.4f} - Time: {duration}")
                self.monitor.collect(self.scenario_provider,
                                     vsc.get_available_time(),
                                     experiment,
                                     t,
                                     exp_name,
                                     strategy_type,
                                     self.evaluation_metric,
                                     self.scenario_provider.total_build_duration,
                                     agent_time_duration + duration,
                                     0,
                                     action)

                # Save experiment each 50000 builds
                if restore and t % 50000 == 0:
                    self.save_experiment(experiment, t)
            #gc.collect()

    def store_experiment(self, name):
        self.monitor.save(name)

    def load_experiment(self, experiment):
        filename = f'backup/{str(self.scenario_provider)}_ex_{experiment}.p'


        if os.path.exists(filename):
            return pickle.load(open(filename, "rb"))

        return 0, self.agents, self.monitor

    def save_experiment(self, experiment, t):
        filename = f'backup/{str(self.scenario_provider)}_ex_{experiment}.p'
        pickle.dump([t, self.agents, self.monitor], open(filename, "wb"))
