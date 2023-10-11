import argparse
import gc
import os
import time
from multiprocessing import Pool
from pathlib import Path

from crffortcpci.agent import Agent
from crffortcpci.environment import Environment
from crffortcpci.evaluation import NAPFDMetric
from crffortcpci.evaluation import NAPFDVerdictMetric
from crffortcpci.scenarios import IndustrialDatasetContextScenarioProvider
from crffortcpci.strategy.contextual_random_forest import RandomForestHistorical
from crffortcpci.utils.features_utils import GROUPS


INDUSTRIAL_DATASETS = ['iofrol', 'paintcontrol', 'gsdtsr', 'lexisnexis']
DEFAULT_EXPERIMENT_DIR = 'results/'
DEFAULT_SCHED_TIME_RATIO = [0.1, 0.5, 0.8]
# DEFAULT_SCHED_TIME_RATIO = [0.1]

ITERATIONS = 30
PARALLEL_POOL_SIZE = 10
# PARALLEL_POOL_SIZE = 1


def run_experiments_with_threads(dataset_dir, dataset, metric, agents, time_budget, feature_group):
    # Get scenario
    # TODO Ajust this code segment with a try sequence -> TRY Contextual, EXCEPT -> features-engineered, ELSE -> FILE NOT FOUND
    scenario_provider = IndustrialDatasetContextScenarioProvider(
        f"{dataset_dir}/{dataset}/features-engineered-contextual.csv",
        feature_group,
        time_budget)
    # scenario_provider = IndustrialDatasetScenarioProvider(f"{dataset_dir}/{dataset}/features-engineered.csv",
    #                                                       time_budget)

    # Stop conditional
    trials = scenario_provider.max_builds

    env = Environment(agents, scenario_provider, metric)

    parameters = [(i + 1, trials, env) for i in range(ITERATIONS)]

    # create a file with a unique header for the scenario (workaround)
    env.monitor.create_file(f"{EXPERIMENT_DIR}{str(env.scenario_provider)}.csv")

    # Compute time
    start = time.time()

    with Pool(PARALLEL_POOL_SIZE) as p:
        p.starmap(exp_run, parameters)

    end = time.time()

    print(f"Time expend to run the experiments: {end - start}")


def exp_run(iteration, trials, env: Environment):
    env.run_single(iteration, trials)
    env.store_experiment(f"{EXPERIMENT_DIR}{str(env.scenario_provider)}.csv")


def get_agents(fg):
    """
    This function prepares the possible ML architectures to be evaluated
    We can use multiple instances of different or the same ML algorithm to evaluate
    :return:
    """
    agents = []

    # RF standard

    # agents.append(Agent(strategy=RandomForestHistorical(history_size=4), feature_group=fg))
    # agents.append(Agent(strategy=RandomForestHistorical(history_size=10), feature_group=fg))
    # agents.append(Agent(strategy=RandomForestHistorical(history_size=25), feature_group=fg))
    # agents.append(Agent(strategy=RandomForestHistorical(history_size=30), feature_group=fg))
    agents.append(Agent(strategy=RandomForestHistorical(history_size=40), feature_group=fg)) # Best overall
    # agents.append(Agent(strategy=RandomForestHistorical(history_size=50), feature_group=fg))
    # agents.append(Agent(strategy=RandomForestHistorical(history_size=4, n_estimators=100), feature_group=fg))
    # agents.append(Agent(strategy=RandomForestHistorical(history_size=75), feature_group=fg))
    # agents.append(Agent(strategy=RandomForestHistorical(history_size=100), feature_group=fg))
    # agents.append(Agent(strategy=RandomForestHistorical(history_size=50, n_estimators=750), feature_group=fg))
    return agents


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Contextual RF for TCPCI')

    ap.add_argument('--parallel_pool_size', type=int,
                    default=PARALLEL_POOL_SIZE)
    ap.add_argument('--dataset_dir', default='context_datasets')
    ap.add_argument('--datasets', nargs='+', default=[
        'alibaba@druid',
        'alibaba@fastjson',
        'DSpace@DSpace',
        'square@okhttp',
        'square@retrofit',
        'zxing@zxing'
    ],
                    help='Datasets to analyse. Ex: \'deeplearning4j@deeplearning4j\'')

    ap.add_argument('--sched_time_ratio', nargs='+',
                    default=DEFAULT_SCHED_TIME_RATIO, help='Schedule Time Ratio')

    ap.add_argument('--feature_groups', nargs='+',
                    default=GROUPS, help='Feature Groups')

    ap.add_argument('-o', '--output_dir', default=DEFAULT_EXPERIMENT_DIR)

    args = ap.parse_args()

    PARALLEL_POOL_SIZE = args.parallel_pool_size

    for tr in args.sched_time_ratio:
        EXPERIMENT_DIR = os.path.join(args.output_dir, f"time_ratio_{int(tr * 100)}/")
        Path(EXPERIMENT_DIR).mkdir(parents=True, exist_ok=True)

        for dataset in args.datasets:
            metric = NAPFDVerdictMetric() if dataset in INDUSTRIAL_DATASETS else NAPFDMetric()
            for fg in GROUPS:
                run_experiments_with_threads(dataset_dir=args.dataset_dir,
                                             dataset=dataset,
                                             metric=metric,
                                             feature_group=fg,
                                             agents=get_agents(fg),
                                             time_budget=tr)
            gc.collect()
