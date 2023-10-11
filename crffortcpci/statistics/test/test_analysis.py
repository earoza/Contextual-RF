import unittest
from unittest import TestCase

from constantine.statistics.analysis import Analisys


class TestAnalisys(TestCase):
    def setUp(self):
        self.project_dir = "../../../data/deeplearning4j@deeplearning4j"
        self.results_dir = '../../../results/experiments/'
        self.names = {
            'ε-greedy (ε=0.3)': 'ε-Greedy',
            'ε-greedy (ε=0.5)': 'ε-Greedy',
            'UCB (C=0.5)': 'UCB',
            'UCB (C=0.3)': 'UCB',
            'FRRMAB (C=0.3, D=1, SW=100)': 'FRRMAB',
            'LinUCB (Alpha=0.5)': 'LinUCB'
        }

    # @unittest.skip
    def test_total(self):
        ana = Analisys(self.project_dir, self.results_dir)

        ana.replace_names(self.names)

        # ana.plot_accumulative("ACC_NAPFD")
        # ana.plot_accumulative("ACC_APFDc", 'cost')  # APFDc
        # ana.plot_accumulative("ACC_Reward", 'rewards')  # Rewards
        #
        # ana.plot_lines("NAPFD_Variation")
        # ana.plot_lines("APFDc_Variation", 'cost')  # APFDc

        ana.visualize_boxplot_unique()  # NAPFD
        ana.visualize_boxplot_unique('cost')  # APFDc
        ana.visualize_boxplot_unique('ttf')  # RFTC
        ana.visualize_boxplot_unique('prioritization_time')  # Prioritization Time

        # print(
        #     f"\n\n\n\n||||||||||||||||||||||||||||||| NORMALIZED TIME REDUCTION (NTR) |||||||||||||||||||||||||||||||\n")
        # ana.visualize_ntr()
        #
        # print(f"\n\n||||||||||||||||||||||||||||||| PRIORITIZATION DURATION |||||||||||||||||||||||||||||||\n")
        # # Time Spent to Prioritize (or find a suitable solution)
        # ana.visualize_duration()  # TODO: ver se não é redundante se eu usar o teste estatístico

        # Apply the Kruskal-Wallis Test in the Data
        # ana.statistical_test_kruskal()  # NAPFD
        # ana.statistical_test_kruskal('cost')  # APFDc
        # ana.statistical_test_kruskal('ttf')  # RFTC
        # ana.statistical_test_kruskal('prioritization_time')  # Prioritization Time

    @unittest.skip
    def test_change_project(self):
        ana = Analisys(self.project_dir, self.results_dir)
        ana.update_project("../../../data/libssh@libssh-mirror/libssh@CentOS7-openssl")
        ana.replace_names(self.names)

        # ana.plot_accumulative("ACC_NAPFD")
        # ana.plot_accumulative("ACC_APFDc", 'cost')  # APFDc
        # ana.plot_accumulative("ACC_Reward", 'rewards')  # Rewards

        # ana.plot_lines("NAPFD_Variation")
        # ana.plot_lines("APFDc_Variation", 'cost')  # APFDc

        print("\n\n\n\n|||| NORMALIZED TIME REDUCTION (NTR) ||||\n")
        ana.visualize_ntr()
