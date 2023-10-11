"""
Based on Prado Lima COLEMAN extension CONSTANTINE which can use any feature given a dataset, and in our experiments we considers the follow ones:
    - Test Case Duration (Duration): The time spent by a test case to execute;
    - Number of Test Ran Methods (NumRan): The number of test methods executed during the test, considering that some test
methods are not executed due to some previous test method(s) have been failed;
    - Number of Test Failed Methods (NumErrors): The number of test methods which failed during the test.
As we do not have any information about correlation failure, we assume large number of failed test methods have higher
probability to detect different failures;
    - Test Case Age (TcAge): This feature measures how long the test case exists and is given by a number which
is incremented for each new CI Cycle in that the test case is used;
    - Test User-Preference (UserPref_X): Test case importance given by the tester. In this case, this feature is based on
user-preference;
    - Test Case Change (ChangeType): Considers whether a test case changed. If a test case is changed from a commit
to another, there is a high probability that the alteration was performed because some change in the software needs
to be tested. If the test case was changed, we could detect and consider if the test case was renamed, or it added
or removed some methods;
    - Cyclomatic Complexity (McCabe): This feature considers the complexity of McCabe.
High complexity can be related to a more elaborated test case;
    - Test Size (SLOC): Typically, size of a test case refers to either the lines of code or the number of
``assertions'' in a test case. This feature is note correlated with coverage. For instance,
if we have two tests t_1 and t_2 and both cover a method, but t_2 have more assertions than t_1, consequently,
t_2 have higher chances to detect failures;
 """

GROUPS = ['All_Features',
          'Time_Execution',
          'Program_Size',
          'TC_Complexity',
          'TC_Evolution',
          'Feature_Selection'
        ]

FEATURES_GROUPS = {
    'All_Features': ['Duration', 'NumErrors', 'NumRan', 'SLOC', 'McCabe', 'TcAge', 'ChangeType'],
    'Time_Execution': ['Duration', 'NumErrors'],
    'Program_Size': ['SLOC'],
    'TC_Complexity': ['NumRan', 'McCabe'],
    'TC_Evolution': ['TcAge', 'ChangeType'],
    'Feature_Selection': ['TcAge', 'SLOC', 'NumErrors', 'ChangeType']
    # 'Oracle': ['Duration', 'NumErrors', 'SLOC', 'TcAge', 'ChangeType'],
    # 'Feature_Selection_UserPref_10': ['TcAge', 'SLOC', 'NumErrors', 'ChangeType', 'UserPref_10'],
    # 'Feature_Selection_UserPref_50': ['TcAge', 'SLOC', 'NumErrors', 'ChangeType', 'UserPref_50'],
    # 'Feature_Selection_UserPref_80': ['TcAge', 'SLOC', 'NumErrors', 'ChangeType', 'UserPref_80'],
    # 'Time_Execution_UserPref_10': ['Duration', 'NumErrors', 'UserPref_10'],
    # 'Time_Execution_UserPref_50': ['Duration', 'NumErrors', 'UserPref_50'],
    # 'Time_Execution_UserPref_80': ['Duration', 'NumErrors', 'UserPref_80']
}

DEFAULT_PREVIOUS_BUILD = ['Duration', 'NumRan', 'NumErrors']
