INDUSTRIAL_DATASETS = ['iofrol', 'paintcontrol', 'gsdtsr', 'lexisnexis']


def is_industrial_dataset(dataset):
    return dataset in INDUSTRIAL_DATASETS
