import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

COLUMNS = ['original', 'translation', 'z_mean', 'comet', 'bertScore']
FEATURES = ['lan', 'unigram', 'bigram', '3gram', '4gram', '5gram']
DA_COLUMNS = ['original', 'translation', 'z_mean', 'hter']
REF = True


def prepare_data(datasets, da=False):
    # take the required column
    columns_to_take = DA_COLUMNS if da else COLUMNS
    datasets = [dataset[columns_to_take + FEATURES] for dataset in datasets]
    # rename columns
    datasets = [dataset.rename(columns={'original': 'text_a', 'translation': 'text_b', 'z_mean': 'labels'}).dropna()
                      for dataset in datasets]
    # remove text_b
    datasets = [dataset.drop(['text_b'], axis='columns') for dataset in datasets]
    return datasets
