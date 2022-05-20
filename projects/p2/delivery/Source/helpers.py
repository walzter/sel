import numpy as np
import pandas as pd


def entropy(probability: int) -> float:
    """
    Calculates the entropy of a given probability.
    According to the formula:
    H(x) = -p(x)log(p(x)) - (1-p(x))log(1-p(x))
    """
    ## first check if the probability is 0 or 1
    if probability in [0, 1]:
        return 0
    else:
        return -(
            probability * np.log2(probability)
            + (1 - probability) * np.log2(1 - probability)
        )


def load_dataset(dataset_name):
    """
    Loads the dataset from the dictionary.
    """
    DATASETS = {
        "small": "../Data/small/glass.data",
        "medium": "../Data/medium/drug_consumption.data",
        "large": "../Data/large/CTG.csv",
    }
    if dataset_name == "small":
        ## exclude the first column ID (labeled 0 to 10, so use 1-10)
        df = pd.read_csv(DATASETS[dataset_name], header=None)
        df.drop(df.columns[0], axis=1, inplace=True)
        df.columns = list(range(0, 10))
        return df
    if dataset_name == "medium":
        df = pd.read_csv(DATASETS[dataset_name], header=None)
        tgt = 21
        cols_to_keep = list(range(0, 13)) + [tgt]
        # df.select_dtypes(include='object')
        df = df[cols_to_keep]
        ## convert the categorical to numbers
        txt2num = {x: i for i, x in enumerate(df[tgt].unique())}
        df[tgt] = df[tgt].map(txt2num)
        df.drop(0, axis=1, inplace=True)
        return df
    if dataset_name == "large":
        df = pd.read_csv(DATASETS[dataset_name])
        ## fill nans with median value of the column
        df.fillna(df.median(), inplace=True)
        return df
    else:
        print("Dataset not found")


## LOADING
def prepare_dataset(dataset_name: str):
    ## load the data
    df = load_dataset(dataset_name)
    features = df.columns[:-1]
    target = df.columns[-1]
    ## split the data into training and test sets
    nb_train = int(0.8 * len(df))
    df = df.sample(frac=1, random_state=217)
    X_train = df[features][:nb_train]
    y_train = df[target][:nb_train].values
    X_test = df[features][nb_train:]
    y_test = df[target][nb_train:].values
    return X_train, y_train, X_test, y_test


def information_gain(sister: np.array, brother: np.array) -> float:
    """
    Given two children (left and right), calculate the Information Gain according to the formula:

    G(x) = H(x) - H(x|y)
    Which is the entropy of the parent node - the entropy of the children nodes.

    """
    ## calcualte the information gain for the parents (mom or dad)
    mom = sister + brother
    p_x_mom = mom.count(1) / len(mom) if len(mom) > 0 else 0
    information_gain_mom = entropy(p_x_mom)

    ## now calculate the information gain for the children (sister / brother)
    p_x_sister = sister.count(1) / len(sister) if len(sister) > 0 else 0
    information_gain_sister = entropy(p_x_sister)
    ## now calculate the other sibling
    p_x_brother = brother.count(1) / len(brother) if len(brother) > 0 else 0
    information_gain_brother = entropy(p_x_brother)
    ## total information gain
    family_information_gain = (
        information_gain_mom
        - len(sister) / len(mom) * information_gain_sister
        - len(brother) / len(mom) * information_gain_brother
    )
    return family_information_gain
