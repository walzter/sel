## imports
import numpy as np
import pandas as pd
import random
import time
from termcolor import colored
import itertools

## helper functions for the tree
from helpers import *

## argparse
import argparse


def make_boot_sample(X_train: np.array, y_train: np.array, run_type: str) -> np.array:

    if run_type == "rf":
        boot_idx = list(
            np.random.choice(range(len(X_train)), len(X_train), replace=True)
        )
        xboot, yboot = (
            X_train.iloc[boot_idx].values,
            y_train[boot_idx],
        )
    elif run_type == "df":
        xboot, yboot = (
            X_train.values,
            y_train,
        )
    return xboot, yboot


def split_multiple_nodes(xboot: np.array, yboot: np.array, top_features: int) -> dict:
    """ """
    all_features = []
    num_features = len(xboot[0])
    while len(all_features) <= top_features:
        fidx = random.sample(range(num_features), 1)
        if fidx not in all_features:
            all_features.extend(fidx)
    top_score = -9.99 * 1000
    curr_node = None
    for feature_idx in all_features:
        for splitter in xboot[:, feature_idx]:
            lc = {"xboot": [], "yboot": []}
            rc = {"xboot": [], "yboot": []}

            for i, value in enumerate(xboot[:, feature_idx]):
                if value <= splitter:
                    lc["xboot"].append(xboot[i])
                    lc["yboot"].append(yboot[i])
                else:
                    rc["xboot"].append(xboot[i])
                    rc["yboot"].append(yboot[i])

            curr_info_gain = information_gain(lc["yboot"], rc["yboot"])
            if curr_info_gain > top_score:
                top_score = curr_info_gain
                lc["xboot"] = np.array(lc["xboot"])
                rc["xboot"] = np.array(rc["xboot"])
                curr_node = {
                    "information_gain": curr_info_gain,
                    "left_child": lc,
                    "right_child": rc,
                    "split_point": splitter,
                    "feature_idx": feature_idx,
                }

    return curr_node


def leaf(node: dict) -> int:
    """
    Returns a prediction of a class for a node.
    """
    return max(node["yboot"], key=node["yboot"].count)


def split_single_node(
    node: dict, top_features: int, min_obs: int, tiefe: int, depth: int
) -> None:

    """
    Checks if the node is splitted or not.

    PARAMS:
    -------
    node : the node to check if it is splitted or not

    top_features : the number of features to consider when doing the splits

    min_obs : the minimum number of samples to split a node

    tiefe : the maximum depth to split a node

    depth : the current depth of the node

    RETURNS:
    --------
    None : splits node of the tree

    """
    lc, rc = node["left_child"], node["right_child"]

    del node["left_child"]
    del node["right_child"]

    if len(lc["yboot"]) == 0 or len(rc["yboot"]) == 0:
        empty_child = {"yboot": lc["yboot"] + rc["yboot"]}
        node["left_split"] = leaf(empty_child)
        node["right_split"] = leaf(empty_child)
        return

    if depth >= tiefe:
        node["left_split"] = leaf(lc)
        node["right_split"] = leaf(rc)
        return node

    if len(lc["xboot"]) <= min_obs:
        node["left_split"] = node["right_split"] = leaf(lc)
    else:
        node["left_split"] = split_multiple_nodes(
            lc["xboot"], lc["yboot"], top_features
        )
        split_single_node(node["left_split"], tiefe, min_obs, tiefe, depth + 1)
    if len(rc["xboot"]) <= min_obs:
        node["right_split"] = node["left_split"] = leaf(rc)
    else:
        node["right_split"] = split_multiple_nodes(
            rc["xboot"], rc["yboot"], top_features
        )
        split_single_node(node["right_split"], top_features, min_obs, tiefe, depth + 1)


def grow_a_tree(
    xboot: np.array,
    yboot: np.array,
    tiefe: int,
    min_obs: int,
    top_features: int,
):
    base_ = split_multiple_nodes(xboot, yboot, top_features)
    split_single_node(base_, top_features, min_obs, tiefe, 1)
    return base_


def make_model(
    X_train: np.array,
    y_train: np.array,
    n_estimators: int,
    top_features: int,
    run_type: str,
    tiefe: int = 10,
    min_obs: int = 2,
):
    all_trees = [None] * n_estimators
    for i in range(n_estimators):
        xboot, yboot = make_boot_sample(X_train, y_train, run_type)
        tree = grow_a_tree(xboot, yboot, top_features, tiefe, min_obs)
        all_trees[i] = tree
    return all_trees


def runif(M):
    return np.random.randint(low=1, high=M + 1)


def make_prediction(tree, X_test):
    fidx = tree["feature_idx"]

    if X_test[fidx] <= tree["split_point"]:
        if type(tree["left_split"]) == dict:
            return make_prediction(tree["left_split"], X_test)
        else:
            value = tree["left_split"]
            return value
    else:
        if type(tree["right_split"]) == dict:
            return make_prediction(tree["right_split"], X_test)
        else:
            return tree["right_split"]


def predictor(all_trees: list, X_test: np.array) -> np.array:
    rf_predictions = []
    for observation in range(len(X_test)):
        all_preds = [
            make_prediction(tree, X_test.values[observation]) for tree in all_trees
        ]
        final_pred = max(all_preds, key=all_preds.count)
        rf_predictions.append(final_pred)
    return np.array(rf_predictions)


def make_seed(seed: int) -> None:
    """
    Sets the seed for the random number generator.
    """
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    ## Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", type=str, help="dataset")
    parser.add_argument(
        "-rt", type=str, help="run type: rf (RandomForest) or df (DecisionTree)"
    )
    args = parser.parse_args()
    ## Load the dataset
    DATA_SET_SIZE = args.ds
    ## split the dataset
    X_train, y_train, X_test, y_test = prepare_dataset(DATA_SET_SIZE)
    ## NAME FOR THE FILE
    NAME = "Random_Forest" if args.rt == "rf" else "Decision_Forest"
    if args.rt == "df":
        make_seed(42)
    ## NUMBER OF TREES
    NUM_TREES, M = [1, 10, 25, 50, 75, 100], len(X_train)
    ## NUMBER OF FEATURES
    F_RF, F_DF = [1, 3, int(np.log2(M + 1)), int(np.sqrt(M))], [
        int(M / 4),
        int(M / 2),
        int(M * 3 / 4),
        runif(M),
    ]
    ## make combinations of all the parameters
    if args.rt == "rf":
        combs = list(itertools.product(NUM_TREES, F_RF))
    elif args.rt == "df":
        combs = list(itertools.product(NUM_TREES, F_DF))
    ## run the model for each combination
    holders = []
    for nt, f in combs:
        tic = time.time()
        model = make_model(
            X_train,
            y_train,
            n_estimators=nt,
            top_features=f,
            tiefe=10,
            min_obs=2,
            run_type=args.rt,
        )
        toc = time.time()
        tic_toc = toc - tic
        preds = predictor(model, X_test)
        acc = sum(preds == y_test) / len(y_test)
        ## print the results: Trees | Features | Accuracy | Time , in green
        print(
            colored(
                f"Model {NAME} | Trees {nt} | Number of Features {f} | Accuracy {acc*100:.2f} | Time {tic_toc:.2f}s",
                "green",
            )
        )
        ## save the results to a csv file
        model_df = pd.DataFrame.from_records(model)
        model_df.drop(["left_split", "right_split"], axis=1, inplace=True)
        ## add the columns for nt & f
        model_df["Num_trees"] = nt
        model_df["Num_features"] = f
        model_df["Accuracy"] = acc
        model_df["Time"] = tic_toc
        holders.append(model_df)
        ## combine the list of dataframes
    combined_df = pd.concat(holders)
    combined_df.to_csv(f"./Data/out/{DATA_SET_SIZE}_{NAME}.csv", index=False)
