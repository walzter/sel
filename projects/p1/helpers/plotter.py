import pandas as pd
import matplotlib.pyplot as plt
import gc

## visualize some distribution
def plot_class_distribution(
    dataframe: pd.DataFrame, target_col: str, size: str, verbose: bool
) -> None:
    ## visualize the distribution of the target column
    gc.collect()
    values = dataframe[target_col].value_counts()
    ## plot the distribution
    values.plot(
        kind="bar",
        xlabel=target_col,
        ylabel="Count",
        title=f"Distribution of target col: {target_col} for dataset: {size}",
        rot=0,
    )
    plt.show()
    if verbose:
        ## print some statistics
        for i, v in enumerate(values):
            print(
                f"Class {i} (= {values.index[i]}): {v},  Percentage of Total DataFrame: {v/len(dataframe)*100:.2f}%"
            )


## visualize the count of unique values in the different classes
def plot_unique_values_per_class(
    dataframe: pd.DataFrame, target_col: str, figure_size: tuple, verbose: bool = True
) -> None:
    """
    Plots the unique values for the different classes in a given dataframe, given the target column
    dataframe : pd.DataFrame
               The dataframe which holds the data to be visualized
    target_col : str
                The name of the target column to which visualize

    """
    gc.collect()
    ## first get the number of unique class
    uqc = dataframe[target_col].unique()
    ## now create a figure with the corresponding size
    fig, axs = plt.subplots(1, len(uqc), figsize=figure_size)
    axs = axs.ravel()
    ## iterate ver them and plot the unique values
    for idx, _class in enumerate(uqc):
        ## number of unique values per class
        uq_vals_class = dataframe[dataframe["Class"] == _class].nunique().sort_values()
        uq_vals_class.plot(
            kind="bar",
            rot=0,
            ylabel="Count",
            title=f"Number of unique values per feature of {_class}",
            ax=axs[idx],
        )
        ## print some feedback
        if verbose:
            print(f"Number of Unique Values per feature of {_class}:")
            for i, v in enumerate(uq_vals_class):
                print(f"Feature {i} (= {uq_vals_class.index[i]}): {v}")
    plt.tight_layout()
    plt.show()
