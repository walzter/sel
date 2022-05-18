import os
import glob


def make_config(data_directory: str = "raw_data", out_directory: str = "out") -> dict:
    """
    Makes the main config dictionary

    """
    ## MAIN DICTIONARY 
    config = {
        ## DIRECTORIES 
        "DATA_DIR": f"{data_directory}",
        "OUT_DIR": f"{out_directory}",
        "VERBOSITY": True,
        ## TRAINING & TESTING
        "TRAIN_SPLIT":0.1,
        "CLASS_INDEX":0,
        
        ## get the directories and the files
        "small": glob.glob(os.path.join(os.path.join(data_directory, "small"), "*")),
        "medium": glob.glob(os.path.join(os.path.join(data_directory, "medium"), "*")),
        "large": glob.glob(os.path.join(os.path.join(data_directory, "large"), "*")),
    }
    return config
