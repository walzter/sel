import re
import gc

## define the function for the below
def get_feature_name_small(labels: str, verbosity: bool) -> dict:
    """
    Get the feature names from the .names file and returns the list of column names.

    The feature names start after the "initial_line": "7. Attribute Information: "
    and end before the "final_line": "8. Missing Attribute Values:"

    They are defined as:
        1. feature : possible values, or ranges
    Therefore we can split by the period and empty space (". "), and get:
        feature: possible values or ranges
    Ultimately, splitting by (":"), to get
        feature
    """
    gc.collect()
    ## define the sizes of each dataset, 10, 6, 22
    holder = [None] * 10
    ## define a tracker to know which line to print:
    printing = False
    ## open the file and read
    norm_idx = -1
    with open(labels, "r") as f:
        for line in f:
            if line.startswith(
                "7. Attribute Information:"
            ):  ## the first line we want to start reading from
                printing = True
                continue  # go to next line
            elif (
                line.startswith("8. Missing Attribute Values:") and len(line) > 0
            ):  ## we want to skip this line and those that length is 0
                printing = False
                break  # quit file reading
            if printing:
                if len(line) > 1:
                    c_line = line.strip()
                    ## remove the numbers that are int he beginning and start with 1., 2., .. 10. or more
                    c_line = c_line.split(".")[1]
                    if len(c_line) > 1:
                        norm_idx += 1
                        holder[norm_idx] = c_line.strip()
                        # holder[norm_idx] = c_line.strip()
    ## we can get the column names
    ## column names
    col_names = [x.split(":")[0] for x in holder]
    ## number of features
    num_features = len(col_names)
    ## put it all in a final dictionary
    features = {"small": {"n_features": num_features, "column_names": col_names}}
    if verbosity:
        ## go through the dictionary
        for _, infos in features.items():
            print(f"Dataset Size: Small  ")
            ## for each value in the dictionary
            for inf, values in infos.items():
                ## print the key and the value
                print(f"For {inf}, the value is: {values}")
    return features


## define the function for the below
def get_feature_name_medium(labels: str, verbosity: bool) -> dict:
    """
    Get the feature names from the .names file and returns the list of column names.

    The feature names start after the "initial_line": "7. Attribute Information: "
    and end before the "final_line": "8. Missing Attribute Values:"

    They are defined as:
        1. feature : possible values, or ranges
    Therefore we can split by the period and empty space (". "), and get:
        feature: possible values or ranges
    Ultimately, splitting by (":"), to get
        feature
    """
    gc.collect()
    ## define the sizes of each dataset, 10, 6, 22
    holder = [None] * 6
    ## define a tracker to know which line to print:
    printing = False
    ## open the file and read
    norm_idx = -1
    with open(labels, "r") as f:
        for line in f:
            if line.startswith(
                "7. Attribute Values:"
            ):  ## the first line we want to start reading from
                printing = True
                continue  # go to next line
            elif (
                line.startswith("8. Missing Attribute Values:") and len(line) > 0
            ):  ## we want to skip this line and those that length is 0
                printing = False
                break  # quit file reading
            if printing:
                if len(line) > 1:
                    norm_idx += 1
                    c_line = line.strip()
                    c_line = c_line.split(" ")
                    holder[norm_idx] = c_line[0]
    ## for this specific dataset, we add the target name, since it's the lastone: class
    holder = holder + ["class"]
    ## number of features
    num_features = len(holder)
    ## put it all in a final dictionary
    features = {"medium": {"n_features": num_features, "column_names": holder}}
    if verbosity:
        ## go through the dictionary
        for _, infos in features.items():
            print(f"\nDataset Size: Medium ")
            ## for each value in the dictionary
            for inf, values in infos.items():
                ## print the key and the value
                print(f"For {inf}, the value is: {values}")
    return features


## define the function for the below
def get_feature_name_large(labels: str, verbosity: bool) -> dict:
    """
    Get the feature names from the .names file and returns the list of column names.

    The feature names start after the "initial_line": "7. Attribute Information: "
    and end before the "final_line": "8. Missing Attribute Values:"

    They are defined as:
        1. feature : possible values, or ranges
    Therefore we can split by the period and empty space (". "), and get:
        feature: possible values or ranges
    Ultimately, splitting by (":"), to get
        feature
    """
    gc.collect()
    holder = [None] * 23
    ## define a tracker to know which line to print:
    printing = False
    ## open the file and read
    norm_idx = -1
    with open(labels, "r") as f:
        for idx, line in enumerate(f):
            if line.startswith(
                "7. Attribute Information: (classes: edible=e, poisonous=p)"
            ):  ## the first line we want to start reading from
                printing = True
                if not line.startswith(str(norm_idx + 1)) and "=" in line:
                    printing = True
                    continue
                else:
                    printing = False
                    continue
            elif line.startswith(
                "8. Missing Attribute Values"
            ):  ## we want to skip this line and those that length is 0
                printing = False
                break  # quit file reading
            if printing:
                cline = line.strip()
                ## we get the 1. feature: possible values
                if ":" in cline and "=" in cline:
                    norm_idx += 1
                    ## clean the strings
                    ## string format is: "1. feature name: possible_value1=b, possible_value2=c, possible_value3=d,..."
                    ## remove whitespace
                    pattern = re.compile(r"\s+")
                    ## apply it
                    ncline = re.sub(pattern, "", cline)
                    ## remove the numbers
                    ncline = "".join(ncline.split(".")[-1])
                    ## append to the holder
                    holder[norm_idx] = ncline
    col_names = [x.split(":")[0] for x in holder]
    ## insert class at the position 0 of col_names
    ## number of features
    num_features = len(col_names)
    ## possible values for dictionary afterwards
    possible_values = [x.split(":")[1] for x in holder]
    ## put it all in a final dictionary
    features = {"large": {"n_features": num_features, "column_names": col_names}}
    ## dictionary to replace the possible values with the actual values

    ## convert each item of the list to a dictionary corresponding to the feature name
    ## to have {"feature_name"}: {"possible_value1": "b", "possible_value2": "c", "possible_value3": "d", ...}
    main_dict = {}
    for idx, item in enumerate(possible_values):
        ## split the item by the comma
        item = item.split(",")
        ## get the feature name
        feature_name = col_names[idx]
        ## get the possible values
        possible_values = [x.split("=")[1] for x in item]
        ## get the actual values
        actual_values = [x.split("=")[0] for x in item]
        ## create a dictionary
        d = dict(zip(possible_values, actual_values))
        ## add to the main dictionary
        main_dict[feature_name] = d

    if verbosity:
        ## go through the dictionary
        for _, infos in features.items():
            print(f"\nDataset Size: Large ")
            ## for each value in the dictionary
            for inf, values in infos.items():
                ## print the key and the value
                print(f"For {inf}, the value is: {values}")
    return features, main_dict
