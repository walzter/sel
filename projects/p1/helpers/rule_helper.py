## helper function 
def print_rule_ext(fcomb, av:int, _class:int, sat_idx:list, training_rows:int,columns:list):
    out = 'IF '
    for feature_index, attribute_value in zip(fcomb, av):
        out += str(columns[feature_index]) + ' = ' + str(attribute_value) + ' AND '
    txt_rules = f"{out[:-4]} THEN {_class}"
    coverage = len(sat_idx)
    pct_coverage = len(sat_idx) * 100 / training_rows
    txt_metrics = f"Coverage: {coverage} - Number of instances covered: {pct_coverage:.2f}%"
    txt_full = txt_rules + " | " + txt_metrics
    return txt_metrics, txt_full

def print_rule_small(fcomb, av:int, _class:int,columns:list):
    out = 'IF '
    for fidx, attributes in zip(fcomb, av):
        out += f"{columns[fidx]} = {attributes} AND "
    txt_rules = f"{out[:-4]} THEN {_class}"
    return txt_rules