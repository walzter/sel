from helpers.config import make_config
from helpers.Cleaner.Cleaner import Cleaner
from helpers.RULES.RULES import RULES
import argparse

## parsing the arguments 
arguments = argparse.ArgumentParser('RULES')
## need data_size / split, and the dataset  

## the dataset size to use 
arguments.add_argument('-ds','--dataset_size',
                        help="The size of the dataset to chose:\nBreast-Cancer - small\nCars - medium\nMushrooms - large", type=str)
## the data split to chose 
arguments.add_argument('-ts', '--train_split',
                    help="The train-test split to do. It defaults to 10% Training / 90% Testing",
                    type=float, default=0.1)
args = arguments.parse_args()

def main():
    
    #data_size = args.dataset_size
    CONFIG = make_config()
    ## cleaning
    df = Cleaner(CONFIG[args.dataset_size], args.dataset_size, CONFIG["OUT_DIR"], CONFIG['VERBOSITY'])._make_dataframe()
    # ## processing OPTIONAL
    # proc = Processor(dataframe = df, config = CONFIG, size = data_size)
    # ## get the mapping 
    # small_mapper = proc._process()
    # ## mapping forwards 
    # dff = proc._forwards_backwards_map(df, small_mapper,direction='f') ## works with forwards, forw, f
    ## mapping backwards 
    ### dfb =  proc._forwards_backwards_map(small_df, proc.holder,direction='b') ## works with backwards, back, b 
    df.columns = list(range(0,len(df.columns)))
    
    ##### --- RULES
    rules = RULES(train_split = args.train_split, 
              dataframe = df,
              class_idx=CONFIG["CLASS_INDEX"],
              verbose=True,
              )
    #### --- RUNNING RULES 
    _ = rules._run_RULES()
    
    ## -- PREDICT
    rules._predict()
    
    return rules 

## if name == main 
if __name__ == "__main__":
    rules = main()
    