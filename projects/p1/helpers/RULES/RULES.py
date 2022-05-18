import pandas as pd 
import numpy as np 
import itertools as it
from helpers.rule_helper import print_rule_ext, print_rule_small
from termcolor import colored 
class RULES:
    def __init__(self, train_split:float, dataframe: pd.DataFrame, verbose:bool, class_idx:int):
        self.data = dataframe ## dataframe to read 
        self.size = self.data.shape[0] ## the number of instances
        self.verbose = verbose # verbosity 
        self.train_split = train_split # % of data to use for training 
        self.test_split = 1-self.train_split # % of data to use for testing 
        self.train_size = int(self.train_split * self.size)  ## train size 
        self.test_size = self.size - self.train_size         ## test size 
        self.cols = self.data.shape[1]  ## number of attributes 
        self.class_idx = class_idx ## class index 
        self.features = list(self.data) ## the columns 
        self.f_idx = [x for x in self.features if x!=self.class_idx] ## columns without class 
        self.xtrain, self.xtest = self._split_data() ## train test data 
        self.rules = [] ## all the rules 
        self.classify_corr = 0 
        self.rules_used = []
        self.final_accuracy = 0
    
    
    ## define a function to split the data 
    
    def _split_data(self):
        """Splits the data """
        ## shuffle the data 
        data = self.data.values
        np.random.seed(52)
        np.random.shuffle(data)
        #train_size = int(self.size * self.train_size)
        xtrain, xtest = data[:self.train_size], data[self.train_size:]
        if self.verbose:        
            ## feedback on the number of 
            print(f"\nTRAINING-TESTING SPLIT = {self.train_split*100}%/{self.test_split*100}%")
            print(f"TRAINING - Number of Samples = {self.train_size}")
            print(f"TESTING  - Number of Samples = {self.test_size}")
        return xtrain, xtest
    
    def _run_RULES(self):
        """Runs the RULES algorithm """
        not_c = self.xtrain.copy() # in the beginning the whole training set is not_classified
        # Continue as long as there are not_classified examples.
        while not_c.shape[0]:
            for nc in range(1, self.cols):
                all_in_same_class = None # just declare it outside the below loop so we can propagate the break
                for combx in it.combinations(self.f_idx, nc):
                    sat_idx = [0]
                    all_in_same_class = True
                    for idx in range(1, not_c.shape[0]):
                        conditions_satisfied = True
                        for feature_index in combx:
                            if not_c[0][feature_index] != not_c[idx][feature_index]:
                                conditions_satisfied = False
                                break
                        if conditions_satisfied:
                            if not_c[0][self.class_idx] != not_c[idx][self.class_idx]:
                                all_in_same_class = False
                                break
                            sat_idx.append(idx)
                    # Create a new rule only if all instances that satisfy the condition
                    # are in the same class
                    if all_in_same_class:
                        new_rule = (combx,
                                    [not_c[0][feature_index] for feature_index in combx],not_c[0][self.class_idx])
                        metrics_txt, rule_txt = print_rule_ext(fcomb = new_rule[0],
                                                  av = new_rule[1],
                                                  _class = new_rule[2],
                                                  sat_idx = sat_idx,
                                                  training_rows = self.train_size,
                                                  columns = self.features)
                        if self.verbose:
                            print(colored(rule_txt, "green"),colored(metrics_txt, "red"))
                        ## keep this rule 
                        self.rules.append(new_rule)
                        ## remove the ones that are not classified 
                        not_c = np.delete(not_c, sat_idx, 0)
                        break
                if all_in_same_class:
                    break
        ## update the parameters         
        self.good_rules = [0] * len(self.rules)
        self.rules_used = [0] * len(self.rules)
        
        

    def _predict(self):
        """Predicts with the RULES Algorithm
        
        This will iterate over the test split of the data. 
        It will then find 
        """
        ## define the predict function 
        # Finally use our rules to predict class labels on the test dataset
        for datapoint in self.xtest:
            for idx, r in enumerate(self.rules):
                CONDITION = True
                for (x, y) in zip(r[0], range(len(r[0]))):
                    if datapoint[x] != r[1][y]:
                        CONDITION = False
                        break
                if CONDITION:
                    self.rules_used[idx] += 1
                    # Compare the predicted label to the actual label
                    if datapoint[self.class_idx] == r[2]:
                        self.classify_corr += 1
                        self.rules_used[idx] += 1
                    break

        if self.verbose:
            for x, y, z in zip(self.rules, self.rules_used, self.rules_used):
                ## unpack the 
                res_txt = print_rule_small(*x, self.features)
                txt_to_print = f"Coverage: {y} | Pct-Coverage: {y / self.test_size * 100:.2f} | Precision {z / y * 100}"
                print(colored(res_txt, "green"),colored(txt_to_print, "red"))
        ## final accuracy of the model 
        accuracy = self.classify_corr / self.test_size * 100
        self.final_accuracy += accuracy 
        ## verbosity 
        if self.verbose: 
            print(colored(f"FINAL ACCURACY: {self.final_accuracy:.2f}%",'red'))
            print(colored(f"TOTAL NUMBER OF RULES FOUND: {len(self.rules)}", 'red'))

        