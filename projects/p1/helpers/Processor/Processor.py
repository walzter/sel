import pandas as pd 
import pickle
class Processor:
    def __init__(self, dataframe:pd.DataFrame, config:dict,size:str):
        self.data = dataframe
        self.config = config 
        self.size = size 
        self.dir = f"./{self.config['OUT_DIR']}/{self.size}_mapper.pkl"
        self.holder = None
        
        
    
    def _make_mapping_dict(self) -> dict:
        """
        Returns a dictionary with a mapping of: 
            - The categorical values in each column
            - The numerical map is going to be the index of unique values 
        """
        ## copy the dataframe
        _dataframe = self.data.copy(deep=True)
        ## holder 
        holder = dict()
        ## select the categorical columns
        cat_cols = _dataframe.select_dtypes("object")
        ## get all the unique values for each column 
        unique_vals = [self.data[x].unique() for x in cat_cols.columns]
        ## get all the maps
        mpf = [{e: ix for ix,e in enumerate(x)} for x in unique_vals] ## forwards (original -> encoded)
        mpb = [{ix: e for ix,e in enumerate(x)} for x in unique_vals] ## backwards (encoded -> original)
        ## make a dictionary of column_name: dictionary for each column name 
        ## map each element to the corresponding column 
        holder['forwards_mapper'] = dict(zip(cat_cols.columns, mpf))
        holder['backwards_mapper'] = dict(zip(cat_cols.columns, mpb))
        ## save the mapper
        self.holder = holder
        return holder
    @staticmethod
    def _forwards_backwards_map(dataframe:pd.DataFrame, mapping_dictionary:dict, direction:str) -> pd.DataFrame:
        """
        Mapping of the values in the original dataframe to numbers 
        
        PARAMS: 
        -------
        dataframe : the dataframeto be converted back 
        
        mapping dictionary : contains the directional mappings 
        
        RETURNS:
        --------
        c_dataframe : copy of the original dataframe 
        
        """
        ## copy the dataframe 
        gf = dataframe.copy()
        ## forward mapping
        if direction in ['forwards','fw','f']:
            _ = [gf[k].replace(v,inplace=True) for k, v in mapping_dictionary['forwards_mapper'].items()]
        elif direction in ['backwards','back','b']:
            _ = [gf[k].replace(v,inplace=True) for k, v in mapping_dictionary['backwards_mapper'].items()]
        else:
            raise ValueError("Direction of mapping not recognized. Not any of forwards, or backwards")
        return gf
    
    
    def _dict2pickle(self) -> None:
        """
        Saves a dictionary in a pickle format
        """
        with open(self.dir, 'wb') as file: 
            pickle.dump(self.holder, file)
        
    
    def _pickle2dict(self) -> None:
        """
        Saves a dictionary in a pickle format
        """
        with open(self.dir, 'rb') as f: 
            dd_small = pickle.load(f)
        return dd_small
    
    def _process(self):
        ## first get the mapping dictionary 
        self._make_mapping_dict()
        ## save the data 
        self._dict2pickle()
        ## 
        return self.holder 
    
    