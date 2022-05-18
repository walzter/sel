## Instruction of use 

0. Install the requirements.txt with pip install -r requirements.txt 
1. The RULES notebook can be used and then select the corresponding size and run the model. 
2. As a CLI and in the following way: 
    ```python 
    python run.py -ds small -ts 0.1 ## small & default value of ts 
    python run.py -ds medium -ts 0.7 ## medium  
    python run.py -ds large -ts 0.7 ## large
    Where the value after "-ds" is equal to the dataset size: ["small","medium","large"]
    While the value "-ts" is the training split used (the percentage is for training). 
    ```
