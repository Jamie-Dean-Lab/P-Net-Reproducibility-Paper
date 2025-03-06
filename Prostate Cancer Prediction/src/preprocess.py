import pandas as pd

def mut_binary(x : pd.DataFrame):
    """
    Preprocessing function to convert mutations into binary 

    args:
        x (DataFrame) : input DataFrame containing mutation data
    
    returns:
        DataFrame : modified DataFrame
    """
    x[x > 1.0] = 1.0
    return x

def cnv_del(x : pd.DataFrame):
    """
    Preprocessing function to filter CNV mutations to only include deletions.
    Ignores single events similar to original P-Net paper

    args:
        x (DataFrame) : input DataFrame containing mutation data
    
    returns:
        DataFrame : modified DataFrame
    """
    # Ignore amplifications aka positive CNV
    x[x >= 0] = 0.0
    # remove single event due to noisiness similar to original P-Net paper
    x[x == -1.0] = 0.0
    x[x == -2.0] = 1.0
    return x

def cnv_amp(x : pd.DataFrame):
    """
    Preprocessing function to filter CNV mutations to only include amplifications
    Ignores single events similar to original P-Net paper

    args:
        x (DataFrame) : input DataFrame containing mutation data
    
    returns:
        DataFrame : modified DataFrame
    """
    # Ignore deletions aka negative CNV
    x[x <= 0.0] = 0.0
    # remove single event due to noisiness similar to original P-Net paper
    x[x == 1.0] = 0.0
    x[x == 2.0] = 1.0
    return x