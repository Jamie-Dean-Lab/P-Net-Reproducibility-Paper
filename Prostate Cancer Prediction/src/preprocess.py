import pandas as pd

def mut_binary(x : pd.DataFrame):
    """
    Preprocessing function to convert mutations into binary 

    args:
        x (DataFrame) : input DataFrame containing mutation data
    
    returns:
        DataFrame : modified DataFrame
    """
    x.loc[x.to_numpy() > 1] = 1
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
    x.loc[x.to_numpy() >= 0] = 0
    # remove single event due to noisiness similar to original P-Net paper
    x.loc[x.to_numpy() == -1] = 0
    x.loc[x.to_numpy() == -2] = 1
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
    x.loc[x.to_numpy() <= 0] = 0
    # remove single event due to noisiness similar to original P-Net paper
    x.loc[x.to_numpy() == 1] = 0
    x.loc[x.to_numpy() == 2] = 1
    return x

def cnv(x : pd.DataFrame):
    """
    Preprocessing function to filter CNV mutations to only include amplifications
    Ignores single events similar to original P-Net paper

    args:
        x (DataFrame) : input DataFrame containing mutation data
    
    returns:
        DataFrame : modified DataFrame
    """
    # remove single event due to noisiness similar to original P-Net paper
    x.loc[x.to_numpy() == 1] = 0
    x.loc[x.to_numpy() == 2] = 1
    x.loc[x.to_numpy() == -1] = 0
    x.loc[x.to_numpy() == -2] = -1
    return x