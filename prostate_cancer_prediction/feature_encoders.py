import pandas as pd


def mut_binary(x: pd.DataFrame):
    """
    Preprocessing function to convert mutations into binary 

    args:
        x (DataFrame) : input DataFrame containing mutation data

    returns:
        DataFrame : a new DataFrame with the encoding applied (input is not modified)
    """
    x = x.copy()
    x[x > 1.0] = 1.0
    return x


def cnv_del(x: pd.DataFrame):
    """
    Preprocessing function to filter CNV mutations to only include deletions.
    Ignores single events similar to original P-Net paper

    args:
        x (DataFrame) : input DataFrame containing mutation data

    returns:
        DataFrame : a new DataFrame with the encoding applied (input is not modified)
    """
    x = x.copy()
    # Ignore amplifications aka positive CNV
    x[x >= 0.0] = 0.0
    # remove single event due to noisiness similar to original P-Net paper
    x[x == -1.0] = 0.0
    x[x == -2.0] = 1.0
    return x


def cnv_amp(x: pd.DataFrame):
    """
    Preprocessing function to filter CNV mutations to only include amplifications
    Ignores single events similar to original P-Net paper

    args:
        x (DataFrame) : input DataFrame containing mutation data

    returns:
        DataFrame : a new DataFrame with the encoding applied (input is not modified)
    """
    x = x.copy()
    # Ignore deletions aka negative CNV
    x[x <= 0.0] = 0.0
    # remove single event due to noisiness similar to original P-Net paper
    x[x == 1.0] = 0.0
    x[x == 2.0] = 1.0
    return x


def cnv(x: pd.DataFrame):
    """
    Preprocessing function to collapse CNV values into a signed indicator
    keeping only high-level events. Double deletions (-2) map to -1 and
    double amplifications (2) map to 1, while single events (-1, 1) are
    dropped to 0 due to noisiness similar to the original P-Net paper.
    0 is unchanged and any other value is left as-is.

    args:
        x (DataFrame) : input DataFrame containing CNV data

    returns:
        DataFrame : a new DataFrame with the encoding applied (input is not modified)
    """
    x = x.copy()
    x[x == -1.] = 0.0
    x[x == -2.] = -1.0
    x[x == 1.] = 0.0
    x[x == 2.] = 1.0
    return x

def cnv_signed(x: pd.DataFrame):
    """
    Preprocessing function to collapse CNV values into a signed indicator.
    Any amplification (> 1) maps to 1 and any deletion (< 0) maps to -1,
    leaving 0 and single-copy events unchanged. Unlike cnv, single events
    are retained rather than dropped.

    args:
        x (DataFrame) : input DataFrame containing CNV data

    returns:
        DataFrame : a new DataFrame with the encoding applied (input is not modified)
    """
    x = x.copy()
    x[x > 1.] = 1.
    x[x < 0.] = -1.
    return x


def cnv_amp_from_signed(x: pd.DataFrame):
    """
    Extract amplifications from already-collapsed signed CNV ({-1, 0, 1}) as a
    binary {0, 1} indicator (amplification -> 1, otherwise 0). This is the signed
    counterpart of cnv_amp: use it when the input has already been collapsed to
    the signed indicator (e.g. the combined external-validation matrix). Applying
    the raw cnv_amp to such data would instead drop the single-level +1 values and
    zero everything out.

    args:
        x (DataFrame) : input DataFrame containing signed CNV data

    returns:
        DataFrame : a new DataFrame with the encoding applied (input is not modified)
    """
    x = x.copy()
    x[x < 0.0] = 0.0
    return x


def cnv_del_from_signed(x: pd.DataFrame):
    """
    Extract deletions from already-collapsed signed CNV ({-1, 0, 1}) as a binary
    {0, 1} indicator (deletion -> 1, otherwise 0). Signed counterpart of cnv_del
    (see cnv_amp_from_signed).

    args:
        x (DataFrame) : input DataFrame containing signed CNV data

    returns:
        DataFrame : a new DataFrame with the encoding applied (input is not modified)
    """
    x = x.copy()
    x[x > 0.0] = 0.0
    x[x < 0.0] = 1.0
    return x
