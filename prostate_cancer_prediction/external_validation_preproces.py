import pandas as pd
import numpy as np

def generate_external_validation_labels():
    # Process metastatic
    common_samples = ['MO_1008',
                        'MO_1012',
                        'MO_1013',
                        'MO_1014',
                        'MO_1015',
                        'MO_1020',
                        'MO_1040',
                        'MO_1074',
                        'MO_1084',
                        'MO_1094',
                        'MO_1095',
                        'MO_1096',
                        'MO_1114',
                        'MO_1118',
                        'MO_1124',
                        'MO_1128',
                        'MO_1130',
                        'MO_1132',
                        'MO_1139',
                        'MO_1161',
                        'MO_1162',
                        'MO_1176',
                        'MO_1179',
                        'MO_1184',
                        'MO_1192',
                        'MO_1202',
                        'MO_1215',
                        'MO_1219',
                        'MO_1232',
                        'MO_1241',
                        'MO_1244',
                        'MO_1249',
                        'MO_1262',
                        'MO_1277',
                        'MO_1316',
                        'MO_1337',
                        'MO_1339',
                        'MO_1410',
                        'MO_1421',
                        'MO_1447',
                        'MO_1460',
                        'MO_1473',
                        'TP_2001',
                        'TP_2010',
                        'TP_2020',
                        'TP_2032',
                        'TP_2034',
                        'TP_2054',
                        'TP_2060',
                        'TP_2061',
                        'TP_2064',
                        'TP_2069',
                        'TP_2077',
                        'TP_2078',
                        'TP_2079']

    prostate_samples = ['MO_1008', 'MO_1012', 'MO_1013', 'MO_1014', 'MO_1015', 'MO_1020', 'MO_1040', 'MO_1066',
                            'MO_1074', 'MO_1084',
                            'MO_1093', 'MO_1094', 'MO_1095', 'MO_1096', 'MO_1112', 'MO_1114', 'MO_1118', 'MO_1124',
                            'MO_1128', 'MO_1130',
                            'MO_1132', 'MO_1139', 'MO_1161', 'MO_1162', 'MO_1176', 'MO_1179', 'MO_1184', 'MO_1192',
                            'MO_1200', 'MO_1201',
                            'MO_1202', 'MO_1214', 'MO_1215', 'MO_1219', 'MO_1221', 'MO_1232', 'MO_1240', 'MO_1241',
                            'MO_1244', 'MO_1249',
                            'MO_1260', 'MO_1262', 'MO_1263', 'MO_1277', 'MO_1307', 'MO_1316', 'MO_1336', 'MO_1337',
                            'MO_1339', 'MO_1410',
                            'MO_1420', 'MO_1421', 'MO_1437', 'MO_1443', 'MO_1446', 'MO_1447', 'MO_1460', 'MO_1469',
                            'MO_1472', 'MO_1473',
                            'MO_1482', 'MO_1490', 'MO_1492', 'MO_1496', 'MO_1499', 'MO_1510', 'MO_1511', 'MO_1514',
                            'MO_1517', 'MO_1541',
                            'MO_1543', 'MO_1553', 'MO_1556', 'TP_2001', 'TP_2009', 'TP_2010', 'TP_2020', 'TP_2032',
                            'TP_2034', 'TP_2037',
                            'TP_2043', 'TP_2054', 'TP_2060', 'TP_2061', 'TP_2064', 'TP_2069', 'TP_2077', 'TP_2078',
                            'TP_2079', 'TP_2080',
                            'TP_2081', 'TP_2090', 'TP_2093', 'TP_2096', 'TP_2156']

    met500_samples = list(set(prostate_samples).difference(common_samples))
    valid_cnv = pd.read_csv("prostate_cancer_prediction/data/_database/prostate/external_validation/Met500/Met500_cnv.txt", sep="\t", index_col=0).fillna(0)
    valid_mut = pd.read_csv("prostate_cancer_prediction/data/_database/prostate/external_validation/Met500/Met500_mut_matrix.csv", index_col=0).fillna(0)
    valid_cnv.T.to_csv("prostate_cancer_prediction/data/_database/prostate/external_validation/Met500/Met500_cnv_processed.csv")
    valid_mut.index = valid_mut.index.str.split(".").str[0]
    valid_mut.to_csv("prostate_cancer_prediction/data/_database/prostate/external_validation/Met500/Met500_mut_matrix_processed.csv")
    pd.DataFrame({"metastatic": 1}, index=met500_samples).to_csv("prostate_cancer_prediction/data/_database/prostate/external_validation/Met500/Met500_labels.csv")

    # Process primary
    valid_cnv = pd.read_csv("prostate_cancer_prediction/data/_database/prostate/external_validation/PRAD/cnv_matrix.csv", index_col=0).fillna(0)
    valid_mut = pd.read_csv("prostate_cancer_prediction/data/_database/prostate/external_validation/PRAD/mut_matrix.csv", index_col=0).fillna(0)
    PRAD_samples = list(set(valid_cnv.index.to_list() + valid_mut.index.to_list()))
    pd.DataFrame({"metastatic": 0}, index=PRAD_samples).to_csv("prostate_cancer_prediction/data/_database/prostate/external_validation/PRAD/PRAD_labels.csv")

generate_external_validation_labels()

def get_balanced_training_sets():
    df = pd.read_csv("prostate_cancer_prediction/data/_database/prostate/processed/response_paper.csv", index_col=0)
    index_pos = np.where(df["response"] == 1)[0]
    index_neg = np.where(df["response"] == 0)[0]
    n_pos = index_pos.shape[0]
    index_neg1 = index_neg[0:n_pos]
    index_neg2 = index_neg[n_pos:]
    df.iloc[np.concatenate([index_pos, index_neg1]), :].to_csv("prostate_cancer_prediction/data/_database/prostate/processed/response_paper_external_validation_1.csv")
    df.iloc[np.concatenate([index_pos, index_neg2]), :].to_csv("prostate_cancer_prediction/data/_database/prostate/processed/response_paper_external_validation_2.csv")