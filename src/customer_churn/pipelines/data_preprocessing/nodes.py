"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.3
"""
import pandas as pd
from mrmr import mrmr_classif

def drop_column(df:pd.DataFrame):
    #Menghapus fitur dengan type data kategori
    df_model = df.select_dtypes(exclude = ['object', 'bool'])
    return df_model

def mrmr_selection(df:pd.DataFrame):
    selected_features = mrmr_classif(df.drop(['label', 'index'], axis = 1), df.label, K = 15)
    df_selected_features = df.loc[:, selected_features]
    df_model = pd.concat([df_selected_features, df.label], axis = 1)
    features = pd.DataFrame(selected_features, columns = ['kolom'])
    return df_model, features
