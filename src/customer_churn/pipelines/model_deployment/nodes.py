"""
This is a boilerplate pipeline 'model_deployment'
generated using Kedro 0.18.3
"""
import pandas as pd
import pickle

def drop_column(df:pd.DataFrame):
    #Menghapus fitur dengan type data kategori
    df_model = df.select_dtypes(exclude = ['object', 'bool'])
    return df_model

def data_oot(data:pd.DataFrame, features:pd.DataFrame):
  data_gojek_1 = data.loc[(data.from_access == 'gojek')]
  data_grab_1 = data.loc[(data.from_access == 'grape')]

  selected_features = features['kolom'].tolist()

  data_gojek = drop_column(data_gojek_1)
  data_grab = drop_column(data_grab_1)
  data_gojek = data_gojek.loc[:, selected_features]
  data_grab = data_grab.loc[:, selected_features]

  data_gojek['label'] = 0
  data_gojek.loc[(data_gojek_1.to_access == 'grape'), 'label'] = 1
  data_grab['label'] = 0
  data_grab.loc[(data_grab_1.to_access == 'gojek'), 'label'] = 1

  return data_gojek, data_grab

def split_xy_oot(df:pd.DataFrame):
  X = df.drop('label', axis = 1)
  y = df.label
  return X, y

def predict_proba(X:pd.DataFrame, y:pd.DataFrame, model:pickle):
  prediction = pd.DataFrame(X).copy()
  probability = model.predict_proba(X)
  prediction['probability'] = probability[:,1] 
  prediction['label'] = y
  return prediction