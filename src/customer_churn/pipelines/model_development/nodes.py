"""
This is a boilerplate pipeline 'model_development'
generated using Kedro 0.18.3
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import resample
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import make_scorer, roc_auc_score

def split_data(df:pd.DataFrame):
    training, testing = train_test_split(df, test_size = 0.2, random_state = 42)
    return training, testing

def bootstrap(df:pd.DataFrame):
    loop = 4200
    coe_atl_s3 = df.copy()
    df_training_under = pd.DataFrame()
    for i in range(loop):
        sample = resample(coe_atl_s3, replace=False, n_samples = 15, random_state = i+11)
        df_training_under = pd.concat([df_training_under, sample], axis = 0)
    return df_training_under

def split_xy(training:pd.DataFrame, testing:pd.DataFrame):
    X_train = training.drop(['label'], axis = 1)
    y_train = training.label
    X_test = testing.drop(['label'], axis = 1)
    y_test = testing.label
    return X_train, y_train, X_test, y_test

def model_cat(X_train:pd.DataFrame, y_train:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.DataFrame):
    """Machine learning process consists of 
    data training, and data testing process (i.e. prediction) with Catboost Algorithm
    """
    # prepare a new DataFrame
    training = pd.DataFrame(X_train).copy()
    testing = pd.DataFrame(X_test).copy()

    clf = CatBoostClassifier()
    params = {'iterations': [25, 30],
              'learning_rate': [0.001, 0.005, 0.01],
              'depth': [4, 5, 6],
              'loss_function': ['LogLoss', 'CrossEntropy'],
              'l2_leaf_reg': np.logspace(-20, -19, 3),
              'leaf_estimation_iterations': [10],
    #           'eval_metric': ['Accuracy'],
    #           'use_best_model': ['True'],
              'logging_level':['Silent'],
              'random_strength':[0,5,10],
              'random_seed': [33, 42]
             }
    roc_auc_ovr_scorer = make_scorer(roc_auc_score, needs_proba=True,
                                     multi_class='ovr')
    clf_grid = GridSearchCV(estimator=clf, param_grid=params, scoring = roc_auc_ovr_scorer, cv=5)
    clf_grid.fit(X_train, y_train)
    best_param = clf_grid.best_params_

    model_1 = CatBoostClassifier(iterations=best_param['iterations'],
                            learning_rate=best_param['learning_rate'],
                            depth=best_param['depth'],
                            loss_function=best_param['loss_function'],
                            l2_leaf_reg=best_param['l2_leaf_reg'],
                            eval_metric='Accuracy',
                            leaf_estimation_iterations=10,
                            use_best_model=True,
                            logging_level='Silent',
                            random_strength=best_param['random_strength'],
                            random_seed=best_param['random_seed']
                            )

    train_pool = Pool(X_train, y_train)
    
    model = model_1.fit(train_pool, eval_set = (X_test, y_test))

    prediction_train = model.predict(X_train)
    probability_train = model.predict_proba(X_train)
    training['prediction'] = prediction_train
    training['probability'] = probability_train[:,1] 

    prediction_test = model.predict(X_test)
    probability_test = model.predict_proba(X_test)
    testing['prediction'] = prediction_test
    testing['probability'] = probability_test[:,1]
          
    # add the churn and target class into dataframe as validation data
    training['label'] = y_train
    testing['label'] = y_test

    return model, training, testing