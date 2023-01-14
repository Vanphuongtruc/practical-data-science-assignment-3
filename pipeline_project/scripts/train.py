# %% tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = ['preprocess', 'feature_eng']

# This is a placeholder, leave it as None
product = None





# %%
# Parameters
upstream = {
    "preprocess": {
        "nb": "C:\\Users\\Admin\\pipeline_project\\products\\preprocess.ipynb",
        "data": "C:\\Users\\Admin\\pipeline_project\\products\\preprocess.csv",
    },
    "feature_eng": {
        "nb": "C:\\Users\\Admin\\pipeline_project\\products\\feature_eng.ipynb",
        "data": "C:\\Users\\Admin\\pipeline_project\\products\\feature_eng.csv",
    
    }
}
product = {
    "nb": "C:\\Users\\Admin\\pipeline_project\\products\\train.ipynb",
    "model": "C:\\Users\\Admin\\pipeline_project\\products\\model.pickle",
}



# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit, cross_val_score, StratifiedKFold 
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_curve, roc_auc_score, confusion_matrix,auc,cohen_kappa_score


from pathlib import Path
import pickle

# %%
def train_metrics(y_train,y_hat_train):
    print('\n')
    print(f'Training Precision: ', round(precision_score(y_train, y_hat_train),5))
    
    print(f'Training Recall: ',round(recall_score(y_train,y_hat_train),5))
    
    print(f'Training Accuracy: ',round(accuracy_score(y_train,y_hat_train),5))
    
    print(f'Training F1-score: ', round(f1_score(y_train, y_hat_train),5))
    
def test_metrics(y_test,y_hat_test):
    print('\n')
    print(f'Testing Precision: ', round(precision_score(y_test, y_hat_test),5))

   
    print(f'Testing Recall: ',round(recall_score(y_test, y_hat_test),5))
    
   
    print(f'Testing Accuracy: ',round(accuracy_score(y_test, y_hat_test),5))

   
    print(f'Testing F1-score: ', round(f1_score(y_test, y_hat_test),5))

# %% [markdown]
# # Classifiers performance comparison - with all features

# %%
# import data
dataf= pd.read_csv(upstream['preprocess']['data'])

# replace churn values with 0 and 1
dataf.churn.replace({'Yes':1, 'No':0}, inplace=True)

X = dataf.drop('churn', axis = 1)
y = dataf['churn'].copy()
# train - test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2)

# %%
clfs = []
clfs.append(("LogReg",
             Pipeline([("Scaler", StandardScaler()),
                       ("LogReg", LogisticRegression())])))
clfs.append(("LiReg",
             Pipeline([("Scaler", StandardScaler()),
                       ("LiReg", SGDClassifier())])))        
clfs.append(("DecisionTreeClassifier",
             Pipeline([("Scaler", StandardScaler()),
                       ("DecisionTrees", DecisionTreeClassifier())])))
clfs.append(("RandomForestClassifier",
             Pipeline([("Scaler", StandardScaler()),
                       ("RandomForest", RandomForestClassifier())])))
clfs.append(("KNN",
             Pipeline([("Scaler", StandardScaler()),
                       ("KNN", KNeighborsClassifier())])))



# %%
#gives model report in dataframe
def model_report(model,training_x,testing_x,training_y,testing_y,name) :
    model.fit(training_x,training_y)
    predictions  = model.predict(testing_x)
    accuracy     = accuracy_score(testing_y,predictions)
    recallscore  = recall_score(testing_y,predictions)
    precision    = precision_score(testing_y,predictions)
    roc_auc      = roc_auc_score(testing_y,predictions)
    f1score      = f1_score(testing_y,predictions) 
    kappa_metric = cohen_kappa_score(testing_y,predictions)
    
    df = pd.DataFrame({"Model"           : [name],
                       "Accuracy_score"  : [accuracy],
                       "Recall_score"    : [recallscore],
                       "Precision"       : [precision],
                       "f1_score"        : [f1score],
                       "Area_under_curve": [roc_auc],
                       
                      })
    
    return df

# %%
#'neg_mean_absolute_error', 'neg_mean_squared_error','r2'
scoring = "f1"
n_folds = 10
test_size = 0.2

results, names = [], []

for name, model in clfs:
    kfold = StratifiedKFold(n_splits=n_folds)
    cv = ShuffleSplit(n_splits=n_folds, test_size=test_size)
    cv_results = cross_val_score(model,
                                 X_train,
                                 y_train,
                                 cv=cv,
                                 scoring=scoring,
                                 n_jobs=-1)
    names.append(name)
    results.append(cv_results)
    msg = "%s: %f (+/- %f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = plt.figure(figsize=(15, 6))
fig.suptitle('Performance Comparison', fontsize=22)
ax = fig.add_subplot(111)


foo = pd.DataFrame({'Names':names,'Results':results})
result = foo.explode('Results').reset_index(drop=True)
result = result.assign(Names=result['Names'].astype('category'), 
                       Values=result['Results'].astype(np.float32))

sns.boxplot(x='Names', y='Results', data=result)
ax.set_xticklabels(names)
ax.set_xlabel("Model", fontsize=20)
ax.set_ylabel("Accuracy score", fontsize=18)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

plt.show()

# %%
results, names  = [], [] 

import plotly.figure_factory as ff
import plotly.offline as py

for name, model  in clfs:    
    names.append(name)
    model_results = model_report(model, X_train, X_test, y_train, y_test,name)
     #concat all models
    results.append(model_results)
    
    
model_performances = pd.concat(results)

table  = ff.create_table(np.round(model_performances,4))

py.iplot(table)

# %% [markdown]
# # Classifiers performance comparison - with selected features
# 

# %%
# import data
dataf= pd.read_csv(upstream['feature_eng']['data'])

# replace churn values with 0 and 1
dataf.churn.replace({'Yes':1, 'No':0}, inplace=True)

X = dataf.drop('churn', axis = 1)
y = dataf['churn'].copy()
# train - test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2)

# %%
#gives model report in dataframe
def model_report(model,training_x,testing_x,training_y,testing_y,name) :
    model.fit(training_x,training_y)
    predictions  = model.predict(testing_x)
    accuracy     = accuracy_score(testing_y,predictions)
    recallscore  = recall_score(testing_y,predictions)
    precision    = precision_score(testing_y,predictions)
    roc_auc      = roc_auc_score(testing_y,predictions)
    f1score      = f1_score(testing_y,predictions) 
    kappa_metric = cohen_kappa_score(testing_y,predictions)
    
    df = pd.DataFrame({"Model"           : [name],
                       "Accuracy_score"  : [accuracy],
                       "Recall_score"    : [recallscore],
                       "Precision"       : [precision],
                       "f1_score"        : [f1score],
                       "Area_under_curve": [roc_auc],
                       
                      })
    
    return df

# %%
#'neg_mean_absolute_error', 'neg_mean_squared_error','r2'
scoring = "f1"
n_folds = 10
test_size = 0.2

results, names = [], []

for name, model in clfs:
    kfold = StratifiedKFold(n_splits=n_folds)
    cv = ShuffleSplit(n_splits=n_folds, test_size=test_size)
    cv_results = cross_val_score(model,
                                 X_train,
                                 y_train,
                                 cv=cv,
                                 scoring=scoring,
                                 n_jobs=-1)
    names.append(name)
    results.append(cv_results)
    msg = "%s: %f (+/- %f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = plt.figure(figsize=(15, 6))
fig.suptitle('Performance Comparison', fontsize=22)
ax = fig.add_subplot(111)


foo = pd.DataFrame({'Names':names,'Results':results})
result = foo.explode('Results').reset_index(drop=True)
result = result.assign(Names=result['Names'].astype('category'), 
                       Values=result['Results'].astype(np.float32))

sns.boxplot(x='Names', y='Results', data=result)
ax.set_xticklabels(names)
ax.set_xlabel("Model", fontsize=20)
ax.set_ylabel("Accuracy score", fontsize=18)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

plt.show()

# %%
results, names  = [], [] 

for name, model  in clfs:    
    names.append(name)
    model_results = model_report(model, X_train, X_test, y_train, y_test,name)
     #concat all models
    results.append(model_results)
    
    
model_performances = pd.concat(results)

table  = ff.create_table(np.round(model_performances,4))

py.iplot(table)

# %% [markdown]
# Based on the comparision, Logistic Regression is the optimal model.

# %% [markdown]
# ### OVersampling Logistic Regression with hyperparameter tuning

# %%
from sklearn.model_selection import GridSearchCV, StratifiedKFold
#oversampling
from imblearn.over_sampling import SMOTE

# %%
# import data
datahyper= pd.read_csv(upstream['feature_eng']['data'])

# replace churn values with 0 and 1
datahyper.churn.replace({'Yes':1, 'No':0}, inplace=True)

X = datahyper.drop('churn', axis = 1)
y = datahyper['churn'].copy()
# train - test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2)

# %%
sm = SMOTE(sampling_strategy ='all')
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Before oversampling
unique, counts = np.unique(y_train, return_counts = True)
print(np.asarray((unique, counts)).T)

# After oversampling
unique, counts = np.unique(y_train_res, return_counts = True)
print(np.asarray((unique, counts)).T)

# %%
chosen_hyper = LogisticRegression(fit_intercept=False, C=1e12, solver='liblinear')
params = {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [100, 110, 120, 130, 140]}

skf = StratifiedKFold(n_splits = 10)
gscv = GridSearchCV(chosen_hyper, param_grid = params, n_jobs = -1, cv = skf)

LR_hyper = gscv.fit(X_train,y_train)

# %%
predicts = LR_hyper.predict(X_test)
print('Accuracy score: ', accuracy_score(y_test,predicts))
print(f'Roc_auc_score: ', roc_auc_score(y_test,predicts))
print(f'f1_score: ',f1_score(y_test,predicts))

# %%
Path(product['model']).write_bytes(pickle.dumps(LR_hyper))

# %%





