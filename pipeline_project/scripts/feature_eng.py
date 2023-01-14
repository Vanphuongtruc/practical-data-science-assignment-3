# %% tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = ['clean','preprocess']

product = None


# %%
# Parameters
upstream = {
    "clean": {
        "nb": "C:\\Users\\Admin\\pipeline_project\\products\\clean.ipynb",
        "data": "C:\\Users\\Admin\\pipeline_project\\products\\clean.csv",
    },
    "preprocess": {
        "nb": "C:\\Users\\Admin\\pipeline_project\\products\\preprocess.ipynb",
        "data": "C:\\Users\\Admin\\pipeline_project\\products\\preprocess.csv",
    
    }
}

product = {
    "nb": "C:\\Users\\Admin\\pipeline_project\\products\\feature_eng.ipynb",
    "data": "C:\\Users\\Admin\\pipeline_project\\products\\feature_eng.csv",
}

# %%
import pandas as pd
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split

# %%
dfcleaned = pd.read_csv(upstream['clean']['data'])
dfpreprocess = pd.read_csv(upstream['preprocess']['data'])

# replace churn values with 0 and 1
dfpreprocess.churn.replace({'Yes':1, 'No':0}, inplace=True)

X = dfcleaned.copy()
X_rc = dfpreprocess.drop('churn', axis = 1)
y = dfpreprocess['churn'].copy()

# %% [markdown]
# # Finding Important Features and remove them from the dataframe

# %%
list_one =[]
feature_ranking = SelectKBest(chi2, k=5)
fit = feature_ranking.fit(X, y)

fmt = '%-8s%-20s%s'

for i, (score, feature) in enumerate(zip(feature_ranking.scores_, X.columns)):
    list_one.append((score, feature))
    
dfObj = pd.DataFrame(list_one) 
dfObj.sort_values(by=[0], ascending = False)

# %%
# dropping the last 5
X_rc.drop(['multiplelines_No phone service','phoneservice_No','gender_Female','gender_Male','phoneservice_Yes'],axis=1,inplace=True)

# %%
X_rc.join(y).to_csv(product['data'], index=False)

# %%





