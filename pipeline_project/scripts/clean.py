# %% tags=["parameters"]
# If this task has dependencies, list them them here
# (e.g. upstream = ['some_task']), otherwise leave as None.
upstream = ['get']

# This is a placeholder, leave it as None
product = None

# %%
# Parameters
upstream = {
    "get": {
        "nb": "C:\\Users\\Admin\\pipeline_project\\products\\get.ipynb",
        "data": "C:\\Users\\Admin\\pipeline_project\\products\\raw.csv",
    }
}
product = {
    "nb": "C:\\Users\\Admin\\pipeline_project\\products\\clean.ipynb",
    "data": "C:\\Users\\Admin\\pipeline_project\\products\\clean.csv",
}


# %%
import pandas as pd

# %%
df = pd.read_csv(upstream['get']['data'])

# %% [markdown]
# # Producing dummy variables for categorical data

# %% [markdown]
# Drop customerid columns because it does not add values to the analysis. Dropping will also reduce the dimensionality.

# %%
df = df.iloc[:,1:]
df.head(2)

# %%
# make all column names lower case
df.columns = map(str.lower, df.columns)
df.columns

# %%
dummydf = df[['gender','partner','dependents','phoneservice','multiplelines','internetservice','onlinesecurity','onlinebackup',

            'deviceprotection','techsupport','streamingtv','streamingmovies','contract','paperlessbilling','paymentmethod']]
dummydf.info()

# %%
# create pandas dummy variable for categorical features
dummy = pd.get_dummies(dummydf)
dummy.info()


# %%
dfcleaned = dummy.copy()
dfcleaned.to_csv(product['data'],index=False)

# %%



