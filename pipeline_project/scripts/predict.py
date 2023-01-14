# %% tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = [ 'feature_eng']

# This is a placeholder, leave it as None
product = None
model = None

# %%
# Parameters
upstream = {
    "feature_eng": {
        "nb": "C:\\Users\\Admin\\pipeline_project\\products\\feature_eng.ipynb",
        "data": "C:\\Users\\Admin\\pipeline_project\\products\\feature_eng.csv",
    }
} 
product = {"nb": "C:\\Users\\Admin\\pipeline_project\\products\\predict.ipynb", 
        "data": "C:\\Users\\Admin\\pipeline_project\\products\\predicts.csv"}

model = "C:\\Users\\Admin\\pipeline_project\\products\\model.pickle"

# %%
from pathlib import Path
import pickle

import pandas as pd

model = pickle.loads(Path(model).read_bytes())


# %%
df = pd.read_csv(upstream['feature_eng']['data']).drop('churn', axis = 1)
y_pred = model.predict(df)

pd.DataFrame({'y_pred': y_pred}).to_csv(product['data'], index=False)

# %%



