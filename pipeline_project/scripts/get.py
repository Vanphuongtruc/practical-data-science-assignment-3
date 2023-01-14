# %% tags=["parameters"]
# If this task has dependencies, list them them here
# (e.g. upstream = ['some_task']), otherwise leave as None.
upstream = None

# This is a placeholder, leave it as None
product = None

# %%
# Parameters
product = {
    "nb": "C:\\Users\\Admin\\pipeline_project\\products\\get.ipynb",
    "data": "C:\\Users\\Admin\\pipeline_project\\products\\raw.csv",
}

# %%

input_path = 'C:\\Users\\Admin\\pipeline_project\\CustomerChurn.csv'

# %%
import pandas as pd

# %%
# Reading shopping data
X_train = pd.read_csv(input_path)
df = X_train.copy()
df.head()

# %%
df.to_csv(product['data'], index=False)

# %%






