# %%
# If this task has dependencies, list them them here
# (e.g. upstream = ['some_task']), otherwise leave as None.
upstream = None

# This is a placeholder, leave it as None
product = None

# %%
# Parameters
product = {"nb": "C:\\Users\\Admin\\pipeline_project\\products\\new.ipynb", "data": "C:\\Users\\Admin\\pipeline_project\\products\\new.csv"}


# %%

input_path = 'C:\\Users\\Admin\\pipeline_project\\CustomerChurn.csv'

# %%
import pandas as pd

# %%
dfnew = pd.read_csv(input_path)
dfnew.head()

# %%
dfnew.to_csv(product['data'], index=False)

# %%



