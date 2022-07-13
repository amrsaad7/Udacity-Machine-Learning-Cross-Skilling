# TODO: Add import statements
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv('data.csv', header = None)
# print(train_data)
# print("done!")
X = train_data[train_data.columns[:6]]
# print(X)
# print("done!")
y = train_data[train_data.columns[6]]
# print(y)
# print("done!")
# TODO: Create the linear regression model with lasso regularization.
lasso_reg = Lasso()
lasso_reg = lasso_reg.fit(X, y)
# TODO: Fit the model.


# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)