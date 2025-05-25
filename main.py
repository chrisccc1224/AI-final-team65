import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from random_forest import DecisionTree

from random_forest import RandomForest
from pre import preprocess

df,scores=preprocess()


X_train,X_test,y_train,y_test=train_test_split(df,scores,test_size=0.2,shuffle=True,random_state=42)

## model from sklearn
model=RandomForestRegressor(
    n_estimators=50,
    max_depth=None,
    max_features='sqrt'
)
model.fit(X_train, y_train)
# predict
y_pred=model.predict(X_test)

# evaluate

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")


# Decision tree
# model = DecisionTree(
#     max_depth=None,
#     max_features=None,
# )
# model.fit(X_train,y_train)
# y_pred=model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error: {mse}")
# print(f"R^2 Score: {r2}")


# our random forest
model = RandomForest(
    n_trees=10,#check 50
    max_depth=5,
    max_features='sqrt'
)
model.fit(X_train, y_train)
# predict
y_pred=model.predict(X_test)

# evaluate

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(y_pred[:10])  # Print first 10 predictions
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
