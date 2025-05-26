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
tree_num = np.arange(4, 71, 3)
mse_list = [0] * len(tree_num)
r2_list = [0] * len(tree_num)
for _ in range(10):
    for i in tree_num:
        print(f"Training Random Forest with {i} trees")
        # Initialize the RandomForest model
        model = RandomForest(
            n_trees=i,
            max_depth=5,
            max_features='sqrt'
        )
        model.fit(X_train, y_train)
        # predict
        y_pred=model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        """print(y_pred[:10])  # Print first 10 predictions
        print(f"Mean Squared Error: {mse}")
        print(f"R^2 Score: {r2}")"""
        mse_list[i // 3 - 1] += mse
        r2_list[i // 3 - 1] += r2

mse_list = np.array(mse_list)
r2_list = np.array(r2_list)
# evaluate
plt.plot(tree_num, mse_list / 10, label='MSE', marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('MSE')
plt.title('Number of Trees vs MSE')
plt.legend()
plt.savefig('mse_vs_trees.png')
plt.show()

plt.plot(tree_num, r2_list / 10, label='R^2', marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('R^2')
plt.title('Number of Trees vs R^2')
plt.legend()
plt.savefig('r2_vs_trees.png')
plt.show()
