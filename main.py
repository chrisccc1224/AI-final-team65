import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
tree_num = np.arange(25, 55, 3)
depth_num = np.arange(20, 55, 3)
mse_list = np.zeros((len(depth_num), len(tree_num)))
r2_list = np.zeros((len(depth_num), len(tree_num)))
for i, depth in enumerate(depth_num):
    print(f"Training Random Forest with max depth {depth}")
    for j, n_tree in enumerate(tree_num):
        print(f"Training Random Forest with {n_tree} trees")
        # Initialize the RandomForest model
        model = RandomForest(
            n_trees=n_tree,
            max_depth=depth,
            max_features=13
        )
        model.fit(X_train, y_train)
        # predict
        y_pred=model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse_list[i, j] = mse
        r2_list[i, j] = r2

mse_list = pd.DataFrame(mse_list, index=depth_num, columns=tree_num)
r2_list = pd.DataFrame(r2_list, index=depth_num, columns=tree_num)
# evaluate
plt.figure(figsize=(12, 8))
sns.heatmap(mse_list, annot=True, fmt=".4f", cmap='viridis', xticklabels=tree_num, yticklabels=depth_num, annot_kws={"size": 6})
plt.xlabel('Number of Trees')
plt.ylabel('Depth of Trees')
plt.title('MSE heatmap')
plt.savefig('mse_heatmap.png')
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(r2_list, annot=True, fmt=".4f", cmap='viridis', xticklabels=tree_num, yticklabels=depth_num, annot_kws={"size": 6})
plt.xlabel('Number of Trees')
plt.ylabel('Depth of Trees')
plt.title('R^2 heatmap')
plt.savefig('r2_heatmap.png')
plt.show()
