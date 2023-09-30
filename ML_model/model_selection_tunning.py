import pandas as pd
import numpy as np
import seaborn
import difflib
from scipy import stats
import time
import category_encoders as encoders
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, OneHotEncoder, OrdinalEncoder, \
    LabelEncoder
from sklearn.impute import KNNImputer
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, \
    r2_score
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import GridSearchCV
import os
from lce import LCERegressor, LCEClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC, LinearSVC, LinearSVR
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,Matern, WhiteKernel

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc, precision_score, \
    recall_score, f1_score, confusion_matrix, accuracy_score,mean_absolute_percentage_error,mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plot

regression_model={"LR":LinearRegression(),
     "KNN":KNeighborsRegressor(),
     "SVR":SVR(),
     "DT":DecisionTreeRegressor(),
     "RF":RandomForestRegressor(),
     "GPR":GaussianProcessRegressor(kernel=1.0 * RBF(1.0)),
     "LGBM":LGBMRegressor(),
     "XGB":XGBRegressor(),
     "LCE": LCERegressor(),
     "CatBoost":CatBoostRegressor(verbose=False),
     "MLP":MLPRegressor(),
     "TabNet":TabNetRegressor()
}

###库存插补，实在无法插补就不要了
# data=pd.read_csv("adspendweekends.csv").dropna()
#data=pd.read_csv("adspendweekdays.csv").dropna()
data=pd.read_csv("sku15.csv").dropna(axis=0)
print(data)
# data=pd.read_csv("pcaweekend.csv").dropna()
# target_list= "unitsordered"
X=np.array(data[data.columns.values[0:-1]])
y=np.array(data[data.columns.values[-1]])
x_train, x_test, y_train, y_test = train_test_split(X,y,
                                                        test_size=0.2,
                                                        #random_state=6,
                                                        shuffle=True)


###MINMAX,标准化
# 实例化一个归一化对象
# scaler = MinMaxScaler()
# X_minmax = scaler.fit_transform(X)
# scaler = StandardScaler()
# # X_standardized = scaler.fit_transform(X)
# poly = PolynomialFeatures(degree=2)  # 2 阶多项式
# X_poly = poly.fit_transform(X)

###
# print(X,y)




####模型筛选框架
import sys
f= open("output.csv","w")
sys.stdout=f
print("model","mape","mape_train","mae","mae_train","mse","mse_train","r2","r2_train",sep=",")
# 初始化LR模型
#print("LinearRegression")
model = LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_output=model.predict(x_train)
mape = mean_absolute_percentage_error(y_test+1, y_pred)
mape_train = mean_absolute_percentage_error(y_train+1, y_output)
#print(f"Test Set Mean Absolute Percentage Error: {mape} Train Set Mean Absolute Percentage Error: {mape_train}")
mae_train = mean_absolute_error(y_train, y_output)
mae = mean_absolute_error(y_test, y_pred)
#print(f"Test Set Mean Absolute Error: {mae} Train Set Mean Absolute Error: {mae_train}")
mse_train = mean_squared_error(y_train, y_output)
mse = mean_squared_error(y_test, y_pred)
#print(f"Test Set Mean Squared Error: {mse} Train Set Mean Squared Error: {mse_train}")
r2_train = r2_score(y_train, y_output)
r2 = r2_score(y_test, y_pred)
#print(f"Test Set R2 : {r2} Train Set R2 : {r2_train}")
print("LR",mape,mape_train,mae,mae_train,mse,mse_train,r2,r2_train,sep=",")

#print("SVR")
model = SVR()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_output=model.predict(x_train)
mape = mean_absolute_percentage_error(y_test+1, y_pred)
mape_train = mean_absolute_percentage_error(y_train+1, y_output)
#print(f"Test Set Mean Absolute Percentage Error: {mape} Train Set Mean Absolute Percentage Error: {mape_train}")
mae_train = mean_absolute_error(y_train, y_output)
mae = mean_absolute_error(y_test, y_pred)
#print(f"Test Set Mean Absolute Error: {mae} Train Set Mean Absolute Error: {mae_train}")
mse_train = mean_squared_error(y_train, y_output)
mse = mean_squared_error(y_test, y_pred)
#print(f"Test Set Mean Squared Error: {mse} Train Set Mean Squared Error: {mse_train}")
r2_train = r2_score(y_train, y_output)
r2 = r2_score(y_test, y_pred)
#print(f"Test Set R2 : {r2} Train Set R2: {r2_train}")
print("SVR",mape,mape_train,mae,mae_train,mse,mse_train,r2,r2_train,sep=",")

# print("KNN")
model = KNeighborsRegressor(n_neighbors=11)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_output=model.predict(x_train)
mape = mean_absolute_percentage_error(y_test+1, y_pred)
mape_train = mean_absolute_percentage_error(y_train+1, y_output)
# print(f"Test Set Mean Absolute Percentage Error: {mape} Train Set Mean Absolute Percentage Error: {mape_train}")
mae_train = mean_absolute_error(y_train, y_output)
mae = mean_absolute_error(y_test, y_pred)
# print(f"Test Set Mean Absolute Error: {mae} Train Set Mean Absolute Error: {mae_train}")
mse_train = mean_squared_error(y_train, y_output)
mse = mean_squared_error(y_test, y_pred)
# print(f"Test Set Mean Squared Error: {mse} Train Set Mean Squared Error: {mse_train}")
r2_train = r2_score(y_train, y_output)
r2 = r2_score(y_test, y_pred)
# print(f"Test Set R2: {r2} Train Set R2: {r2_train}")
print("KNN",mape,mape_train,mae,mae_train,mse,mse_train,r2,r2_train,sep=",")

# print("GPR")
model = GaussianProcessRegressor()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_output=model.predict(x_train)
mape = mean_absolute_percentage_error(y_test+1, y_pred)
mape_train = mean_absolute_percentage_error(y_train+1, y_output)
# print(f"Test Set Mean Absolute Percentage Error: {mape} Train Set Mean Absolute Percentage Error: {mape_train}")
mae_train = mean_absolute_error(y_train, y_output)
mae = mean_absolute_error(y_test, y_pred)
# print(f"Test Set Mean Absolute Error: {mae} Train Set Mean Absolute Error: {mae_train}")
mse_train = mean_squared_error(y_train, y_output)
mse = mean_squared_error(y_test, y_pred)
# print(f"Test Set Mean Squared Error: {mse} Train Set Mean Squared Error: {mse_train}")
r2_train = r2_score(y_train, y_output)
r2 = r2_score(y_test, y_pred)
# print(f"Test Set R2: {r2} Train Set R2: {r2_train}")


# print("DT")
model = DecisionTreeRegressor()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_output=model.predict(x_train)
mape = mean_absolute_percentage_error(y_test+1, y_pred)
mape_train = mean_absolute_percentage_error(y_train+1, y_output)
# print(f"Test Set Mean Absolute Percentage Error: {mape} Train Set Mean Absolute Percentage Error: {mape_train}")
mae_train = mean_absolute_error(y_train, y_output)
mae = mean_absolute_error(y_test, y_pred)
# print(f"Test Set Mean Absolute Error: {mae} Train Set Mean Absolute Error: {mae_train}")
mse_train = mean_squared_error(y_train, y_output)
mse = mean_squared_error(y_test, y_pred)
# print(f"Test Set Mean Squared Error: {mse} Train Set Mean Squared Error: {mse_train}")
r2_train = r2_score(y_train, y_output)
r2 = r2_score(y_test, y_pred)
# print(f"Test Set R2: {r2} Train Set R2: {r2_train}")
print("DT",mape,mape_train,mae,mae_train,mse,mse_train,r2,r2_train,sep=",")

# print("RF")
model = RandomForestRegressor()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_output=model.predict(x_train)
mape = mean_absolute_percentage_error(y_test+1, y_pred)
mape_train = mean_absolute_percentage_error(y_train+1, y_output)
# print(f"Test Set Mean Absolute Percentage Error: {mape} Train Set Mean Absolute Percentage Error: {mape_train}")
mae_train = mean_absolute_error(y_train, y_output)
mae = mean_absolute_error(y_test, y_pred)
# print(f"Test Set Mean Absolute Error: {mae} Train Set Mean Absolute Error: {mae_train}")
mse_train = mean_squared_error(y_train, y_output)
mse = mean_squared_error(y_test, y_pred)
# print(f"Test Set Mean Squared Error: {mse} Train Set Mean Squared Error: {mse_train}")
r2_train = r2_score(y_train, y_output)
r2 = r2_score(y_test, y_pred)
# print(f"Test Set R2: {r2} Train Set R2: {r2_train}")
print("RF",mape,mape_train,mae,mae_train,mse,mse_train,r2,r2_train,sep=",")

# print("XGB")
model = XGBRegressor()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_output=model.predict(x_train)
mape = mean_absolute_percentage_error(y_test+1, y_pred)
mape_train = mean_absolute_percentage_error(y_train+1, y_output)
# print(f"Test Set Mean Absolute Percentage Error: {mape} Train Set Mean Absolute Percentage Error: {mape_train}")
mae_train = mean_absolute_error(y_train, y_output)
mae = mean_absolute_error(y_test, y_pred)
# print(f"Test Set Mean Absolute Error: {mae} Train Set Mean Absolute Error: {mae_train}")
mse_train = mean_squared_error(y_train, y_output)
mse = mean_squared_error(y_test, y_pred)
# print(f"Test Set Mean Squared Error: {mse} Train Set Mean Squared Error: {mse_train}")
r2_train = r2_score(y_train, y_output)
r2 = r2_score(y_test, y_pred)
# print(f"Test Set R2: {r2} Train Set R2: {r2_train}")
print("XGB",mape,mape_train,mae,mae_train,mse,mse_train,r2,r2_train,sep=",")

# print("LGBM")
model = LGBMRegressor()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_output=model.predict(x_train)
mape = mean_absolute_percentage_error(y_test+1, y_pred)
mape_train = mean_absolute_percentage_error(y_train+1, y_output)
# print(f"Test Set Mean Absolute Percentage Error: {mape} Train Set Mean Absolute Percentage Error: {mape_train}")
mae_train = mean_absolute_error(y_train, y_output)
mae = mean_absolute_error(y_test, y_pred)
# print(f"Test Set Mean Absolute Error: {mae} Train Set Mean Absolute Error: {mae_train}")
mse_train = mean_squared_error(y_train, y_output)
mse = mean_squared_error(y_test, y_pred)
# print(f"Test Set Mean Squared Error: {mse} Train Set Mean Squared Error: {mse_train}")
r2_train = r2_score(y_train, y_output)
r2 = r2_score(y_test, y_pred)
# print(f"Test Set R2: {r2} Train Set R2: {r2_train}")
print("LGBM",mape,mape_train,mae,mae_train,mse,mse_train,r2,r2_train,sep=",")

#print("CatBoost")
model = CatBoostRegressor(verbose=0)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_output=model.predict(x_train)
mape = mean_absolute_percentage_error(y_test+1, y_pred)
mape_train = mean_absolute_percentage_error(y_train+1, y_output)
#print(f"Test Set Mean Absolute Percentage Error: {mape} Train Set Mean Absolute Percentage Error: {mape_train}")
mae_train = mean_absolute_error(y_train, y_output)
mae = mean_absolute_error(y_test, y_pred)
#print(f"Test Set Mean Absolute Error: {mae} Train Set Mean Absolute Error: {mae_train}")
mse_train = mean_squared_error(y_train, y_output)
mse = mean_squared_error(y_test, y_pred)
#print(f"Test Set Mean Squared Error: {mse} Train Set Mean Squared Error: {mse_train}")
r2_train = r2_score(y_train, y_output)
r2 = r2_score(y_test, y_pred)
#print(f"Test Set R2: {r2} Train Set R2: {r2_train}")
print("Catboost",mape,mape_train,mae,mae_train,mse,mse_train,r2,r2_train,sep=",")

#print("LCERegression")
model = LCERegressor(
    #base_learner="catboost"
)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_output=model.predict(x_train)
mape = mean_absolute_percentage_error(y_test+1, y_pred)
mape_train = mean_absolute_percentage_error(y_train+1, y_output)
#print(f"Test Set Mean Absolute Percentage Error: {mape} Train Set Mean Absolute Percentage Error: {mape_train}")
mae_train = mean_absolute_error(y_train, y_output)
mae = mean_absolute_error(y_test, y_pred)
#print(f"Test Set Mean Absolute Error: {mae} Train Set Mean Absolute Error: {mae_train}")
mse_train = mean_squared_error(y_train, y_output)
mse = mean_squared_error(y_test, y_pred)
#print(f"Test Set Mean Squared Error: {mse} Train Set Mean Squared Error: {mse_train}")
r2_train = r2_score(y_train, y_output)
r2 = r2_score(y_test, y_pred)
#print(f"Test Set R2: {r2} Train Set R2: {r2_train}")
print("LCE",mape,mape_train,mae,mae_train,mse,mse_train,r2,r2_train,sep=",")

#print("TabNetRegression")
model = TabNetRegressor(verbose=0)
model.fit(x_train,y_train.reshape(-1,1),batch_size=32)
y_pred=model.predict(x_test)
y_output=model.predict(x_train)
mape = mean_absolute_percentage_error(y_test+1, y_pred)
mape_train = mean_absolute_percentage_error(y_train+1, y_output)
#print(f"Test Set Mean Absolute Percentage Error: {mape} Train Set Mean Absolute Percentage Error: {mape_train}")
mae_train = mean_absolute_error(y_train, y_output)
mae = mean_absolute_error(y_test, y_pred)
#print(f"Test Set Mean Absolute Error: {mae} Train Set Mean Absolute Error: {mae_train}")
mse_train = mean_squared_error(y_train, y_output)
mse = mean_squared_error(y_test, y_pred)
#print(f"Test Set Mean Squared Error: {mse} Train Set Mean Squared Error: {mse_train}")
r2_train = r2_score(y_train, y_output)
r2 = r2_score(y_test, y_pred)
#print(f"Test Set R2: {r2} Train Set R2: {r2_train}")
print("TabNet",mape,mape_train,mae,mae_train,mse,mse_train,r2,r2_train,sep=",")

#print("MLPRegression")
model = MLPRegressor()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_output=model.predict(x_train)
mape = mean_absolute_percentage_error(y_test+1, y_pred)
mape_train = mean_absolute_percentage_error(y_train+1, y_output)
#print(f"Test Set Mean Absolute Percentage Error: {mape} Train Set Mean Absolute Percentage Error: {mape_train}")
mae_train = mean_absolute_error(y_train, y_output)
mae = mean_absolute_error(y_test, y_pred)
#print(f"Test Set Mean Absolute Error: {mae} Train Set Mean Absolute Error: {mae_train}")
mse_train = mean_squared_error(y_train, y_output)
mse = mean_squared_error(y_test, y_pred)
#print(f"Test Set Mean Squared Error: {mse} Train Set Mean Squared Error: {mse_train}")
r2_train = r2_score(y_train, y_output)
r2 = r2_score(y_test, y_pred)
#print(f"Test Set R2: {r2} Train Set R2: {r2_train}")
print("MLP",mape,mape_train,mae,mae_train,mse,mse_train,r2,r2_train,sep=",")

# param_grid_regression = {
# }
# # 初始化LR模型
# model = LinearRegression()
# grid_search_regression = GridSearchCV(model, param_grid_regression, cv=5)
# grid_search_regression.fit(X, y.reshape(-1,1))
# print("LR Best parameters for regression: ", grid_search_regression.best_params_, "Best score for regression:",grid_search_regression.best_score_)
#
# param_grid_regression = {
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # 核函数类型，默认为'rbf'
#     'degree': [2, 3, 4],  # 多项式核函数的度数，仅当kernel='poly'时有效，默认为3
#     'gamma': ['scale', 'auto'],  # 核系数，'scale'（默认）或'auto'
#     'coef0': [0.0, 0.1, 0.5],  # 核函数中的独立项，仅在kernel='poly'和'sigmoid'时有效，默认为0.0
#     'tol': [1e-3, 1e-4, 1e-5],  # 停止条件的公差，默认为1e-3
#     'C': [1, 10, 100],  # 正则化参数，默认为1.0
#     'epsilon': [0.1, 0.2, 0.3],  # 不计算损失的间隔大小，默认为0.1
#     'shrinking': [True, False],  # 是否使用缩小启发式，默认为True
#     'max_iter': [-1, 1000, 2000],  # 迭代的最大次数，-1表示无限制，默认为-1
# }
# # SVR
# model = SVR()
# grid_search_regression = GridSearchCV(model, param_grid_regression, cv=5)
# grid_search_regression.fit(X, y.reshape(-1,1))
# print("SVR Best parameters for regression: ", grid_search_regression.best_params_, "Best score for regression:",grid_search_regression.best_score_)

# param_grid_regression = {
#     'n_neighbors': [3, 5, 7, 11,13,15,17],  # 邻居的数量，默认为5
#     'weights': ['uniform', 'distance'],  # 邻居的权重函数，默认为'uniform'
#     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # 用于计算最近邻的算法，默认为'auto'
#     'leaf_size': [10, 20, 30, 40, 50],  # 叶节点的大小（仅适用于BallTree或KDTree算法），默认为30
#     'p': [1, 2],  # Minkowski距离的幂参数，默认为2
#     'metric': ['minkowski', 'euclidean', 'manhattan'],  # 距离度量，默认为'minkowski'
#     'n_jobs': [-1]  # 用于搜索邻居的并行作业数，默认为None（使用1个作业）
# }
# # 初始化KNN模型
# model = KNeighborsRegressor()
# grid_search_regression = GridSearchCV(model, param_grid_regression, cv=5)
# grid_search_regression.fit(X, y.reshape(-1,1))
# print("KNN Best parameters for regression: ", grid_search_regression.best_params_, "Best score for regression:",grid_search_regression.best_score_)

# param_grid_regression = {
#     'criterion': ['mse', 'absolute_error'],  # 分割标准，默认为'mse'
#     'splitter': ['best', 'random'],  # 分割策略，默认为'best'
#     #'max_depth': [None, 10, 20, 30, 40, 50],  # 树的最大深度，默认为None
#     'min_samples_split': [2, 3, 5],  # 分割内部节点所需的最小样本数，默认为2
#     'min_samples_leaf': [1, 2, 3],  # 在叶节点处需要的最小样本数，默认为1
#     'min_weight_fraction_leaf': [0.0, 0.1, 0.2],  # 叶节点处（所有输入样本的）权重总和的最小加权分数，默认为0.0
#     'max_features': ['auto', 'sqrt', 'log2'],  # 寻找最佳分割时考虑的特征数量，默认为None
#     #'max_leaf_nodes': [None, 30, 40],  # 以最优方法种植新叶节点的最大数量，默认为None
#     'min_impurity_decrease': [0.0, 0.1, 0.2],  # 如果节点分割导致不纯度减少大于或等于该值，则该节点将被分割，默认为0.0
#     'ccp_alpha': [0.0, 0.01, 0.1]  # 复杂性参数。用于Minimal Cost-Complexity Pruning，默认为0.0
# }
# # DT
# model = DecisionTreeRegressor()
# grid_search_regression = GridSearchCV(model, param_grid_regression, cv=5)
# grid_search_regression.fit(X, y.reshape(-1,1))
# print("DT Best parameters for regression: ", grid_search_regression.best_params_, "Best score for regression:",grid_search_regression.best_score_)



# param_grid_regression={
#     'n_estimators': [50, 100, 200, 500],  # 树的个数，即弱学习器的数量，默认为100
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],  # 学习速率，值范围[0,1]，默认为0.3
#     "grow_policy":[0,1],
#     'max_depth': [4, 5, 6, 8,10],  # 最大树深度，值范围[1,∞]，默认为6
#     'min_child_weight': [1, 2, 3, 5],  # 孩子节点中最小的样本权重和，值范围[0,∞]，默认为1
#     'gamma': [0, 0.1, 0.2, 0.3],  # 节点分裂所需的最小损失函数下降值，值范围[0,∞]，默认为0
#     'subsample': [0.6, 0.8, 1.0],  # 建立每棵树时对样本的随机采样的比例，值范围(0,1]，默认为1
#     'colsample_bytree': [0.6, 0.8, 1.0],  # 建立每棵树时对特征的随机采样的比例，值范围(0,1]，默认为1
#     'colsample_bylevel': [0.6, 0.8, 1.0],  # 树的每一级的每一次分裂，对列数的采样的占比，值范围(0,1]，默认为1
#     'colsample_bynode': [0.6, 0.8, 1.0],  # 每个节点（即每一次分裂）对列数的采样的占比，值范围(0,1]，默认为1
#     'reg_alpha': [0, 0.1, 0.5, 1],  # L1正则化项权重，值范围[0,∞]，默认为0
#     'reg_lambda': [0.5, 1, 1.5, 2],  # L2正则化项权重，值范围[0,∞]，默认为1
#     'scale_pos_weight': [0.8, 1, 1.2],  # 在高度不平衡的类别权重时，控制正负权重的平衡，默认为1
#     'booster': ['gbtree', 'gblinear', 'dart'],  # 选择每次迭代的模型，有三种选择gbtree、gblinear或dart，默认为'gbtree'
#     'eval_metric': ['rmse', 'mae']  # 验证数据的评价指标，默认为根据目标函数自动选择
# }
# model=XGBRegressor()
# grid_search_regression = GridSearchCV(model, param_grid_regression, cv=5)
# grid_search_regression.fit(X, y.reshape(-1,1))
# print("XGB Best parameters for regression: ", grid_search_regression.best_params_, "Best score for regression:",grid_search_regression.best_score_)
#
# param_grid_regression={
#     'boosting_type': ['gbdt', 'dart', 'goss', 'rf'],  # 提升类型，默认为'gbdt'
#     'num_leaves': [6,8,16,31],  # 一棵树上的叶子数，默认为31
#     'max_depth': [-1, 5, 10, 15],  # 树的最大深度，默认为-1，表示无限制
#     'learning_rate': [0.001, 0.01, 0.05, 0.1],  # 学习率，默认为0.1
#     'n_estimators': [50, 100, 200, 500],  # 树的个数，默认为100
#     #'subsample_for_bin': [200000, 500000],  # 用于构建直方图的样本数量，默认为200000
#     'class_weight': [None, 'balanced'],  # 类别权重，默认为None
#     'min_split_gain': [0.0, 0.1, 0.5],  # 执行切分的最小增益，默认为0
#     'min_child_weight': [1e-3, 1e-2, 1e-1, 1],  # 叶子节点的最小样本权重和，默认为1e-3
#     'min_child_samples': [5, 10, 20, 30],  # 叶子节点的最小样本数，默认为20
#     'subsample': [0.7, 0.8, 0.9, 1.0],  # 子样本比率，默认为1.0
#     'subsample_freq': [0, 1, 5, 10],  # 子样本频率，默认为0
#     'colsample_bytree': [0.6, 0.8, 1.0],  # 每棵树的列采样比率，默认为1.0
#     'reg_alpha': [0, 1, 5, 10],  # L1正则化系数，默认为0
#     'reg_lambda': [0, 1, 5, 10],  # L2正则化系数，默认为0
#     'importance_type': ['split', 'gain']  # 特征重要度类型，默认为'split'
# }
# model=LGBMRegressor()
# grid_search_regression = GridSearchCV(model, param_grid_regression, cv=5)
# grid_search_regression.fit(X, y.reshape(-1,1))
# print("LGBM Best parameters for regression: ", grid_search_regression.best_params_, "Best score for regression:",grid_search_regression.best_score_)
#
# param_grid_regression={
#     #'iterations': [100, 500, 1000],  # Boosting迭代次数，默认1000
#     'learning_rate': [0.01, 0.05, 0.03],  # 学习率，默认0.03
#     'depth': [6, 4,2],  # 树的最大深度，默认6
#     'l2_leaf_reg': [3, 5, 7, 9],  # 叶子的L2正则化系数，默认3
#     'model_size_reg': [0.5, 0.75, 1.0],  # 模型大小正则化系数，默认10
#     'rsm': [0.8, 0.9, 1.0],  # 随机选择的列的比例，默认1
#     'loss_function': ['RMSE', 'MAE', 'Poisson'],  # 损失函数，默认'RMSE'
#     'border_count': [32, 128],  # 分箱数，默认128
#     #'ctr_border_count': [50, 100, 200],  # CTR分箱数，默认50
#     'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS'],  # 自助采样类型，默认'Bayesian'
#     'subsample': [0.6, 0.8, 1.0],  # 子样本比例，仅当bootstrap_type为'Bernoulli'或'MVS'时有效，默认取决于bootstrap_type
#     #'sampling_frequency': ['PerTree', 'PerTreeLevel'],  # 采样频率，默认'PerTree'
#     'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],  # 树增长策略，默认'SymmetricTree'
#     'min_data_in_leaf': [1, 2, 3],  # 叶子节点最小样本数，默认1
#     'max_leaves': [31, 16,8],  # Lossguide生长策略下树的最大叶子数，默认31
#     'score_function': ['Cosine', 'L2'],  # 分裂评分函数，默认'Cosine'
#     'leaf_estimation_iterations': [1, 5, 10],  # 叶子估算的迭代次数，默认1
#     'leaf_estimation_backtracking': ['No', 'AnyImprovement'],  # 叶子估算的回溯类型，默认AnyImprovement
#     'thread_count': [4, 8, 12],  # 用于模型训练的线程数，默认取决于可用CPU核数
#     'random_seed': [42, 56, 72],  # 随机数种子，默认为0
#     'verbose': [False],  # 日志详细程度，默认True
#     'task_type': ['GPU'],  # 计算任务类型，默认'CPU'
#     'max_ctr_complexity': [1, 2, 3, 4],  # CTR特征的最大复杂性，默认4
#     'simple_ctr': ['Buckets', 'Borders'],  # 用于生成简单CTR的分箱方式，默认'Borders'
#     #'combinations_ctr': ['Buckets', 'Borders'],  # 用于生成组合CTR的分箱方式，默认'Borders'
#     'per_feature_ctr': [None],  # 特定特征的CTR描述，默认None
#     #'ctr_leaf_count_limit': [None, 10, 100],  # 用于计算CTR的叶子数限制，默认None
#     'max_depth': [4, 6, 8],  # 树的最大深度，默认0
#     'has_time': [False],  # 是否使用时间信息，默认False
#     'fold_len_multiplier': [1.5, 2.0],  # 折叠大小的倍乘因子，默认2
#     'fold_permutation_block': [1, 2]  # 折叠排列块大小，默认1
#
# }
# model=CatBoostRegressor()
# grid_search_regression = GridSearchCV(model, param_grid_regression, cv=5)
# grid_search_regression.fit(X, y.reshape(-1,1))
# print("Catboost Best parameters for regression: ", grid_search_regression.best_params_, "Best score for regression:",grid_search_regression.best_score_)
#
# param_grid_regression = {
#     #"n_estimators":[10,20,50],
#     "bootstrap":[True,False],
#     "splitter":['best',"random"],
#     "max_depth":[2,4,6],
#     "max_samples":[1.0],
#     "min_samples_leaf":[1],
#     "n_iter":[100],
#     "base_learner":['xgboost',"catboost", "lightgbm"],
#     "base_n_estimators":[10, 50, 100],
#     "base_max_depth":[3, 6, 9],
#     #"base_num_leaves":[20, 50, 100, 500],
#     "base_learning_rate":[0.01, 0.1, 0.3, 0.5],
#     "base_booster":['gbtree', 'gblinear', 'dart'],
#     "base_boosting_type":['gbdt', 'dart', 'rf'],
#     "base_gamma":[0, 1, 10],
#     "base_min_child_weight":[1, 5, 15, 100],
#     "base_subsample":[1.0,],
#     "base_subsample_for_bin":[200000,],
#     "base_colsample_bytree":[1.0,],
#     "base_colsample_bylevel":[1.0,],
#     "base_colsample_bynode":(1.0,),
#     "base_reg_alpha":[0,],
#     "base_reg_lambda":[0.1, 1.0, 5.0]
# }
# model = LCERegressor()
# grid_search_regression = GridSearchCV(model, param_grid_regression, cv=5)
# grid_search_regression.fit(X, y.reshape(-1,1))
# print("LCE Best parameters for regression: ", grid_search_regression.best_params_, "Best score for regression:",grid_search_regression.best_score_)
#
#
# param_grid_regression = {
#     'kernel': [  # 核函数，默认为1.0 * RBF(1.0)
#         1.0 * RBF(1.0),
#         1.0 * RBF(0.5),
#         1.0 * Matern(length_scale=1.0, nu=1.5),
#         1.0 * Matern(length_scale=1.0, nu=0.5),
#         1.0 * WhiteKernel(noise_level=1),
#         1.0 * WhiteKernel(noise_level=0.1)
#     ],
#     'alpha': [1e-10, 1e-8, 1e-6, 1e-4],  # 增加到核矩阵对角线的值，以提高数值稳定性，默认为1e-10
#     'n_restarts_optimizer': [0, 1, 2, 3],  # 优化器重新启动的次数，默认为0
#     'normalize_y': [False, True],  # 是否标准化目标值y，使其均值为零，默认为False
#
# }
# model = GaussianProcessRegressor()
# grid_search_regression = GridSearchCV(model, param_grid_regression, cv=5)
# grid_search_regression.fit(X, y.reshape(-1,1))
# print("GPR Best parameters for regression: ", grid_search_regression.best_params_, "Best score for regression:",grid_search_regression.best_score_)
#
# param_grid_regression = {
#     'n_estimators': [32,64, 100],  # 决策树的数量，默认为100
#     #'max_depth': [None, 20, 30, 40],  # 树的最大深度，默认为None（节点会扩展直到包含少于min_samples_split样本）
#     'min_samples_split': [2, 5, ],  # 分裂内部节点所需的最小样本数，默认为2
#     'min_samples_leaf': [1, 2,3],  # 在叶节点处需要的最小样本数，默认为1
#     'min_weight_fraction_leaf': [0.0, 0.1, 0.2],  # 在所有叶节点处（所有输入样本）的权重总和中的最小加权分数，默认为0.0
#     'max_features': ['auto', 'sqrt', 'log2'],  # 寻找最佳分割时要考虑的特征数量，默认为'auto'
#     #'max_leaf_nodes': [None, 20, 30],  # 以最佳优先方式种植树时使用的最大叶节点数，默认为None
#     'min_impurity_decrease': [0.0, 0.1, 0.2],  # 如果节点分裂会导致杂质的减少大于或等于该值，则该节点将被分裂，默认为0.0
#     #'bootstrap': [True, False],  # 是否在构建树时使用样本的有放回抽样，默认为True
#     #'oob_score': [True, False],  # 是否使用袋外样本来估计泛化精度，默认为False
#     'n_jobs': [-1],  # 用于拟合和预测的作业数，默认为None（使用1个作业）
#     #'warm_start': [False, True],  # 设置为True时，请重用上一个调用的解决方案以适应并添加更多的估计量，否则，只需适应一个全新的森林，默认为False
#     'ccp_alpha': [0.0, 0.1, 0.2],  # 用于剪枝的复杂性参数，默认为0.0
#     'max_samples': [None, 0.8]  # 从X中抽取以训练每个基础估计量的样本数，默认为None（使用所有样本）
#
# }
#
# model = RandomForestRegressor()
# grid_search_regression = GridSearchCV(model, param_grid_regression, cv=5)
# grid_search_regression.fit(X, y.reshape(-1,1))
# print("RF Best parameters for regression: ", grid_search_regression.best_params_, "Best score for regression:",grid_search_regression.best_score_)
#
#
#
#
# param_grid_regression = {
#     'hidden_layer_sizes': [(50,), (50, 50), (100,)],  # 隐藏层神经元数量，默认为(100,)
#     'activation': ['tanh', 'relu'],  # 激活函数，默认为'relu'
#     'alpha': [0.0001, 0.001, 0.01],  # L2正则化参数，默认为0.0001
#     'batch_size': ['auto', 64, 128],  # 批处理大小，默认为'auto'（即 min(200, n_samples)）
#     'learning_rate': ['constant', 'invscaling', 'adaptive'],  # 学习率设置，默认为'constant'
#     'learning_rate_init': [0.001, 0.01],  # 初始学习率，默认为0.001
#     'power_t': [0.5, 0.6],  # “invscaling”和“adaptive”学习率调度器的指数衰减率，默认为0.5
#     'max_iter': [200, 300],  # 最大迭代次数，默认为200
#     'tol': [1e-4, 1e-5],  # 优化公差，默认为1e-4
#     'validation_fraction': [0.1, 0.15],  # 保留用于早期停止的验证集的比例，默认为0.1
#     # 'beta_1': [0.9, 0.8],  # 用于估计一阶矩向量的指数衰减率（仅当solver='adam'时适用），默认为0.9
#     # 'beta_2': [0.999, 0.99],  # 用于估计二阶矩向量的指数衰减率（仅当solver='adam'时适用），默认为0.999
#     # 'epsilon': [1e-8, 1e-9]  # 数值稳定性（仅当solver='adam'时适用），默认为1e-8
#
# }
# model = MLPRegressor()
# grid_search_regression = GridSearchCV(model, param_grid_regression, cv=5)
# grid_search_regression.fit(X, y.reshape(-1,1))
# print("MLP Best parameters for regression: ", grid_search_regression.best_params_, "Best score for regression:",grid_search_regression.best_score_)
#
#
# import torch
# ###batchsize也要调整
# param_grid_regression = {
#     'n_d': [8, 16,],  # 主要决策路径的维度，默认为8
#     'n_a': [8, 16,],  # 主要决策路径的维度，默认为8
#     'n_steps': [3, 5, 7],  # 决策步数，默认为3
#     'gamma': [1.0, 1.3, 1.5],  # 压缩中间特征维度的强度，默认为1.3
#     'n_independent': [1, 2],  #
#     'n_shared': [1, 2],  # 每个步骤中独立变换器的数量，默认为1
#     'lambda_sparse': [1e-4, 1e-3, 1e-2],  # 用于模型稀疏性的正则化，默认为1e-3
#     'momentum': [0.02, 0.3, 0.6],  # 批标准化的动量，默认为0.02
#     'clip_value': [None, 1.0, 2.0],  # 梯度剪裁值，默认为None
#     'optimizer_fn': [torch.optim.Adam, torch.optim.AdamW],  # 优化器，默认为torch.optim.Adam
#     'optimizer_params': [
#         dict(lr=2e-2),
#         dict(lr=1e-2)
#     ],  # 优化器参数，默认为dict(lr=2e-2)
#     'scheduler_fn': [None, torch.optim.lr_scheduler.ReduceLROnPlateau],  # 学习率调度器，默认为None
# }
# model = TabNetRegressor()
# grid_search_regression = GridSearchCV(model, param_grid_regression, cv=5)
# grid_search_regression.fit(X, y.reshape(-1,1),batch_size=48)
# print("TabNet Best parameters for regression: ", grid_search_regression.best_params_, "Best score for regression:",grid_search_regression.best_score_)