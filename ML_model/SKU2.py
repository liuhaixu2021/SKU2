import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error,median_absolute_error,r2_score
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR,SVC,LinearSVC,LinearSVR
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import precision_recall_curve, average_precision_score,roc_curve, auc, precision_score, recall_score,mean_absolute_percentage_error, f1_score, confusion_matrix, accuracy_score

###读入数据
path = "SKU1.csv"
data = pd.read_csv(path)
###要求序列已经被填补
def clean_data(data):
    print(data.columns.values)
    data.drop(["sku",'comp_5_price'])
    ###求比例
    data['comp_1_price']= data['price']/ data['price']
    data['comp_1_price'] = data['price'] / data['price']
    data['comp_1_price'] = data['price'] / data['price']
    data['comp_1_price'] = data['price'] / data['price']
    ###做时序特征
    data['salesdate'] = pd.to_datetime(data['salesdate'])
    data['weekday'] = data['salesdate'].dt.weekday
    data['is_weekend'] = np.where(data['weekday'] > 4, 1, 0)
    ###
    return data
data=clean_data(data)