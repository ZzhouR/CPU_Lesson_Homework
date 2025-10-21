import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

classifiers = [
    {
        'name':'SVM',
        'parameters':{
            'kernel':['linear','poly','rbf'],   # 核函数类型列表：线性、多项式、高斯径向基核
            'C':np.arange(30,60,5)  # 惩罚参数C的取值范围，从30到55，步长5
        },
        'cached': SVC(kernel ='rbf', C=50),   # 已经初始化的SVM模型，用于快速调用，默认核函数为rbf，C=50
        'method':SVC
    },
    {
        'name':'Random Forest' ,
        'parameters':{
            'max_depth':[1,2,3,4,5],    # 树的最大深度选项
            'n_estimators':[10,20,30,40,50,60,70,80,90,100,110,120, 300, 400, 600],   # 树的数量选项
            'max_features':[1,2,3]   # 每棵树考虑的最大特征数量选项
        },
        'cached': RandomForestClassifier(max_depth=6, n_estimators=800),    # 已初始化的随机森林模型，深度6，树数量800
        'method':RandomForestClassifier
    },
    {
        'name':'XGboost' ,
        'parameters':{
            'max_depth':[1,2,3,4,5],    # 最大深度范围
            'n_estimators':[10,20,30,40,50,60,70,80,90,100,110,120, 300, 400, 600],   # 弱分类器个数范围
            'max_features':[1,2,3]   # 每次分裂考虑的最大特征数
        },
        'cached': XGBClassifier(max_depth=6, n_estimators=9),   # 预定义的XGBoost模型，深度6，弱分类器数量9
        'method':XGBClassifier
    },
]