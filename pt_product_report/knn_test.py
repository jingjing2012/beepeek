import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris

# 加载示例数据集
data = load_iris()
X = data.data
y = data.target

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义模型和参数网格
models = {
    'svc': SVC(random_state=1),
    'decisiontreeclassifier': DecisionTreeClassifier(random_state=1),
    'randomforestclassifier': RandomForestClassifier(random_state=1),
    'kneighborsclassifier': KNeighborsClassifier()
}

param_grids = {
    'svc': {'svc__C': [0.1, 1, 10, 100], 'svc__gamma': [0.001, 0.01, 0.1, 1]},
    'decisiontreeclassifier': {'decisiontreeclassifier__max_depth': [3, 5, 10, 15]},
    'randomforestclassifier': {'randomforestclassifier__n_estimators': [10, 50, 100, 200], 'randomforestclassifier__max_depth': [3, 5, 10, 15]},
    'kneighborsclassifier': {'kneighborsclassifier__n_neighbors': [3, 5, 7, 9]}
}

# 定义函数进行网格搜索和模型评估
def GridSearchCV_working(model_name, model, param_grids, train_x, train_y, test_x, test_y, score='accuracy'):
    response = {}
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        (model_name, model)
    ])

    param_grid = param_grids[model_name]

    gridsearch = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=score, cv=KFold(n_splits=5, shuffle=True, random_state=42))
    search = gridsearch.fit(train_x, train_y)

    response['best_params'] = search.best_params_
    response['best_score'] = search.best_score_
    response['best_estimator'] = search.best_estimator_
    predict_y = search.predict(test_x)
    response['accuracy_score'] = accuracy_score(test_y, predict_y)
    response['classification_report'] = classification_report(test_y, predict_y)
    response['confusion_matrix'] = confusion_matrix(test_y, predict_y)

    print(f"Model: {model_name}")
    print("GridSearch最优参数：", response['best_params'])
    print("GridSearch最优分数： %0.4lf" % response['best_score'])
    print("准确率 %0.4lf" % response['accuracy_score'])
    print("分类报告:\n", response['classification_report'])
    print("混淆矩阵:\n", response['confusion_matrix'])
    print("\n")

    return response

# 迭代不同的模型和参数网格进行搜索和评估
results = {}
for model_name, model in models.items():
    results[model_name] = GridSearchCV_working(model_name, model, param_grids, X_train, y_train, X_test, y_test)
