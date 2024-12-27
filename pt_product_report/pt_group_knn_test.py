import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from conn import sql_engine, mysql_config as config
import pt_product_sql as sql

# 数据获取
df_group = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password, config.product_database,
                                         sql.sampling_knn_sql)

# 数据探索
pd.set_option('display.max_columns', None)

# 数据清洗
df_group = df_group.drop(df_group.columns[0:2], axis=1)
df_group = df_group.drop(df_group.columns[43:62], axis=1)
# df_group = df_group.drop('ratings_revenue_pass', axis=1)
df_group = df_group.drop('冒出品低星款数', axis=1)
print(df_group.shape)
print(df_group.columns)

# 替换无穷大值，使用最大/最小有限值替换
df_group = df_group.replace([np.inf, -np.inf], np.nan)

# 填充 NaN 值，使用均值填充
df_group = df_group.fillna(df_group.min())

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换为系统中可用的中文字体
plt.rcParams['axes.unicode_minus'] = False

"""
# 数据可视化
features = list(df_group[1:44])
corr = df_group[features].corr()
plt.figure(figsize=(50, 50))
# sns.heatmap(corr, annot=True, fmt='.2f', linecolor='gray', cmap='coolwarm')
plt.show()
"""

# 分割数据
x = df_group.drop('clue_tag', axis=1)
y = df_group['clue_tag']

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=42)

# 采用Z-Score规范化
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)

# 创建KNN分类器
knn = KNeighborsClassifier()
knn.fit(train_ss_x, train_y)
predict_y = knn.predict(test_ss_x)
print("KNN准确率: %.4lf" % accuracy_score(test_y, predict_y))
print("KNN精确率: %.4lf" % precision_score(test_y, predict_y, zero_division=0))
print("KNN召回率: %.4lf" % recall_score(test_y, predict_y))
print("KNNF1得分: %.4lf" % f1_score(test_y, predict_y))
print("KNNROC曲线和AUC: %.4lf" % roc_auc_score(test_y, predict_y))
print("----")

# 创建SVM分类器
svm = SVC()
svm.fit(train_ss_x, train_y)
predict_y = svm.predict(test_ss_x)
print('SVM准确率: %0.4lf' % accuracy_score(test_y, predict_y))
print("SVM精确率: %.4lf" % precision_score(test_y, predict_y, zero_division=0))
print("SVM召回率: %.4lf" % recall_score(test_y, predict_y))
print("SVMF1得分: %.4lf" % f1_score(test_y, predict_y))
print("SVMROC曲线和AUC: %.4lf" % roc_auc_score(test_y, predict_y))
print("----")

# 采用Min-Max规范化
mm = preprocessing.MinMaxScaler()
train_mm_x = mm.fit_transform(train_x)
test_mm_x = mm.transform(test_x)

# 创建Naive Bayes分类器
mnb = MultinomialNB()
mnb.fit(train_mm_x, train_y)
predict_y = mnb.predict(test_mm_x)
print("多项式朴素贝叶斯准确率: %.4lf" % accuracy_score(test_y, predict_y))
print("多项式朴素贝叶斯精确率: %.4lf" % precision_score(test_y, predict_y, zero_division=0))
print("多项式朴素贝叶斯召回率: %.4lf" % recall_score(test_y, predict_y))
print("多项式朴素贝叶斯F1得分: %.4lf" % f1_score(test_y, predict_y))
print("多项式朴素贝叶斯ROC曲线和AUC: %.4lf" % roc_auc_score(test_y, predict_y))
print("----")

# 创建CART决策树分类器
dtc = DecisionTreeClassifier()
dtc.fit(train_mm_x, train_y)
predict_y = dtc.predict(test_mm_x)
print("CART决策树准确率: %.4lf" % accuracy_score(test_y, predict_y))
print("CART决策树精确率: %.4lf" % precision_score(test_y, predict_y, zero_division=0))
print("CART决策树召回率: %.4lf" % recall_score(test_y, predict_y))
print("CART决策树F1得分: %.4lf" % f1_score(test_y, predict_y))
print("CART决策树ROC曲线和AUC: %.4lf" % roc_auc_score(test_y, predict_y))
print("----")

# 创建随机森林分类器
rfc = RandomForestClassifier()
rfc.fit(train_mm_x, train_y)
pridect_y = rfc.predict(test_mm_x)
print("随机森林准确率：%.4lf" % accuracy_score(test_y, predict_y))
print("随机森林精确率: %.4lf" % precision_score(test_y, predict_y, zero_division=0))
print("随机森林召回率: %.4lf" % recall_score(test_y, predict_y))
print("随机森林F1得分: %.4lf" % f1_score(test_y, predict_y))
print("随机森林ROC曲线和AUC: %.4lf" % roc_auc_score(test_y, predict_y))
print("----")

"""
# 生成混淆矩阵
conf_matrix = confusion_matrix(test_y, predict_y)
# 可视化混淆矩阵
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('conf_matrix')
plt.show()

# 生成分类报告
class_report = classification_report(test_y, predict_y, target_names=['clue_tag_0', 'clue_tag_1'])
print('Classification Report:')
print(class_report)
"""

print("--------------------------自动调参 V1--------------------------------")


def GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, param_grid, score):
    reponse = {}
    gridsearch = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=score)
    search = gridsearch.fit(train_x, train_y)
    print('GridSearchCV最优参数：', search.best_params_)
    print('GridSeatchCV最优分数： %.4lf' % search.best_score_)
    predict_y = search.predict(test_x)
    # print(predict_y)
    # print('\n')
    print('准确率 %.4lf' % accuracy_score(test_y, predict_y))
    print("精确率: %.4lf" % precision_score(test_y, predict_y, zero_division=0))
    print("召回率: %.4lf" % recall_score(test_y, predict_y))
    print("F1得分: %.4lf" % f1_score(test_y, predict_y))
    print("ROC曲线和AUC: %.4lf" % roc_auc_score(test_y, predict_y))
    # print('分类报告:\n', classification_report(test_y, predict_y))
    print('混淆矩阵:\n', confusion_matrix(test_y, predict_y))
    print('\n')
    # reponse['predict_y'] = predict_y
    # reponse['accuracy_score'] = accuracy_score(test_y, predict_y)
    return reponse


# 构造各种分类器
classifiers = [
    SVC(random_state=42, kernel='rbf', class_weight='balanced'),
    DecisionTreeClassifier(random_state=42, criterion='gini'),
    RandomForestClassifier(random_state=42, criterion='gini'),
    KNeighborsClassifier(metric='minkowski')
]

# 分类器名称
classifier_names = [
    'svc',
    'decisiontreeclassifier',
    'randomforestclassifier',
    'kneighborsclassifier'
]

# 分类器参数
classifier_param_grid = [
    {'svc__C': [0.1, 1, 10, 100], 'svc__gamma': [0.001, 0.01, 0.01, 1]},
    {'decisiontreeclassifier__max_depth': [3, 5, 10, 15]},
    {'randomforestclassifier__n_estimators': [10, 50, 100, 200]},
    {'kneighborsclassifier__n_neighbors': [3, 5, 7, 9]}
]

for model, model_name, model_param_prid in zip(classifiers, classifier_names, classifier_param_grid):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        (model_name, model)
    ])
    print(f'Model: {model_name}')
    result = GridSearchCV_work(pipeline, train_mm_x, train_y, test_mm_x, test_y, model_param_prid, 'f1')

print("--------------------------自动调参 V2--------------------------------")


def GridSearchCV_working(model_name, model, param_grid, train_x, train_y, test_x, test_y, score):
    reponse = {}
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        (model_name, model)
    ])
    gridsearch = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=score,
                              cv=StratifiedKFold(n_splits=10), n_jobs=-1)
    search = gridsearch.fit(train_x, train_y)
    reponse['best_params'] = search.best_params_
    reponse['best_score'] = search.best_score_
    reponse['best_estimators'] = search.best_estimator_
    predict_y = search.predict(test_x)
    # print(predict_y)
    # print('\n')
    reponse['accuracy_score'] = accuracy_score(test_y, predict_y)
    reponse['precision_score'] = precision_score(test_y, predict_y, zero_division=0)
    reponse['recall_score'] = recall_score(test_y, predict_y)
    reponse['f1_score'] = f1_score(test_y, predict_y)
    reponse['roc_auc_score'] = roc_auc_score(test_y, predict_y)
    # reponse['log_loss'] = log_loss(test_y, predict_y)
    # reponse['classifier_report'] = classification_report(test_y, predict_y)
    reponse['confusion_matrix'] = confusion_matrix(test_y, predict_y)

    print(f'Model: {model_name}')
    print('GridSearch最优参数：', reponse['best_params'])
    print('GridSearch最优分数：%.4lf' % reponse['best_score'])
    print('准确率 %.4lf' % reponse['accuracy_score'])
    print("精确率: %.4lf" % reponse['precision_score'])
    print("召回率: %.4lf" % reponse['recall_score'])
    print("F1得分: %.4lf" % reponse['f1_score'])
    print("ROC曲线和AUC: %.4lf" % reponse['roc_auc_score'])
    # print("log_loss: %.4lf" % reponse['log_loss'])
    # print('分类报告:\n', reponse['classifier_report'])
    print('混淆矩阵:\n', reponse['confusion_matrix'])
    print('\n')

    return reponse


# 定义模型和参数网格
models = {
    'svc': SVC(random_state=42, kernel='rbf', class_weight='balanced'),
    # 'svc': SVC(random_state=42, kernel='rbf', probability=True),
    'decisiontreeclassifier': DecisionTreeClassifier(random_state=42, criterion='gini'),
    'randomforestclassifier': RandomForestClassifier(random_state=42, criterion='gini'),
    'kneighborsclassifier': KNeighborsClassifier(metric='minkowski'),
    # 'gradientboostingclassifier': GradientBoostingClassifier(random_state=42)
}

param_grids = {
    'svc': {'svc__C': [0.01, 0.1, 1, 10, 100, 1000],  # 正则化参数：控制过拟合与欠拟合的平衡
            'svc__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],  # 核函数系数：定义 RBF、poly 等核函数的影响范围
            'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid']},  # 核函数类型：选择不同的核方法
    'decisiontreeclassifier': {'decisiontreeclassifier__max_depth': [None, 3, 5, 10, 20, 50],  # 树的最大深度：防止树过度生长
                               'decisiontreeclassifier__min_samples_split': [2, 5, 10, 20],  # 节点分裂所需的最小样本数
                               'decisiontreeclassifier__min_samples_leaf': [1, 2, 5, 10],  # 叶子节点最小样本数
                               'decisiontreeclassifier__criterion': ['gini', 'entropy', 'log_loss']},  # 分裂标准
    'randomforestclassifier': {'randomforestclassifier__n_estimators': [10, 50, 100, 200, 500],  # 森林中的树木数量：影响模型性能与训练时间
                               'randomforestclassifier__max_depth': [None, 5, 10, 20, 50],  # 树的最大深度：控制单棵树的复杂度
                               'randomforestclassifier__min_samples_split': [2, 5, 10],  # 节点分裂所需的最小样本数
                               'randomforestclassifier__min_samples_leaf': [1, 2, 5],  # 叶子节点最小样本数
                               'randomforestclassifier__max_features': ['sqrt', 'log2', None, 0.5],  # 每次分裂时的特征数量
                               'randomforestclassifier__bootstrap': [True, False]},  # 是否使用自助采样法
    'kneighborsclassifier': {'kneighborsclassifier__n_neighbors': [3, 5, 7, 9, 15, 20],  # 邻居数：控制预测的平滑性
                             'kneighborsclassifier__weights': ['uniform', 'distance'],  # 权重函数：定义邻居的权重
                             'kneighborsclassifier__metric': ['euclidean', 'manhattan', 'minkowski'],  # 距离度量方式
                             'kneighborsclassifier__p': [1, 2]},  # 距离的幂参数
    # 'gradientboostingclassifier': {'gradientboostingclassifier__n_estimators': [50, 100, 200],
    #                                'gradientboostingclassifier__learning_rate': [0.01, 0.1, 0.2],
    #                                'gradientboostingclassifier__max_depth': [3, 5, 7],
    #                                'gradientboostingclassifier__subsample': [0.8, 0.9, 1],
    #                                'gradientboostingclassifier__min_samples_split': [2, 5, 10],
    #                                'gradientboostingclassifier__min_samples_leaf': [1, 2, 4]}
}

# 迭代不同的模型和参数网格进行搜索和评估

print("--------------------------自动调参 V2_accuracy--------------------------------")
result = {}
for model_name, model in models.items():
    param_grid = param_grids[model_name]
    result[model_name] = GridSearchCV_working(model_name, model, param_grid, train_x, train_y, test_x, test_y,
                                              'accuracy')

print("--------------------------自动调参 V2_recall--------------------------------")
result = {}
for model_name, model in models.items():
    param_grid = param_grids[model_name]
    result[model_name] = GridSearchCV_working(model_name, model, param_grid, train_x, train_y, test_x, test_y, 'recall')

print("--------------------------自动调参 V2_F1--------------------------------")
result = {}
for model_name, model in models.items():
    param_grid = param_grids[model_name]
    result[model_name] = GridSearchCV_working(model_name, model, param_grid, train_x, train_y, test_x, test_y, 'f1')

print("--------------------------自动调参 V2_roc_auc--------------------------------")
result = {}
for model_name, model in models.items():
    param_grid = param_grids[model_name]
    result[model_name] = GridSearchCV_working(model_name, model, param_grid, train_x, train_y, test_x, test_y,
                                              'roc_auc')

"""
print("--------------------------自动调参 V2_precision--------------------------------")
result = {}
for model_name, model in models.items():
    param_grid = param_grids[model_name]
    result[model_name] = GridSearchCV_working(model_name, model, param_grid, train_x, train_y, test_x, test_y,
                                              'precision')
                                              
print("--------------------------自动调参 V2_log_loss--------------------------------")
result = {}
# 定义log_loss评分标准
scorer = make_scorer(log_loss, needs_proba=True, greater_is_better=False)
for model_name, model in models.items():
    param_grid = param_grids[model_name]
    result[model_name] = GridSearchCV_working(model_name, model, param_grid, train_x, train_y, test_x, test_y,
                                              scorer)
"""
