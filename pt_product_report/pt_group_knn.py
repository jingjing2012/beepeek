import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from conn import sql_engine, mysql_config as config
import pt_product_sql as sql
import pt_product_report_path as path
import pt_product_report_parameter as para


# 数据准备
def model_prepare(df):
    # 数据清洗
    df = df.drop(df.columns[0:2], axis=1)
    df = df.drop(df.columns[43:62], axis=1)
    df_train = df.drop('冒出品低星款数', axis=1)
    # df_train = df[para.sampling_list]
    # print(df_train.shape)
    # print(df_train.columns)
    # 替换无穷大值，使用最大/最小有限值替换
    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    # 填充 NaN 值，使用最小值填充
    df_train = df_train.fillna(df_train.min())
    return df_train


# 机器学习训练最优参数
def model_parameter(max_depth_decisiontree, max_depth_randomforest, n_estimators_randomforest):
    models = {
        'svc': SVC(random_state=42, kernel='rbf', class_weight='balanced', C=1, gamma=1),
        'decisiontreeclassifier': DecisionTreeClassifier(random_state=42, criterion='gini',
                                                         max_depth=max_depth_decisiontree,
                                                         min_samples_split=5),
        'randomforestclassifier': RandomForestClassifier(random_state=42, criterion='gini',
                                                         max_depth=max_depth_randomforest,
                                                         n_estimators=n_estimators_randomforest)
    }
    return models


"""
# 机器学习训练最优参数
def model_parameter():
    models = {
        'svc': SVC(random_state=42, kernel='rbf', class_weight='balanced', C=1, gamma=1),
        'decisiontreeclassifier': DecisionTreeClassifier(random_state=42, criterion='gini', max_depth=15,
                                                         min_samples_split=5),
        'randomforestclassifier': RandomForestClassifier(random_state=42, criterion='gini', max_depth=15,
                                                         n_estimators=200),
        # 'kneighborsclassifier': KNeighborsClassifier(n_neighbors=9)
    }
    return models
"""


# 模型训练
def model_training(df, models):
    df_train = model_prepare(df)
    # 分割数据
    x = df_train.drop('clue_tag', axis=1)
    y = df_train['clue_tag']
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=42)

    # 保存训练时的列名
    joblib.dump(train_x.columns, 'feature_columns.pkl')

    for model_name, model in models.items():
        # 规范化
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            (model_name, model)
        ])
        pipeline.fit(train_x, train_y)
        joblib.dump(pipeline, f'{model_name}_pipeline.pkl')


# 模型预测
def model_predict(df, model_name):
    test_x = model_prepare(df)

    # 加载训练时的列名并对齐
    feature_columns = joblib.load('feature_columns.pkl')
    test_x = test_x[feature_columns]  # 确保测试数据和训练数据列名一致

    pipeline = joblib.load(f'{model_name}_pipeline.pkl')
    predict_y = pipeline.predict(test_x)
    return predict_y


# -------------------------------------数据预测-----------------------------------------------
# 预测数据准备
df_group = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password, config.product_database,
                                         sql.sampling_group_sql)

# -------------------------------------精铺预测-----------------------------------------------
# 机器学习训练数据准备
df_group_knn_fba = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password, config.product_database,
                                                 sql.sampling_knn_fba_sql)
models = model_parameter(15, 15, 200)
model_training(df_group_knn_fba, models)

# 机器学习预测
for model_name in models.keys():
    # df_group.insert(loc=0, column='idx', value=df_group.index)
    df_group[f'{model_name}_predict_fba'] = model_predict(df_group, model_name)

# -------------------------------------直发预测-----------------------------------------------
# 机器学习训练数据准备
df_group_knn_fbm = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password, config.product_database,
                                                 sql.sampling_knn_fbm_sql)
models = model_parameter(3, 2, 70)
model_training(df_group_knn_fbm, models)

# 机器学习预测
for model_name in models.keys():
    # df_group.insert(loc=0, column='idx', value=df_group.index)
    df_group[f'{model_name}_predict_fbm'] = model_predict(df_group, model_name)

# -------------------------------------数据整合入库-----------------------------------------------
df_group['randomforestclassifier_predict'] = np.where(df_group['randomforestclassifier_predict_fba'] == 1, 2,
                                                      df_group['randomforestclassifier_predict_fbm'])

df_predict = df_group[
    ['原ASIN', '数据更新时间', 'svc_predict_fba', 'decisiontreeclassifier_predict_fba', 'randomforestclassifier_predict_fba',
     'svc_predict_fbm', 'decisiontreeclassifier_predict_fbm', 'randomforestclassifier_predict_fbm',
     'randomforestclassifier_predict']]

sql_engine.data_to_sql(df_predict, path.product_group_predict, 'append', config.connet_product_db_sql)

print('done')

"""
# 机器学习训练数据准备
df_group_knn = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password, config.product_database,
                                             sql.sampling_knn_sql)
models = model_parameter(2, 70)
model_training(df_group_knn, models)

# 机器学习预测
df_group = df_group_knn.drop(columns=['clue_tag'], errors='ignore')
for model_name in models.keys():
    df_group[f'{model_name}_predict'] = model_predict(df_group, model_name)

# df_predict = df_group[
#     ['原ASIN', '数据更新时间', 'svc_predict', 'decisiontreeclassifier_predict', 'randomforestclassifier_predict',
#      'kneighborsclassifier_predict']]
df_predict = df_group[
    ['原ASIN', '数据更新时间', 'randomforestclassifier_predict']]
sql_engine.data_to_sql(df_predict, path.product_group_sampling_predict_fbm, 'append', config.connet_product_db_sql)

print('done')

# model_parameter(15, 200)
#
# model_parameter(2, 70)
"""
