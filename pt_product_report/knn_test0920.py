from conn import sql_engine, mysql_config as config
import pt_product_sql as sql
import pt_group_knn as knn

# 机器学习训练数据准备
df_group_knn = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password, config.product_database,
                                             sql.sampling_knn_sql)
# 数据获取
df_group = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password, config.product_database,
                                         sql.sampling_knn_group_sql)
df_group = df_group.drop(df_group.columns[60:], axis=1)

models = knn.model_parameter()
knn.model_training(df_group_knn, models)

# 机器学习预测
for model_name in models.keys():
    df_group[f'{model_name}_predict'] = knn.model_predict(df_group, model_name)

df_group.to_excel(r'C:\Users\Administrator\Desktop\knn_test_roc_auc.xlsx')
