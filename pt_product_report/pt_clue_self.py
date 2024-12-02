import warnings

import pandas as pd
from pandas.errors import SettingWithCopyWarning
import logging
from datetime import datetime

from conn import sql_engine, mysql_config as config
import pt_product_report_path as data_path
import pt_product_sql as sql

# 设置日志
logging.basicConfig(filename='script_log.log', level=logging.INFO)


def main():
    logging.info(f'Script started at {datetime.now()}')
    # 你的代码逻辑
    logging.info(f'Script ended at {datetime.now()}')


# 自主提报数据获取
def data_read(path, args):
    df_data = pd.read_excel(path, sheet_name=args)
    df_data = df_data.dropna(axis=0, how='all', subset='ASIN', inplace=False)
    df_data = df_data.drop_duplicates(subset='ASIN')
    return df_data


# 历史数据去重
def data_duplicate(df, df_history):
    df['duplicate_tag'] = df['asin'].isin(df_history['asin']).map({True: 1, False: 0})
    df = df[df['duplicate_tag'] == 0]
    return df.drop('duplicate_tag', axis=1)


if __name__ == "__main__":
    main()

# 忽略与 Pandas SettingWithCopyWarning 模块相关的警告
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

# 忽略与 Pandas SQL 模块相关的警告
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.io.sql")
warnings.filterwarnings("ignore", category=UserWarning, module='pandas')
warnings.filterwarnings("ignore", message="pandas only support SQLAlchemy connectable.*")

# 数据读取
df_product_self = data_read(data_path.pt_product_table_self, data_path.sheet_self)
df_product_sampling = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password, config.sbi_database,
                                                    sql.sql_product_sampling)
df_product_sbi = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password, config.product_database,
                                               sql.sql_product_sbi)
df_position_self = data_read(data_path.position_table_self, data_path.sheet_position)
df_product_fbm1 = data_read(data_path.pt_product_table_fbm, data_path.sheet_fbm1)
df_product_fbm2 = data_read(data_path.pt_product_table_fbm, data_path.sheet_fbm2)

df_shop_self = data_read(data_path.pt_shop_table_self, data_path.sheet_shop)

# 数据处理
df_product_self = df_product_self[['ASIN', '提报人员', '提报日期', '批次Tag']]
df_product_self.rename(columns={'ASIN': 'asin',
                                '提报人员': 'name',
                                '提报日期': 'update_time',
                                '批次Tag': 'asin_tag'}, inplace=True)

df_position_self.rename(columns={'ASIN': 'asin',
                                 '头图链接': 'image',
                                 '站点': 'country',
                                 '价格_美元': 'price_us',
                                 '价格_英镑': 'price_uk',
                                 '价格_欧元': 'price_de',
                                 '星级': 'rating',
                                 '星数': 'ratings',
                                 '最长边_厘米': 'length_max',
                                 '次长边_厘米': 'length_mid',
                                 '最短边_厘米': 'length_min',
                                 '包裹重_克': 'weight',
                                 '采购价_人民币': 'price_value',
                                 '是否查询UK站点': 'status_uk',
                                 '是否查询DE站点': 'status_de',
                                 '提报人': 'name',
                                 '提报日期': 'update_time'}, inplace=True)
df_position_self = df_position_self[df_position_self['去重'] == 1]
df_position_self['status_uk'] = df_position_self['status_uk'].map({'是': 1, '否': 0})
df_position_self['status_de'] = df_position_self['status_de'].map({'是': 1, '否': 0})

df_product_fbm = pd.concat([df_product_fbm1, df_product_fbm2])
df_product_fbm.rename(columns={'ASIN': 'asin',
                               '提报日期': 'update_time',
                               '类型': 'data_tag'}, inplace=True)

df_shop_self.rename(columns={'ASIN': 'asin',
                             '提报人员': 'name',
                             '提报日期': 'task_time',
                             '任务名称': 'task_tag'}, inplace=True)

# 爬取数据去重
df_clue = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                        config.clue_self_database, sql.sql_sellersprite_clue_self)
df_position = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                            config.clue_position_database, sql.sql_sellersprite_clue_position)
df_shop = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                        config.clue_shop_database, sql.sql_sellersprite_clue_shop)

df_product_self_pt = data_duplicate(df_product_self, df_clue)
df_product_sampling_pt = data_duplicate(df_product_sampling, df_clue)
df_product_sbi_pt = data_duplicate(df_product_sbi, df_clue)
df_clue_position = data_duplicate(df_position_self, df_position)
df_product_fbm_pt = data_duplicate(df_product_fbm, df_clue)
df_clue_shop = data_duplicate(df_shop_self, df_shop)

# 历史记录数据去重
df_clue_self_history = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password, config.product_database,
                                                     sql.sql_clue_self_history)
df_clue_sampling_history = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password,
                                                         config.product_database,
                                                         sql.sql_clue_sampling_history)
df_clue_sbi_history = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password, config.product_database,
                                                    sql.sql_clue_sbi_history)
df_clue_fbm_history = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password, config.product_database,
                                                    sql.sql_clue_fbm_history)

df_clue_self = data_duplicate(df_product_self, df_clue_self_history)
df_clue_sampling = data_duplicate(df_product_sampling, df_clue_sampling_history)
df_clue_sbi = data_duplicate(df_product_sbi, df_clue_sbi_history)
df_clue_fbm = data_duplicate(df_product_fbm, df_clue_fbm_history)

# 数据整合
df_clue_self_pt = df_product_self_pt[['asin', 'update_time']]
df_clue_self_pt['data_tag'] = '自主提报'

df_clue_sampling_pt = df_product_sampling_pt[['asin', 'update_time']]
df_clue_sampling_pt['data_tag'] = '历史开品'

df_clue_sbi = df_clue_sbi[['asin']]
df_clue_sbi_pt = df_product_sbi_pt[['asin', 'update_time']]
df_clue_sbi_pt['data_tag'] = '历史开售'

df_clue_pt = pd.concat([df_clue_self_pt, df_clue_sampling_pt, df_clue_sbi_pt, df_clue_fbm], ignore_index=True)

df_clue_position = df_clue_position[
    ['asin', 'image', 'price_us', 'price_uk', 'price_de', 'rating', 'ratings', 'length_max', 'length_mid', 'length_min',
     'weight', 'price_value', 'status_uk', 'status_de', 'name', 'update_time']]

# 数据入库
sql_engine.data_to_sql(df_clue_pt, data_path.pt_clue_asin, "append", config.connet_clue_self_db_sql)

sql_engine.data_to_sql(df_clue_self, data_path.product_clue_self, "append", config.connet_product_db_sql)
sql_engine.data_to_sql(df_clue_sampling, data_path.product_clue_sampling, "append", config.connet_product_db_sql)
sql_engine.data_to_sql(df_clue_sbi, data_path.product_clue_sbi, "append", config.connet_product_db_sql)
sql_engine.data_to_sql(df_clue_fbm, data_path.product_clue_fbm, "append", config.connet_product_db_sql)

# 竞品提报数据入库
sql_engine.data_to_sql(df_clue_position, data_path.pt_clue_asin, "append", config.connet_clue_position_db_sql)

# 店铺挖掘提报数据入库
sql_engine.data_to_sql(df_clue_shop, data_path.pt_seed_asin, "append", config.connet_clue_shop_db_sql)
