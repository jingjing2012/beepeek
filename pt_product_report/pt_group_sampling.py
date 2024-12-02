from datetime import datetime
import numpy as np
import pandas as pd
import pymysql
import warnings

from pandas.core.common import SettingWithCopyWarning
from pandas.errors import PerformanceWarning

from conn import sql_engine, mysql_config as config
import pt_product_report_parameter as para
import pt_product_report_path as path
import pt_product_sql as sql


# 数据连接
def connect_pt_product(hostname, database, product_sql_pt):
    conn_oe = pymysql.connect(host=hostname, user=config.oe_username, passwd=config.oe_password,
                              database=database, charset='utf8')
    df = pd.read_sql(product_sql_pt, conn_oe)
    return df


# 数据打标签
def get_cut(df, col_str, bins_cut, labels_cut):
    return pd.cut(df[col_str], bins_cut, right=False, labels=labels_cut, include_lowest=True)


# PMI计算
def pmi_score(row):
    P_score, M_score = 0, 0
    P_tags_correction, M_tags_correction, I_tags_correction = [], [], []
    for col in para.pmi_list:
        col_pmi = col + "_score"
        col_pmi_tag = col + "_tag"
        if row[col_pmi_tag] in para.p_list:
            P_score += round(row[col_pmi], 1)
            P_tags_correction.append(str(row[col_pmi_tag]))
        elif row[col_pmi_tag] in para.m_list:
            M_score += round(row[col_pmi], 1)
            M_tags_correction.append(str(row[col_pmi_tag]))
        elif row[col_pmi_tag] in para.i_list:
            I_tags_correction.append(str(row[col_pmi_tag]))
    row['P得分'] = P_score
    row['P标签'] = ','.join(P_tags_correction)
    row['M得分'] = M_score
    row['M标签'] = ','.join(M_tags_correction)
    row['I标签'] = ','.join(I_tags_correction)
    return row


def pmi_melt(df, vars_list):
    df_melt = df.melt(id_vars=['原ASIN'], value_vars=vars_list, var_name='pmi_tag', value_name='标签')
    return df_melt


# 基础清洗
def df_clear(df, clear_id):
    df = df.replace('none', np.nan, regex=False)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.drop_duplicates(subset=[clear_id])
    df = df.dropna(subset=[clear_id])
    return df


# 导入数据库
def save_to_sql(df, table, conn, args):
    df.to_sql(table, conn, if_exists=args, index=False, chunksize=1000)


# -------------------------------------------------------------------------------------------------------------------

# 忽略与 Pandas SettingWithCopyWarning 模块相关的警告
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

# 忽略与 Pandas SQL 模块相关的警告
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.io.sql")

# 忽略除以零的警告
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log2")

warnings.filterwarnings("ignore", message="pandas only support SQLAlchemy connectable.*")
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented.", category=PerformanceWarning)

# 1.数据连接
traffic_df = connect_pt_product(config.oe_hostname, config.product_database, sql.sql_asin_group_sampling)
corretion_pmi_df = pd.read_excel(r'C:\Users\Administrator\Desktop\pmi1.xlsx')

# 2.聚合字段计算

# S规模分析

# TOP5月均销额
traffic_df['TOP5月均销额_score'] = get_cut(traffic_df, 'TOP5月均销额', para.s_sales_bins_correction,
                                       para.s_sales_labels_correction)
traffic_df['TOP5月均销额_tag'] = get_cut(traffic_df, 'TOP5月均销额', para.s_sales_bins_correction, para.s_sales_tags_correction)

# 价格集中度
traffic_df['价格集中度_score'] = get_cut(traffic_df, '价格集中度', para.m_price_bins_correction, para.m_price_labels_correction)
traffic_df['价格集中度_tag'] = get_cut(traffic_df, '价格集中度', para.m_price_bins_correction, para.m_price_tags_correction)

# 预估平均毛利率
traffic_df['预估平均毛利率_score'] = get_cut(traffic_df, '预估平均毛利率', para.m_gross_bins_correction,
                                      para.m_gross_labels_correction)
traffic_df['预估平均毛利率_tag'] = get_cut(traffic_df, '预估平均毛利率', para.m_gross_bins_correction, para.m_gross_tags_correction)

# 预估平均资金利用率
traffic_df['预估平均资金利用率_score'] = get_cut(traffic_df, '预估平均资金利用率', para.m_revenue_bins_correction,
                                        para.m_revenue_labels_correction)
traffic_df['预估平均资金利用率_tag'] = get_cut(traffic_df, '预估平均资金利用率', para.m_revenue_bins_correction,
                                      para.m_revenue_tags_correction)

# 加权FBA运费
traffic_df['加权FBA运费_score'] = get_cut(traffic_df, '加权FBA运费', para.m_fba_bins_correction, para.m_fba_labels_correction)
traffic_df['加权FBA运费_tag'] = get_cut(traffic_df, '加权FBA运费', para.m_fba_bins_correction, para.m_fba_tags_correction)

# FBM配送占比
traffic_df['FBM配送占比_score'] = get_cut(traffic_df, 'FBM配送占比', para.m_fbm_bins_correction, para.m_fbm_labels_correction)
traffic_df['FBM配送占比_tag'] = get_cut(traffic_df, 'FBM配送占比', para.m_fbm_bins_correction, para.m_fbm_tags_correction)

# 直发FBM销额占比
conditions_fbm_rate = (traffic_df['直发FBM产品占比'] >= 0.15) | (traffic_df['直发FBM销额占比'] >= 0.2)

conditions_fbm_rate_correction = (traffic_df['直发FBM产品占比'] >= 0.12) | (traffic_df['直发FBM销额占比'] >= 0.15)

traffic_df['直发FBM产品占比_score'] = np.where(conditions_fbm_rate_correction, 2, np.nan)
traffic_df['直发FBM产品占比_tag'] = np.where(conditions_fbm_rate_correction, "直发FBM多", np.nan)

# 直发FBM月均销额
traffic_df['直发FBM月均销额_score'] = get_cut(traffic_df, '直发FBM月均销额', para.m_fbm_cal_sales_bins_correction,
                                        para.m_fbm_cal_sales_labels_correction)
traffic_df['直发FBM月均销额_tag'] = get_cut(traffic_df, '直发FBM月均销额', para.m_fbm_cal_sales_bins_correction,
                                      para.m_fbm_cal_sales_tags_correction)

traffic_df['直发FBM产品数'] = round(traffic_df['直发FBM产品占比'] * traffic_df['有销额竞品款数'])
traffic_df['直发FBM月均销额_score'] = np.where(traffic_df['直发FBM产品数'] >= 3, traffic_df['直发FBM月均销额_score'], np.nan)
traffic_df['直发FBM月均销额_tag'] = np.where(traffic_df['直发FBM产品数'] >= 3, traffic_df['直发FBM月均销额_tag'], np.nan)

# I内卷分析
# AMZ直营销额占比
traffic_df['AMZ直营销额占比_score'] = get_cut(traffic_df, 'AMZ直营销额占比', para.i_amz_bins_correction,
                                        para.i_amz_labels_correction)
traffic_df['AMZ直营销额占比_tag'] = get_cut(traffic_df, 'AMZ直营销额占比', para.i_amz_bins_correction, para.i_amz_tags_correction)

# 大牌商标销额占比
traffic_df['大牌商标销额占比_score'] = get_cut(traffic_df, '大牌商标销额占比', para.i_famous_bins_correction,
                                       para.i_famous_labels_correction)
traffic_df['大牌商标销额占比_tag'] = get_cut(traffic_df, '大牌商标销额占比', para.i_famous_bins_correction,
                                     para.i_famous_tags_correction)

# 中国卖家占比
traffic_df['中国卖家占比_score'] = get_cut(traffic_df, '中国卖家占比', para.i_cn_bins_correction, para.i_cn_labels_correction)
traffic_df['中国卖家占比_tag'] = get_cut(traffic_df, '中国卖家占比', para.i_cn_bins_correction, para.i_cn_tags_correction)

# TOP5平均LQS
traffic_df['TOP5平均LQS_score'] = get_cut(traffic_df, 'TOP5平均LQS', para.i_lqs_top5_bins_correction,
                                        para.i_lqs_top5_labels_correction)
traffic_df['TOP5平均LQS_tag'] = get_cut(traffic_df, 'TOP5平均LQS', para.i_lqs_top5_bins_correction,
                                      para.i_lqs_top5_tags_correction)

# 冒出品平均LQS
traffic_df['冒出品平均LQS_score'] = get_cut(traffic_df, '冒出品平均LQS', para.i_lqs_bins_correction, para.i_lqs_labels_correction)
traffic_df['冒出品平均LQS_tag'] = get_cut(traffic_df, '冒出品平均LQS', para.i_lqs_bins_correction, para.i_lqs_tags_correction)

# 冒出品A+占比
traffic_df['冒出品A+占比_score'] = get_cut(traffic_df, '冒出品A+占比', para.i_ebc_bins_correction, para.i_ebc_labels_correction)
traffic_df['冒出品A+占比_tag'] = get_cut(traffic_df, '冒出品A+占比', para.i_ebc_bins_correction, para.i_ebc_tags_correction)

# 冒出品视频占比
traffic_df['冒出品视频占比_score'] = get_cut(traffic_df, '冒出品视频占比', para.i_video_bins_correction,
                                      para.i_video_labels_correction)
traffic_df['冒出品视频占比_tag'] = get_cut(traffic_df, '冒出品视频占比', para.i_video_bins_correction, para.i_video_tags_correction)

# 冒出品QA占比
traffic_df['冒出品QA占比_score'] = get_cut(traffic_df, '冒出品QA占比', para.i_qa_bins_correction, para.i_qa_labels_correction)
traffic_df['冒出品QA占比_tag'] = get_cut(traffic_df, '冒出品QA占比', para.i_qa_bins_correction, para.i_qa_tags_correction)

# 动销品平均星级
traffic_df['动销品平均星级_score'] = get_cut(traffic_df, '动销品平均星级', para.i_rating_bins_correction,
                                      para.i_rating_labels_correction)
traffic_df['动销品平均星级_tag'] = get_cut(traffic_df, '动销品平均星级', para.i_rating_bins_correction, para.i_rating_tags_correction)

# 冒出品低星占比
conditions_rating = [
    (traffic_df['冒出品低星占比'] >= 0.25) & (traffic_df['冒出品低星占比'] < 0.4) & (traffic_df['预估平均毛利率'] >= 0.2) | (
            traffic_df['预估平均资金利用率'] >= 0.8) & (traffic_df['动销品平均星级'] < 4.2),
    (traffic_df['冒出品低星占比'] >= 0.4) & (traffic_df['预估平均毛利率'] >= 0.2) | (traffic_df['预估平均资金利用率'] >= 0.8) & (
            traffic_df['动销品平均星级'] < 4.2)]
traffic_df['冒出品低星占比_score'] = np.select(conditions_rating, para.i_rating_rate_labels_correction, np.nan)
traffic_df['冒出品低星占比_tag'] = np.select(conditions_rating, para.i_rating_rate_tags_correction, np.nan)

# L长尾分析
# TOP5销额占比
traffic_df['TOP5销额占比_score'] = get_cut(traffic_df, 'TOP5销额占比', para.l_sales_top5_bins_correction,
                                       para.l_sales_top5_labels_correction)
traffic_df['TOP5销额占比_tag'] = get_cut(traffic_df, 'TOP5销额占比', para.l_sales_top5_bins_correction,
                                     para.l_sales_top5_tags_correction)

# 非TOP5销额占比
traffic_df['非TOP5销额占比_score'] = get_cut(traffic_df, '非TOP5销额占比', para.l_sales_rate_bins_correction,
                                        para.l_sales_rate_labels_correction)
traffic_df['非TOP5销额占比_tag'] = get_cut(traffic_df, '非TOP5销额占比', para.l_sales_rate_bins_correction,
                                      para.l_sales_rate_tags_correction)

# 非TOP5月均销额
traffic_df['非TOP5月均销额_score'] = get_cut(traffic_df, '非TOP5月均销额', para.l_sales_bins_correction,
                                        para.l_sales_labels_correction)
traffic_df['非TOP5月均销额_tag'] = get_cut(traffic_df, '非TOP5月均销额', para.l_sales_bins_correction,
                                      para.l_sales_tags_correction)

# 动销品变体中位数
traffic_df['动销品变体中位数_score'] = get_cut(traffic_df, '动销品变体中位数', para.l_variations_bins_correction,
                                       para.l_variations_labels_correction)
traffic_df['动销品变体中位数_tag'] = get_cut(traffic_df, '动销品变体中位数', para.l_variations_bins_correction,
                                     para.l_variations_tags_correction)

# E新品冒出
# 平均开售月数
traffic_df['平均开售月数_score'] = get_cut(traffic_df, '平均开售月数', para.e_month_bins_correction, para.e_month_labels_correction)
traffic_df['平均开售月数_tag'] = get_cut(traffic_df, '平均开售月数', para.e_month_bins_correction, para.e_month_tags_correction)

traffic_df['平均开售月数_score'] = np.where((traffic_df['平均开售月数'] >= 48) & (traffic_df['平均星数'] < 200), 0,
                                      traffic_df['平均开售月数_score'])
traffic_df['平均开售月数_tag'] = np.where((traffic_df['平均开售月数'] >= 48) & (traffic_df['平均星数'] < 200), np.nan,
                                    traffic_df['平均开售月数_tag'])

# 冒出品新品占比
traffic_df['冒出品新品占比_score'] = get_cut(traffic_df, '冒出品新品占比', para.e_new_bins_correction, para.e_new_labels_correction)
traffic_df['冒出品新品占比_tag'] = get_cut(traffic_df, '冒出品新品占比', para.e_new_bins_correction, para.e_new_tags_correction)

# 新品平均月销额
traffic_df['新品平均月销额_score'] = get_cut(traffic_df, '新品平均月销额', para.e_new_sales_bins_correction,
                                      para.e_new_sales_labels_correction)
traffic_df['新品平均月销额_tag'] = get_cut(traffic_df, '新品平均月销额', para.e_new_sales_bins_correction,
                                    para.e_new_sales_tags_correction)

# 平均星数
traffic_df['平均星数_score'] = get_cut(traffic_df, '平均星数', para.e_ratings_bins_correction, para.e_ratings_labels_correction)
traffic_df['平均星数_tag'] = get_cut(traffic_df, '平均星数', para.e_ratings_bins_correction, para.e_ratings_tags_correction)

# 三标新品占比
traffic_df['三标新品占比_score'] = get_cut(traffic_df, '三标新品占比', para.e_new_lables_bins_correction,
                                     para.e_new_lables_labels_correction)
traffic_df['三标新品占比_tag'] = get_cut(traffic_df, '三标新品占比', para.e_new_lables_bins_correction,
                                   para.e_new_lables_tags_correction)

# 月销增长率
traffic_df['月销增长率_score'] = get_cut(traffic_df, '月销增长率', para.e_sales_bins_correction, para.e_sales_labels_correction)
traffic_df['月销增长率_tag'] = get_cut(traffic_df, '月销增长率', para.e_sales_bins_correction, para.e_sales_tags_correction)

# PMI计算
traffic_df = traffic_df.apply(lambda row: pmi_score(row), axis=1)

traffic_df['PMI得分'] = traffic_df['P得分'] + traffic_df['M得分']

condition_pmi = traffic_df['有销额竞品款数'] * 1 >= 3
pmi_pre_list = ['PMI得分', 'P得分', 'M得分', 'P标签', 'M标签', 'I标签']

for pmi_col in pmi_pre_list:
    traffic_df[pmi_col] = np.where(condition_pmi, traffic_df[pmi_col], np.nan)

traffic_df['PMI得分'].fillna(0, inplace=True)

# -------------------------------V3版本-------------------------推荐级别重算------------------------------------

# 推荐级别v2
traffic_df['推荐级别v2'] = get_cut(traffic_df, 'PMI得分', para.pmi_bins_correction, para.pmi_labels_correction)

# 综合推荐级别
traffic_df['推荐级别v2'] = traffic_df['推荐级别v2'].astype(int)

traffic_df['综合推荐级别'] = get_cut(traffic_df, '推荐级别v2', para.recommend_bins_correction, para.recommend_labels_correction)
traffic_df['综合推荐级别'] = np.where(traffic_df['有销额竞品款数'] >= 5, traffic_df['综合推荐级别'], '<数据不足>')

# 校正更新时间
now = datetime.now()
traffic_df['校正更新时间'] = now

traffic_tag = ['原ASIN', 'TOP5月均销额_tag', '价格集中度_tag', '预估平均毛利率_tag', '预估平均资金利用率_tag', '加权FBA运费_tag', 'FBM配送占比_tag',
               '直发FBM产品占比_tag', '直发FBM月均销额_tag', 'AMZ直营销额占比_tag', '大牌商标销额占比_tag', '中国卖家占比_tag', 'TOP5平均LQS_tag',
               '冒出品平均LQS_tag', '冒出品A+占比_tag', '冒出品视频占比_tag', '冒出品QA占比_tag', '动销品平均星级_tag', '冒出品低星占比_tag',
               'TOP5销额占比_tag', '非TOP5销额占比_tag', '非TOP5月均销额_tag', '动销品变体中位数_tag', '平均开售月数_tag', '冒出品新品占比_tag',
               '新品平均月销额_tag', '平均星数_tag', '三标新品占比_tag', '月销增长率_tag']

pmi_df = traffic_df.groupby(traffic_df['原ASIN'], as_index=False).apply(lambda x: pmi_melt(x, traffic_tag))
pmi_df['标签'] = pmi_df['标签'].replace({'': np.nan, ' ': np.nan, 'nan': np.nan})

pmi_df['data_id'] = pmi_df['原ASIN'] + ' | ' + pmi_df['标签']
pmi_df = df_clear(pmi_df, 'data_id')

df_pmi = pmi_df[['原ASIN', '标签']]
df_pmi = df_pmi[df_pmi['标签'].notnull()]

df_pmi = df_pmi.merge(corretion_pmi_df, how='left', left_on='标签', right_on='子标签名')

# 3.字段整合
# 聚合表
df_group = traffic_df[
    ['原ASIN', 'clue_tag', 'P得分', 'M得分', 'PMI得分', 'P标签', 'M标签', 'I标签', '推荐级别v2', '综合推荐级别', 'update_time', '校正更新时间']]
df_group = df_clear(df_group, '原ASIN')

df_group.rename(columns={'clue_tag': '检测类别', 'update_time': '数据更新时间'}, inplace=True)

# pmi校正表
df_pmi = df_pmi.merge(df_group, how='left', on='原ASIN')

df_pmi = df_pmi[['原ASIN', '检测类别', '标签', '标签编码', '得分', 'PMI类型(P/M/I)', '推荐级别v2', '综合推荐级别', '数据更新时间', '校正更新时间']]

df_pmi.rename(columns={'PMI类型(P/M/I)': '标签类型'}, inplace=True)

corretion_pmi_df['校正更新时间'] = now

corretion_pmi_df = corretion_pmi_df[['标签编码', '所属SMILE大维度', '标签组', '子标签名', '条件/规则描述', '得分', 'PMI类型(P/M/I)', '校正更新时间']]

# 4.存入数据库
pt_product_conn = sql_engine.create_conn(config.connet_product_db_sql)

save_to_sql(df_group, path.product_group_correction_sampling, pt_product_conn, 'append')
save_to_sql(df_pmi, path.product_group_correction_pmi, pt_product_conn, 'append')
save_to_sql(corretion_pmi_df, path.product_group_sampling_pmi, pt_product_conn, 'append')
