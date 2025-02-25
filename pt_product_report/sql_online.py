# 本地sql文件写入
from conn import sql_engine, mysql_config as config


def extract_insert_statements(file_path, target_table):
    with open(file_path, 'r', encoding='utf-8') as file, \
            open(f"./{target_table}.sql", 'a', encoding='utf-8') as output_file:
        lines = []
        inside_insert = False

        for line in file:
            # 检查是否是插入特定表的语句
            if line.strip().lower().startswith(f"insert into `{target_table}`") or inside_insert:
                lines.append(line)
                # 检查是否是插入语句的结束
                if line.strip().endswith(';'):
                    inside_insert = False
                    # 将收集到的行合并为一个完整的SQL语句
                    statement = ''.join(lines).strip()
                    print(statement)  # 打印或处理完整的SQL语句
                    output_file.write(statement + '\n')
                    lines = []  # 重置行列表
            elif line.strip().lower().startswith("insert into"):
                # 检查是否是插入其他表的语句，如果是，则重置
                inside_insert = False
                lines = []


if __name__ == '__main__':
    # 调用函数，传入SQL文件的路径
    sellersprite_target_table_list = ['pt_relation_traffic']
    # sellersprite_target_table_list = ['pt_product_get_group','pt_relation_traffic', 'pt_relevance_asins']
    sellersprite_file_path = r"\\192.168.10.244\数字化选品\OE数据\product_report_sql\sellersprite_202401.sql"

    # for sellersprite_target_table in sellersprite_target_table_list:
        # extract_insert_statements(sellersprite_file_path, sellersprite_target_table)

    oe_target_table = 'pt_keywords'
    oe_file_path = r"\\192.168.10.244\数字化选品\OE数据\niche_sql\update\oe_us_20241228.sql"
    extract_insert_statements(oe_file_path, oe_target_table)
