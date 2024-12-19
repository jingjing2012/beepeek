def extract_insert_statements(file_path):
    with open(file_path, 'r', encoding='utf-8') as file, \
            open("./report11.sql", 'a', encoding='utf-8') as output_file:
        lines = []
        inside_insert = False
        table_name = 'pt_product_report'

        # table_name_list = ['pt_product_report', 'pt_product_get_cpc', 'pt_keywords', 'pt_product_report',
        #                    'pt_product_report', 'pt_product_report', 'pt_product_report']

        for line in file:
            # 检查是否是插入特定表的语句
            if line.strip().lower().startswith(f"insert into `{table_name}`") or inside_insert:
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
    extract_insert_statements(r"\\192.168.10.244\数字化选品\OE数据\product_report_sql\sellersprite_202411.sql")
