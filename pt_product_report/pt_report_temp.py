import pymysql
import conn.mysql_config as config

db_config = {
    "host": config.sellersprite_hostname,
    "user": config.sellersprite_username,
    "password": config.sellersprite_password,
    "database": "pt_report_temp",
}


def extract_table_sql_from_large_file(file_path, table_name):
    """
    从大SQL文件中逐行提取指定表的SQL
    """
    table_sql = []
    inside_table_section = False

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                if f"CREATE TABLE `{table_name}`" in line or f"INSERT INTO `{table_name}`" in line:
                    inside_table_section = True
                if inside_table_section:
                    table_sql.append(line.strip())
                if inside_table_section and ";" in line and ("INSERT INTO" in line or "CREATE TABLE" in line):
                    inside_table_section = False

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except UnicodeDecodeError:
        raise ValueError("File encoding mismatch. Ensure the file is UTF-8 encoded.")

    if not table_sql:
        raise ValueError(f"No SQL statements found for the table '{table_name}'.")
    return "\n".join(table_sql)


def restore_table_to_mysql(sql_content, db_config, target_table_name):
    """
    将提取的SQL内容恢复到目标MySQL数据库表
    """
    sql_content = sql_content.replace("pt_product_report", target_table_name)

    try:
        connection = pymysql.connect(
            host=db_config["host"],
            user=db_config["user"],
            password=db_config["password"],
            database=db_config["database"]
        )
    except pymysql.err.OperationalError as e:
        print(f"Database connection failed: {e}")
        raise

    try:
        with connection.cursor() as cursor:
            # 删除目标表（如存在）
            try:
                cursor.execute(f"DROP TABLE IF EXISTS `{target_table_name}`")
            except pymysql.err.InternalError as e:
                print(f"Failed to drop existing table `{target_table_name}`: {e}")
                raise

            # 执行SQL语句
            statements = sql_content.split(";")
            for statement in statements:
                statement = statement.strip()
                if statement:
                    print(f"Executing statement: {statement[:100]}...")
                    try:
                        cursor.execute(statement)
                    except pymysql.err.ProgrammingError as e:
                        print(f"SQL syntax error in statement: {statement[:100]}...")
                        print(f"Error details: {e}")
                        raise
            connection.commit()
            print(f"Data successfully restored to {target_table_name} in {db_config['database']} database.")
    except Exception as e:
        connection.rollback()
        print(f"Error during restoration: {e}")
        raise
    finally:
        connection.close()


sql_file_path = r"\\192.168.10.244\数字化选品\OE数据\product_report_sql\sellersprite_202408.sql"
source_table = "pt_product_report"
target_table = "pt_product_report_202408"
db_config = {
    "host": config.sellersprite_hostname,
    "user": config.sellersprite_username,
    "password": config.sellersprite_password,
    "database": "pt_report_temp",
}

try:
    print("Extracting table SQL from source file...")
    table_sql = extract_table_sql_from_large_file(sql_file_path, source_table)

    print("Restoring table to MySQL...")
    restore_table_to_mysql(table_sql, db_config, target_table)

except FileNotFoundError as fnf_error:
    print(f"File not found: {fnf_error}")
except ValueError as val_error:
    print(f"Value error: {val_error}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
