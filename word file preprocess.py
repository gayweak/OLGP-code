import docx
import pymysql
from docx import Document

# 读取Word文档
doc = Document('twitter.docx')  # 替换为你的Word文档路径

# 初始化数据库连接
connection = pymysql.connect(
    host='127.0.0.1',
    port=3306,
    user='root',
    password='098831',
    database='database1'
)

cursor = connection.cursor()

# 遍历Word文档中的每一行数据
for paragraph in doc.paragraphs:
    # 假设每一行数据格式为: "StartID EndID"
    line = paragraph.text.strip()
    if line:  # 确保行不为空
        # 分割每行的两个数字
        start_id, end_id = map(int, line.split())

        # 插入数据到Relations表中
        sql = "INSERT INTO Relations (StartID, EndID, Weight, time) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (start_id, end_id, 1, 2020))

# 提交事务并关闭连接
connection.commit()
cursor.close()
connection.close()

print("Data inserted successfully.")
