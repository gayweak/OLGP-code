import time
import pymysql
import networkx as nx
start=time.perf_counter()
connection=pymysql.connect(
    host='127.0.0.1',
    port=3306,
    user='root',
    password='098831',
    database='database1'
)
cursor=connection.cursor()

sql="create table Relations(RelationID int NOT NULL AUTO_INCREMENT PRIMARY KEY,StartID int NOT NULL,EndID int NOT NULL,Weight decimal(10,2) NOT NULL,time varchar(255) NOT NULL"
cursor.execute(sql)

sql="create table Relation(RelationID int NOT NULL AUTO_INCREMENT PRIMARY KEY,StartID int NOT NULL,EndID int NOT NULL,Weight decimal(10,2) NOT NULL,time varchar(255) NOT NULL"
cursor.execute(sql)

sql="create table Entities(EntityID INT AUTO_INCREMENT PRIMARY KEY, EntityType VARCHAR(255) NOT NULL, EntityName VARCHAR(255) NOT NULL)"
cursor.execute(sql)

connection.commit()
cursor.close()
connection.close()
end=time.perf_counter()
print(end-start)