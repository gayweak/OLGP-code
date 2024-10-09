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

sql="select StartID,EndID from Relations where time=2020"
cursor.execute(sql)
relations1=cursor.fetchall()
relations1=[item[0] for item in relations1]

sql="select StartID,EndID from Relations where time=2021"
cursor.execute(sql)
relations2=cursor.fetchall()
relations2=[item[0] for item in relations2]

sql="select StartID,EndID from Relations where time=2022"
cursor.execute(sql)
relations3=cursor.fetchall()
relations3=[item[0] for item in relations3]

relations4=list(set(relations1+relations2+relations3))
for startID,endID in relations4:
    sql="insert into Relations(StartID,EndID,Weight,time) values (%s,%s,%s,%s)"
    values=(startID,endID,1,2020-2022)
    cursor.execute(sql,values)

sql="select StartID,EndID from Relations where location=Evanston and time=2020"
cursor.execute(sql)
relations1=cursor.fetchall()
relations1=[item[0] for item in relations1]

sql="select StartID,EndID from Relations where location=Chicago and time-2020"
cursor.execute(sql)
relations2=cursor.fetchall()
relations2=[item[0] for item in relations2]

sql="select StartID,EndID from Relations where location=Springfield and time=2020"
cursor.execute(sql)
relations3=cursor.fetchall()
relations3=[item[0] for item in relations3]

relations4=list(set(relations1+relations2+relations3))
for startID,endID in relations4:
    sql="insert into Relations(StartID,EndID,Weight,time) values (%s,%s,%s,%s)"
    values=(startID,endID,1, 2020, illinois)
    cursor.execute(sql,values)

connection.commit()
cursor.close()
connection.close()
end=time.perf_counter()
print(end-start)