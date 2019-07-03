def test_mysql_connector():
        import mysql.connector
        import json
        with open('config.json','r') as f:
            config=json.load(f)
            f.close()
        mydb = mysql.connector.connect(**config)
        cursor = mydb.cursor()
        sql_query = """insert into mtrp_alarm_test(alarmTime,content,fileUrl) values('2019-03-28 19:11:50','attention please!','urllalala')"""
        #print(sql_query)
        #sql_query = "select * from mtrp_alarm_type;"
        #cursor.execute(sql_query)
        for item in cursor:
            print(item)
        try:
            cursor.execute(sql_query)
            mydb.commit()
            print("inser success")
        except:
            print("insert failed!")
        #print(mydb)
        mydb.close()
    

test_mysql_connector()
