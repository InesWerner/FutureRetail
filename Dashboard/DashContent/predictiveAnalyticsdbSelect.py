import mysql.connector
import datetime

def selectShampoo() -> dict:
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="futureretail"
        )
    connection.autocommit = True
    mycursor = connection.cursor()
    mycursor.execute("SELECT * FROM shampoo")
    result = mycursor.fetchall()
    dht_dict = dict()
    for x in result: 
        dht_dict.update({(str(x[1]).split(" ")[1]).replace(":","-"): float(x[0])})
        mycursor.close()
        connection.close()
    return dht_dict

def selectSonnencreme() -> dict:
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="futureretail"
        )
    connection.autocommit = True
    mycursor = connection.cursor()
    mycursor.execute("SELECT * FROM sonnencreme")
    result = mycursor.fetchall()
    dht_dict = dict()
    for x in result: 
        dht_dict.update({(str(x[1]).split(" ")[1]).replace(":","-"): float(x[0])})
        mycursor.close()
        connection.close()
    return dht_dict

def selectMandelmus() -> dict:
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="futureretail"
        )
    connection.autocommit = True
    mycursor = connection.cursor()
    mycursor.execute("SELECT * FROM mandelmus")
    result = mycursor.fetchall()
    dht_dict = dict()
    for x in result: 
        dht_dict.update({(str(x[1]).split(" ")[1]).replace(":","-"): float(x[0])})
        mycursor.close()
        connection.close()
    return dht_dict