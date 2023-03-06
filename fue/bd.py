import sqlite3


def get():
    cur.execute(f"SELECT * FROM tabs")
    a = cur.fetchall()
    db.commit()
    base = str()
    for i in a:
        i2 = str(i)
        i2 = i2.replace("(", "").replace(")", "").replace("'", "").replace(",", "")
        base += i2 + "\n"
    # print(base)
    return base


def gg(a):
    l = a.split("-")
    l2 = list()
    for i in l:
        k = list()
        for j in i.replace("\n", "").replace("X", "").split():
            k.append(float(j))
        l2.append(k)

    return l2


def add(l, boo):
    if boo:
        cur.execute(f"""INSERT INTO tabs(Field1, Field2, Field3, Field4, Field5, Field6, Field7, Field8, Field9, Field10) 
        VALUES('{l[0]}', '{l[1]}', '{l[2]}', '{l[3]}', '{l[4]}', '{l[5]}', '{l[6]}', '{l[7]}', '{l[8]}', '{l[9]}');""")
        db.commit()

    return l[:-1]


db = sqlite3.connect('feu.db', check_same_thread=False)
cur = db.cursor()

