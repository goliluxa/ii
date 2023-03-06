def gg():
    a = """1.21 X
0 X
5.06 X
2.73 X
10.65 X
8.21 X
9.31 X
9.82 X
1.7 X
    """

    l = a.split("-")
    l2 = list()
    for i in l:
        k = list()
        for j in i.replace("\n", "").replace("X", "").split():
            k.append(float(j))
        l2.append(k)

    return l2

print(*gg())
