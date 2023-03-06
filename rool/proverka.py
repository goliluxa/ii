import numpy as np
from numpy import array
from vesa import *


def go_ney(ves, inp):
    weight_1_2, weight_2_3, weight_3_4, weight_4_5 = ves[0], ves[1], ves[2], ves[3]
    layer_2 = np.dot(inp, weight_1_2)
    layer_3 = np.dot(layer_2, weight_2_3)
    layer_4 = np.dot(layer_3, weight_3_4)
    pred = np.sum(np.dot(layer_4, weight_4_5))
    return pred

def read_bd():
    with open("bd.txt", "r") as f:
        a = f.read()

    return [i.split() for i in a.split("\n")[:-1]]

def rod(n, flag=False):
    try:
        a = n[0][1]
        s = [rod(i[1], flag) for i in n]
        return s
    except:
        if flag:
            n = n * 15
            n = str(n).split(".")[0]
            return float(n)
        else:
            return n

#
# s = [[int(i[0]), float(i[1])] for i in read_bd()]
# s.sort()
# # pprint.pprint(s)
# bd = list()
# a_bd = list()
# for i in range(0, len(s) // 10 * 10, 10):
#     for j in range(i + 1, i + 10):
#         if s[j][0] - s[j - 1][0] != 1:
#             # print("no")
#             break
#     else:
#         a = False
#         num, snum = rod(s[i][1], a), rod(s[i + 1:i + 10], a)
#         bd.append(snum)
#         a_bd.append(num)


s = [[int(i[0]), float(i[1])] for i in read_bd()]
s.sort()
# pprint.pprint(s)
bd = list()
a_bd = list()
for i in range(0, len(s) // 2 * 2, 2):
    if s[i - 1][0] - s[i - 2][0] == 1 and s[i][0] - s[i - 1][0] == 1:
        a = False
        # print(i, i - 1)
        num, snum = rod(s[i][1], a), rod([s[i - 1][1], s[i - 2][1]], a)
        bd.append(snum)
        a_bd.append(num)


count = 0
for i in range(len(bd)):
    pred, grod = go_ney(ves4, bd[i]), a_bd[i]
    p = False
    g = False
    if rod(pred, True) in [x for x in range(1, 8)]:
        p = True
    elif rod(pred, True) in [x for x in range(8, 15)]:
        p = False
    else:
        p = None
    if rod(grod, True) in [x for x in range(1, 8)]:
        g = True
    elif rod(grod, True) in [x for x in range(8, 15)]:
        g = False
    else:
        g = None


    # print(pred * 15, grod * 15)
    if p == g:
        count += 1

print(count / len(bd) * 100)

