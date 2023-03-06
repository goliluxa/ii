import pprint
import numpy as np
# from array import array
from vesa import *
from numpy import array



def read_bd():
    with open("bd.txt", "r") as f:
        a = f.read()
    # for i in a.split("\n")[:-1]:
    #     k = i.split()
    #     print(k[0], k[1])
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


def ney4(inp_i, goal_pred_i, n, vesa=list(), alpha=0.00003, ney=9):
    dub = 0
    alpha = alpha / (10 ** ney)
    lerror = 0
    if len(vesa) != 0:
        weight_1_2 = vesa[0]
        weight_2_3 = vesa[1]
        weight_3_4 = vesa[2]
        weight_4_5 = vesa[3]
        weight_5_6 = vesa[4]
        weight_6_7 = vesa[5]
        weight_7_8 = vesa[6]
        weight_8_9 = vesa[7]
        weight_9_10 = vesa[8]
    else:
        weight_1_2 = np.random.randn(9, 9)
        weight_2_3 = np.random.randn(9, 9)
        weight_3_4 = np.random.randn(9, 9)
        weight_4_5 = np.random.randn(9, 9)
        weight_5_6 = np.random.randn(9, 9)
        weight_6_7 = np.random.randn(9, 9)
        weight_7_8 = np.random.randn(9, 9)
        weight_8_9 = np.random.randn(9, 9)
        weight_9_10 = np.random.randn(9, 1)
    # print(weight_)
    # print(weight_1_2, weight_2_3, weight_3_4, weight_4_5, sep="\n\n")
    for iteration in range(n):
        for i in range(len(inp_i)):
            inp = inp_i[i]
            goal_pred = goal_pred_i[i]

            layer_2 = np.dot(inp, weight_1_2)
            layer_3 = np.dot(layer_2, weight_2_3)
            layer_4 = np.dot(layer_3, weight_3_4)
            layer_5 = np.dot(layer_4, weight_4_5)
            layer_6 = np.dot(layer_5, weight_5_6)
            layer_7 = np.dot(layer_6, weight_6_7)
            layer_8 = np.dot(layer_7, weight_7_8)
            layer_9 = np.dot(layer_8, weight_8_9)
            # print(layer_2)
            # exit()
            # if dub != 2:
            #     dub += 1
            # else:
            #     print(weight_1_2)
            #     exit()
            pred = np.sum(np.dot(layer_9, weight_9_10))
            # print(pred)
            # if iteration % 100 == 0:
            #     if pred <= 0.5:
            #         print("It is a - setosa", goal_pred, pred)
            #     elif pred > 0.7 and pred <= 1.5:
            #         print("It is a - versicolor", goal_pred, pred)
            #     elif pred > 1.7 and pred <= 2.5:
            #         print("It is a - virginica", goal_pred, pred)
            #     else:
            #         print("I dont know", goal_pred, pred)

            error = (pred - goal_pred) ** 2

            layer_10_delta = pred - goal_pred
            layer_9_delta = np.sum(np.dot(layer_10_delta, weight_9_10))
            layer_8_delta = np.sum(np.dot(layer_9_delta, weight_8_9))
            layer_7_delta = np.sum(np.dot(layer_8_delta, weight_7_8))
            layer_6_delta = np.sum(np.dot(layer_7_delta, weight_6_7))
            layer_5_delta = np.sum(np.dot(layer_6_delta, weight_5_6))
            layer_4_delta = np.sum(np.dot(layer_5_delta, weight_4_5))
            layer_3_delta = np.sum(np.dot(layer_4_delta, weight_3_4))
            layer_2_delta = np.sum(np.dot(layer_3_delta, weight_2_3))


            weight_delta_1_2 = np.zeros(weight_1_2.shape)
            weight_delta_2_3 = np.zeros(weight_2_3.shape)
            weight_delta_3_4 = np.zeros(weight_3_4.shape)
            weight_delta_4_5 = np.zeros(weight_4_5.shape)
            weight_delta_5_6 = np.zeros(weight_5_6.shape)
            weight_delta_6_7 = np.zeros(weight_6_7.shape)
            weight_delta_7_8 = np.zeros(weight_7_8.shape)
            weight_delta_8_9 = np.zeros(weight_8_9.shape)
            weight_delta_9_10 = np.zeros(weight_9_10.shape)

            # ----------------

            for k in range(len(weight_delta_1_2)):
                for j in range(len(weight_delta_1_2[k])):
                    weight_delta_1_2[k][j] = inp[k] * layer_2_delta
            # print(weight_delta_2_3)
            # exit()
            for k in range(len(weight_delta_2_3)):
                for j in range(len(weight_delta_2_3[k])):
                    weight_delta_2_3[k][j] = np.sum(layer_2.T[j]) * layer_3_delta

            for k in range(len(weight_delta_3_4)):
                for j in range(len(weight_delta_3_4[k])):
                    weight_delta_3_4[k][j] = np.sum(layer_3.T[j]) * layer_4_delta

            for k in range(len(weight_delta_4_5)):
                for j in range(len(weight_delta_4_5[k])):
                    weight_delta_4_5[k][j] = np.sum(layer_4.T[j]) * layer_5_delta

            for k in range(len(weight_delta_5_6)):
                for j in range(len(weight_delta_5_6[k])):
                    weight_delta_5_6[k][j] = np.sum(layer_5.T[j]) * layer_6_delta

            for k in range(len(weight_delta_6_7)):
                for j in range(len(weight_delta_6_7[k])):
                    weight_delta_6_7[k][j] = np.sum(layer_6.T[j]) * layer_7_delta

            for k in range(len(weight_delta_7_8)):
                for j in range(len(weight_delta_7_8[k])):
                    weight_delta_7_8[k][j] = np.sum(layer_7.T[j]) * layer_8_delta

            for k in range(len(weight_delta_8_9)):
                for j in range(len(weight_delta_8_9[k])):
                    weight_delta_8_9[k][j] = np.sum(layer_8.T[j]) * layer_9_delta

            # ----------------

            for k in range(len(weight_delta_9_10)):
                weight_delta_9_10[k] = np.sum(layer_9.T[k]) * layer_10_delta


            # ----------------
            # ----------------

            for k in range(len(weight_1_2)):
                for j in range(len(weight_1_2)):
                    weight_1_2[k][j] -= weight_delta_1_2[k][j] * alpha

            for k in range(len(weight_2_3)):
                for j in range(len(weight_2_3)):
                    weight_2_3[k][j] -= weight_delta_2_3[k][j] * alpha

            for k in range(len(weight_3_4)):
                for j in range(len(weight_3_4)):
                    weight_3_4[k][j] -= weight_delta_3_4[k][j] * alpha

            for k in range(len(weight_4_5)):
                for j in range(len(weight_4_5)):
                    weight_4_5[k][j] -= weight_delta_4_5[k][j] * alpha

            for k in range(len(weight_5_6)):
                for j in range(len(weight_5_6)):
                    weight_5_6[k][j] -= weight_delta_5_6[k][j] * alpha

            for k in range(len(weight_6_7)):
                for j in range(len(weight_6_7)):
                    weight_6_7[k][j] -= weight_delta_6_7[k][j] * alpha

            for k in range(len(weight_7_8)):
                for j in range(len(weight_7_8)):
                    weight_7_8[k][j] -= weight_delta_7_8[k][j] * alpha

            for k in range(len(weight_8_9)):
                for j in range(len(weight_8_9)):
                    weight_8_9[k][j] -= weight_delta_8_9[k][j] * alpha

            # ----------------

            for k in range(len(weight_9_10)):
                weight_9_10[k] -= weight_delta_9_10[k] * alpha
            # print(weight_delta_3_4)
            # exit()
            # ----------------

        print(iteration, error, sep=(" --- " if error < lerror else " +++ "))
        lerror = error
    return [weight_1_2, weight_2_3, weight_3_4, weight_4_5, weight_5_6, weight_6_7, weight_7_8, weight_8_9, weight_9_10]


s = [[int(i[0]), float(i[1])] for i in read_bd()]
s.sort()
# pprint.pprint(s)
bd = list()
a_bd = list()
for i in range(0, len(s) // 2 * 2):
    if s[i][0] - s[i - 2][0] == 2 and \
            s[i - 2][0] - s[i - 4][0] == 1 and \
            s[i - 4][0] - s[i - 6][0] == 2 and \
            s[i - 6][0] - s[i - 8][0] == 2 and \
            s[i - 8][0] - s[i - 10][0] == 2 and \
            s[i - 10][0] - s[i - 12][0] == 2 and \
            s[i - 12][0] - s[i - 14][0] == 2 and \
            s[i - 14][0] - s[i - 16][0] == 2 and \
            s[i - 16][0] - s[i - 18][0] == 2:
        a = False
        # print(i, i - 1)
        num, snum = rod(s[i][1], a), rod([s[i - 2][1], s[i - 4][1], s[i - 6][1], s[i - 8][1], s[i - 10][1], s[i - 12][1], s[i - 14][1], s[i - 16][1], s[i - 18][1]], a)
        bd.append(snum)
        a_bd.append(num)

# print(a_bd[0])
# print(bd[0])

def go_ney(ves, inp):
    weight_1_2, weight_2_3, weight_3_4, weight_4_5, weight_5_6, weight_6_7, weight_7_8, weight_8_9, weight_9_10 = ves[0], ves[1], ves[2], ves[3], ves[4], ves[5], ves[6], ves[7], ves[8]
    layer_2 = np.dot(inp, weight_1_2)
    layer_3 = np.dot(layer_2, weight_2_3)
    layer_4 = np.dot(layer_3, weight_3_4)
    layer_5 = np.dot(layer_4, weight_4_5)
    layer_6 = np.dot(layer_5, weight_5_6)
    layer_7 = np.dot(layer_6, weight_6_7)
    layer_8 = np.dot(layer_7, weight_7_8)
    layer_9 = np.dot(layer_8, weight_8_9)
    pred = np.sum(np.dot(layer_9, weight_9_10))
    return pred


# k = 200
# pred, grod = go_ney(ves1, bd[k]), a_bd[k]
# print(rod(pred, True), rod(grod, True))


# count = 0
# for i in range(len(bd)):
#     pred, grod = go_ney(ves3, bd[i]), a_bd[i]
#     if rod(pred, True) == rod(grod, True):
#         count += 1
#
# print(count / len(bd) * 100)


# print(a_bd[0])
# print(bd[0])
print(ney4(bd, a_bd, 100000, alpha=0.01, ney=9, vesa=ves9n9c_2))


# 0.09463996079259564
# 0.09492957135325965
