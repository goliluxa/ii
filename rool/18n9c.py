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


def ney4(inp_i, goal_pred_i, n, vesa=list(), alpha=0.00003):
    # alpha = 0.00003
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
        weight_10_11 = vesa[9]
        weight_11_12 = vesa[10]
        weight_12_13 = vesa[11]
        weight_13_14 = vesa[12]
        weight_14_15 = vesa[13]
        weight_15_16 = vesa[14]
        weight_16_17 = vesa[15]
        weight_17_18 = vesa[16]
        weight_18_19 = vesa[17]

    else:
        weight_1_2 = np.random.randn(9, 9)
        weight_2_3 = np.random.randn(9, 9)
        weight_3_4 = np.random.randn(9, 9)
        weight_4_5 = np.random.randn(9, 9)
        weight_5_6 = np.random.randn(9, 9)
        weight_6_7 = np.random.randn(9, 9)
        weight_7_8 = np.random.randn(9, 9)
        weight_8_9 = np.random.randn(9, 9)
        weight_9_10 = np.random.randn(9, 9)
        weight_10_11 = np.random.randn(9, 9)
        weight_11_12 = np.random.randn(9, 9)
        weight_12_13 = np.random.randn(9, 9)
        weight_13_14 = np.random.randn(9, 9)
        weight_14_15 = np.random.randn(9, 9)
        weight_15_16 = np.random.randn(9, 9)
        weight_16_17 = np.random.randn(9, 9)
        weight_17_18 = np.random.randn(9, 9)
        weight_18_19 = np.random.randn(9, 1)
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
            layer_10 = np.dot(layer_9, weight_9_10)
            layer_11 = np.dot(layer_10, weight_10_11)
            layer_12 = np.dot(layer_11, weight_11_12)
            layer_13 = np.dot(layer_12, weight_12_13)
            layer_14 = np.dot(layer_13, weight_13_14)
            layer_15 = np.dot(layer_14, weight_14_15)
            layer_16 = np.dot(layer_15, weight_15_16)
            layer_17 = np.dot(layer_16, weight_16_17)
            layer_18 = np.dot(layer_17, weight_17_18)
            pred = np.sum(np.dot(layer_18, weight_18_19))
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
            layer_19_delta = pred - goal_pred
            layer_18_delta = np.sum(np.dot(layer_19_delta, weight_18_19))
            layer_17_delta = np.sum(np.dot(layer_18_delta, weight_17_18))
            layer_16_delta = np.sum(np.dot(layer_17_delta, weight_16_17))
            layer_15_delta = np.sum(np.dot(layer_16_delta, weight_15_16))
            layer_14_delta = np.sum(np.dot(layer_15_delta, weight_14_15))
            layer_13_delta = np.sum(np.dot(layer_14_delta, weight_13_14))
            layer_12_delta = np.sum(np.dot(layer_13_delta, weight_12_13))
            layer_11_delta = np.sum(np.dot(layer_12_delta, weight_11_12))
            layer_10_delta = np.sum(np.dot(layer_11_delta, weight_10_11))
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
            weight_delta_10_11 = np.zeros(weight_10_11.shape)
            weight_delta_11_12 = np.zeros(weight_11_12.shape)
            weight_delta_12_13 = np.zeros(weight_12_13.shape)
            weight_delta_13_14 = np.zeros(weight_13_14.shape)
            weight_delta_14_15 = np.zeros(weight_14_15.shape)
            weight_delta_15_16 = np.zeros(weight_15_16.shape)
            weight_delta_16_17 = np.zeros(weight_16_17.shape)
            weight_delta_17_18 = np.zeros(weight_17_18.shape)
            weight_delta_18_19 = np.zeros(weight_18_19.shape)

            # ----------------

            for k in range(len(weight_delta_1_2)):
                for j in range(len(weight_delta_1_2[k])):
                    weight_delta_1_2[k][j] = inp[k] * layer_2_delta

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

            for k in range(len(weight_delta_9_10)):
                for j in range(len(weight_delta_9_10[k])):
                    weight_delta_9_10[k][j] = np.sum(layer_9.T[j]) * layer_10_delta

            for k in range(len(weight_delta_10_11)):
                for j in range(len(weight_delta_10_11[k])):
                    weight_delta_10_11[k][j] = np.sum(layer_10.T[j]) * layer_11_delta

            for k in range(len(weight_delta_11_12)):
                for j in range(len(weight_delta_11_12[k])):
                    weight_delta_11_12[k][j] = np.sum(layer_11.T[j]) * layer_12_delta

            for k in range(len(weight_delta_12_13)):
                for j in range(len(weight_delta_12_13[k])):
                    weight_delta_12_13[k][j] = np.sum(layer_12.T[j]) * layer_13_delta

            for k in range(len(weight_delta_13_14)):
                for j in range(len(weight_delta_13_14[k])):
                    weight_delta_13_14[k][j] = np.sum(layer_13.T[j]) * layer_14_delta

            for k in range(len(weight_delta_14_15)):
                for j in range(len(weight_delta_14_15[k])):
                    weight_delta_14_15[k][j] = np.sum(layer_14.T[j]) * layer_15_delta

            for k in range(len(weight_delta_15_16)):
                for j in range(len(weight_delta_15_16[k])):
                    weight_delta_15_16[k][j] = np.sum(layer_15.T[j]) * layer_16_delta

            for k in range(len(weight_delta_16_17)):
                for j in range(len(weight_delta_16_17[k])):
                    weight_delta_16_17[k][j] = np.sum(layer_16.T[j]) * layer_17_delta

            for k in range(len(weight_delta_17_18)):
                for j in range(len(weight_delta_17_18[k])):
                    weight_delta_17_18[k][j] = np.sum(layer_17.T[j]) * layer_18_delta

            # ----------------

            for k in range(len(weight_delta_18_19)):
                weight_delta_18_19[k] = np.sum(layer_18.T[k]) * layer_19_delta

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

            for k in range(len(weight_9_10)):
                for j in range(len(weight_9_10)):
                    weight_9_10[k][j] -= weight_delta_9_10[k][j] * alpha

            for k in range(len(weight_10_11)):
                for j in range(len(weight_10_11)):
                    weight_10_11[k][j] -= weight_delta_10_11[k][j] * alpha

            for k in range(len(weight_11_12)):
                for j in range(len(weight_11_12)):
                    weight_11_12[k][j] -= weight_delta_11_12[k][j] * alpha

            for k in range(len(weight_12_13)):
                for j in range(len(weight_12_13)):
                    weight_12_13[k][j] -= weight_delta_12_13[k][j] * alpha

            for k in range(len(weight_13_14)):
                for j in range(len(weight_13_14)):
                    weight_13_14[k][j] -= weight_delta_13_14[k][j] * alpha

            for k in range(len(weight_14_15)):
                for j in range(len(weight_14_15)):
                    weight_14_15[k][j] -= weight_delta_14_15[k][j] * alpha

            for k in range(len(weight_15_16)):
                for j in range(len(weight_15_16)):
                    weight_15_16[k][j] -= weight_delta_15_16[k][j] * alpha

            for k in range(len(weight_16_17)):
                for j in range(len(weight_16_17)):
                    weight_16_17[k][j] -= weight_delta_16_17[k][j] * alpha

            for k in range(len(weight_17_18)):
                for j in range(len(weight_17_18)):
                    weight_17_18[k][j] -= weight_delta_17_18[k][j] * alpha


            # ----------------

            for k in range(len(weight_18_19)):
                weight_18_19[k] -= weight_delta_18_19[k] * alpha

            # ----------------

        print(iteration, error, sep=(" --- " if error < lerror else " +++ "))
        lerror = error
    return [weight_1_2, weight_2_3, weight_3_4, weight_4_5, weight_5_6, weight_6_7, weight_7_8, weight_8_9, weight_9_10,
            weight_10_11, weight_11_12, weight_12_13, weight_13_14, weight_14_15, weight_15_16, weight_16_17, weight_17_18, weight_18_19]


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
    weight_10_11, weight_11_12, weight_12_13, weight_13_14, weight_14_15, weight_15_16, weight_16_17, weight_17_18, weight_18_19 = ves[9], ves[10], ves[11], ves[12], ves[13], ves[14], ves[15], ves[16], ves[17]
    layer_2 = np.dot(inp, weight_1_2)
    layer_3 = np.dot(layer_2, weight_2_3)
    layer_4 = np.dot(layer_3, weight_3_4)
    layer_5 = np.dot(layer_4, weight_4_5)
    layer_6 = np.dot(layer_5, weight_5_6)
    layer_7 = np.dot(layer_6, weight_6_7)
    layer_8 = np.dot(layer_7, weight_7_8)
    layer_9 = np.dot(layer_8, weight_8_9)
    layer_10 = np.dot(layer_9, weight_9_10)
    layer_11 = np.dot(layer_10, weight_10_11)
    layer_12 = np.dot(layer_11, weight_11_12)
    layer_13 = np.dot(layer_12, weight_12_13)
    layer_14 = np.dot(layer_13, weight_13_14)
    layer_15 = np.dot(layer_14, weight_14_15)
    layer_16 = np.dot(layer_15, weight_15_16)
    layer_17 = np.dot(layer_16, weight_16_17)
    layer_18 = np.dot(layer_17, weight_17_18)
    pred = np.sum(np.dot(layer_18, weight_18_19))
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
print(ney4(bd, a_bd, 50000, ves18n9c_2, alpha=0.0000000000000000003))


