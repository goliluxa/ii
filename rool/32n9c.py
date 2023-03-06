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


def ney4(inp_i, goal_pred_i, lifes, vesa=list(), alpha=0.00003, neyrons=4):
    # alpha = 0.00003
    lerror = 0
    weights_ = [""]
    if len(vesa) == neyrons:
        for i in vesa:
            weights_.append(i)
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
        for i in range(neyrons - 1):
            weights_.append(np.random.randn(9, 9))
        weights_.append(np.random.randn(9, 1))

    for iteration in range(n):
        for i in range(len(inp_i)):
            inp = inp_i[i]
            goal_pred = goal_pred_i[i]

            layers = ["", "", np.dot(inp, weights_[0])]
            for i in range(2, neyrons + 3):
                layers.append(np.dot(layers[i], weights_[i]))

            layer_2 = np.dot(inp, weight_1_2)
            layer_3 = np.dot(layer_2, weight_2_3)
            layer_4 = np.dot(layer_3, weight_3_4)
            layer_5 = np.dot(layer_4, weight_4_5)
            layer_6 = np.dot(layer_5, weight_5_6)
            layer_7 = np.dot(layer_6, weight_6_7)
            layer_8 = np.dot(layer_7, weight_7_8)
            layer_9 = np.dot(layer_8, weight_8_9)

            pred2 = np.sum(np.dot(layer_9, weight_9_10))
            pred = np.sum(np.dot(layers[neyrons], weights_[neyrons]))
            print(pred, pred2)
            print(layers[9], layer_9)


            error = (pred - goal_pred) ** 2

            layers_delta = [pred - goal_pred]
            for i in range(neyrons - 1):
                layers_delta.append(np.sum(np.dot(layers_delta[i], weights_[neyrons - i])))

            layers_delta = layers_delta[::-1]
            layers_delta.append("")
            layers_delta.append("")

            layer_10_delta = pred - goal_pred
            layer_9_delta = np.sum(np.dot(layer_10_delta, weight_9_10))
            layer_8_delta = np.sum(np.dot(layer_9_delta, weight_8_9))
            layer_7_delta = np.sum(np.dot(layer_8_delta, weight_7_8))
            layer_6_delta = np.sum(np.dot(layer_7_delta, weight_6_7))
            layer_5_delta = np.sum(np.dot(layer_6_delta, weight_5_6))
            layer_4_delta = np.sum(np.dot(layer_5_delta, weight_4_5))
            layer_3_delta = np.sum(np.dot(layer_4_delta, weight_3_4))
            layer_2_delta = np.sum(np.dot(layer_3_delta, weight_2_3))

            print(layers_delta[2], layer_2_delta)

            weights_delta = [""]
            for i in range(1, neyrons + 1):
                weights_delta.append(np.zeros(weights_[i].shape))

            weight_delta_1_2 = np.zeros(weight_1_2.shape)
            weight_delta_2_3 = np.zeros(weight_2_3.shape)
            weight_delta_3_4 = np.zeros(weight_3_4.shape)
            weight_delta_4_5 = np.zeros(weight_4_5.shape)
            weight_delta_5_6 = np.zeros(weight_5_6.shape)
            weight_delta_6_7 = np.zeros(weight_6_7.shape)
            weight_delta_7_8 = np.zeros(weight_7_8.shape)
            weight_delta_8_9 = np.zeros(weight_8_9.shape)
            weight_delta_9_10 = np.zeros(weight_9_10.shape)

            print(weights_delta[9], weight_delta_9_10)
            # exit()
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



print(ney4(bd, a_bd, n=100000, vesa=ves9n9c_1, alpha=0.00000000003, neyrons=9))
