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


def ney4(inp_i, goal_pred_i, lifes, alpha, vesa=list(), neyrons=4):
    dub = 0
    alpha = alpha / (10 ** neyrons)
    lerror = 0
    weights_ = [""]
    # print(len(vesa), neyrons)
    if len(vesa) == neyrons:
        for i in vesa:
            weights_.append(i)
    else:
        for i in range(neyrons - 1):
            weights_.append(np.random.randn(9, 9))
        weights_.append(np.random.randn(9, 1))

    for iteration in range(lifes):
        try:
            for i in range(len(inp_i)):
                inp = inp_i[i]
                goal_pred = goal_pred_i[i]

                layers = ["", ""]
                layers.append(np.dot(inp, weights_[1]))
                for w in range(3, neyrons + 1):
                    layers.append(np.dot(layers[w - 1], weights_[w - 1]))

                # if dub != 2:
                #     dub += 1
                # else:
                #     print(weights_[1])
                #     exit()

                pred = np.sum(np.dot(layers[neyrons], weights_[neyrons]))
                # print(pred)
                error = (pred - goal_pred) ** 2
                # print(error)
                # exit()
                layers_delta = [pred - goal_pred]
                for w in range(neyrons - 1):
                    # print(neyrons + 1 - w)

                    layers_delta.append(np.sum(np.dot(layers_delta[w], weights_[neyrons - w])))

                layers_delta.append("")
                layers_delta.append("")
                layers_delta = layers_delta[::-1]

                weights_delta = [""]
                for w in range(1, neyrons + 1):
                    weights_delta.append(np.zeros(weights_[w].shape))

                # ----------------
                for k in range(len(weights_delta[1])):
                    for j in range(len(weights_delta[1][k])):
                        weights_delta[1][k][j] = inp[k] * layers_delta[2]

                for w in range(2, neyrons):
                    for k in range(len(weights_delta[w])):
                        for j in range(len(weights_delta[w][k])):
                            weights_delta[w][k][j] = np.sum(layers[w].T[j]) * layers_delta[w + 1]

                # ----------------
                for k in range(len(weights_delta[neyrons])):
                    weights_delta[neyrons][k] = np.sum(layers[neyrons].T[k]) * layers_delta[neyrons + 1]

                # ----------------
                # ----------------
                for w in range(1, neyrons):
                    for k in range(len(weights_[w])):
                        for j in range(len(weights_[w])):
                            weights_[w][k][j] -= weights_delta[w][k][j] * alpha
                # ----------------
                for k in range(len(weights_[neyrons])):
                    weights_[neyrons][k] -= weights_delta[neyrons][k] * alpha
                # ----------------
                # print(weights_delta[3])
                # exit()
                if dub % 2 == 0:
                    l1weights_ = weights_
                else:
                    l2weights_ = weights_
                dub += 1
        except:
            print(iteration, error, sep=(" --- " if error < lerror else " +++ "))
            return weights_
        if str(error) != "nan":
            print(iteration, error, sep=(" --- " if error < lerror else " +++ "))
            lerror = error
            # lweights_ = weights_
        else:
            return l1weights_, l2weights_
    return weights_


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
        num, snum = rod(s[i][1], a), rod(
            [s[i - 2][1], s[i - 4][1], s[i - 6][1], s[i - 8][1], s[i - 10][1], s[i - 12][1], s[i - 14][1], s[i - 16][1],
             s[i - 18][1]], a)
        bd.append(snum)
        a_bd.append(num)


def go_ney(vesa, inp, neyrons):
    weight_1_2, weight_2_3, weight_3_4, weight_4_5, weight_5_6, weight_6_7, weight_7_8, weight_8_9, weight_9_10 = vesa[
                                                                                                                      0], \
                                                                                                                  vesa[
                                                                                                                      1], \
                                                                                                                  vesa[
                                                                                                                      2], \
                                                                                                                  vesa[
                                                                                                                      3], \
                                                                                                                  vesa[
                                                                                                                      4], \
                                                                                                                  vesa[
                                                                                                                      5], \
                                                                                                                  vesa[
                                                                                                                      6], \
                                                                                                                  vesa[
                                                                                                                      7], \
                                                                                                                  vesa[
                                                                                                                      8]

    weights_ = [""]
    if len(vesa) == neyrons:
        for i in vesa:
            weights_.append(i)

    layers = ["", "", np.dot(inp, weights_[0])]
    for i in range(3, neyrons + 3):
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
    return pred


print(ney4(bd, a_bd, lifes=10000, alpha=0.0001, neyrons=90, vesa=ves90n9c_2))
