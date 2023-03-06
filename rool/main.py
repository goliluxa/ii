import pprint
import numpy as np
# from array import array
from vesa import *
from numpy import array


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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


def ney4(inp_i, goal_pred_i, n, vesa=list(), alpha=0.00003):
    # alpha = 0.00003
    if len(vesa) != 0:
        weight_1_2 = vesa[0]
        weight_2_3 = vesa[1]
        weight_3_4 = vesa[2]
        weight_4_5 = vesa[3]
    else:
        weight_1_2 = np.random.randn(9, 9)
        weight_2_3 = np.random.randn(9, 9)
        weight_3_4 = np.random.randn(9, 9)
        weight_4_5 = np.random.randn(9, 1)
    # print(weight_)
    # print(weight_1_2, weight_2_3, weight_3_4, weight_4_5, sep="\n\n")
    for iteration in range(n):
        for i in range(len(inp_i)):
            inp = inp_i[i]
            goal_pred = goal_pred_i[i]

            layer_2 = np.dot(inp, weight_1_2)
            layer_3 = np.dot(layer_2, weight_2_3)
            layer_4 = np.dot(layer_3, weight_3_4)
            pred = np.sum(np.dot(layer_4, weight_4_5))
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
            layer_5_delta = pred - goal_pred
            layer_4_delta = np.sum(np.dot(layer_5_delta, weight_4_5))
            layer_3_delta = np.sum(np.dot(layer_4_delta, weight_3_4))
            layer_2_delta = np.sum(np.dot(layer_3_delta, weight_2_3))

            weight_delta_1_2 = np.zeros(weight_1_2.shape)
            weight_delta_2_3 = np.zeros(weight_2_3.shape)
            weight_delta_3_4 = np.zeros(weight_3_4.shape)
            weight_delta_4_5 = np.zeros(weight_4_5.shape)

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
                weight_delta_4_5[k] = np.sum(layer_4.T[k]) * layer_5_delta

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
                weight_4_5[k] -= weight_delta_4_5[k] * alpha

        print(iteration, error, sep=" --- ")
    return [weight_1_2, weight_2_3, weight_3_4, weight_4_5]


s = [[int(i[0]), float(i[1])] for i in read_bd()]
s.sort()
# pprint.pprint(s)
bd = list()
a_bd = list()
for i in range(0, len(s) // 10 * 10, 10):
    for j in range(i + 1, i + 10):
        if s[j][0] - s[j - 1][0] != 1:
            # print("no")
            break
    else:
        a = False
        num, snum = rod(s[i][1], a), rod(s[i + 1:i + 10], a)
        bd.append(snum)
        a_bd.append(num)


def go_ney(ves, inp):
    weight_1_2, weight_2_3, weight_3_4, weight_4_5 = ves[0], ves[1], ves[2], ves[3]
    layer_2 = np.dot(inp, weight_1_2)
    layer_3 = np.dot(layer_2, weight_2_3)
    layer_4 = np.dot(layer_3, weight_3_4)
    pred = np.sum(np.dot(layer_4, weight_4_5))
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
print(ney4(bd, a_bd, 10000, ves4, 0.00001))

# [array([[ 1.42458475, -0.58591884,  0.09976141,  1.51161347,  2.03083792,
#          1.81847623, -0.14196092, -0.21924071,  1.01290992],
#        [ 0.12713396,  0.1190175 ,  1.81088146,  2.09515913,  0.95191518,
#         -0.10246798,  0.09331078,  0.40901015,  0.34199515],
#        [ 1.01036133, -0.65066076, -0.05940272,  3.68918553,  0.92294271,
#         -0.9788166 , -0.89631881, -1.12909092,  0.66002037],
#        [-1.41651976, -0.29463732,  0.02634653, -2.26094236, -2.61322832,
#         -1.04114348, -2.56359198, -2.90456018, -0.99592192],
#        [ 1.91700444,  0.64339126,  0.95086256,  1.86268438,  2.62932978,
#          1.54257781,  1.32911203,  2.12221097,  1.71762167],
#        [-0.92433469,  0.45662601,  0.45802014, -0.14087157, -0.66539504,
#          0.34154109, -0.70941026, -0.74629693, -0.65240155],
#        [ 0.1740639 ,  0.59373198, -0.13742791, -0.10913786,  1.66942082,
#         -2.85196996, -1.13377573,  0.75802066, -2.24025317],
#        [ 0.87124154, -0.09705697,  0.30859783, -0.48525551, -0.93833149,
#         -0.08733127, -2.86876032, -1.46709872, -1.26984087],
#        [-1.83087714,  0.82737275, -1.1750302 , -0.93200979, -1.40528134,
#         -0.8999564 , -0.46985052, -1.24563367, -0.71172742]]), array([[ 0.10496961, -2.39551855, -0.6154065 ,  0.08533597,  2.20178843,
#          0.61389661,  0.50261739,  1.72616107,  0.3119018 ],
#        [-0.05113319, -0.59303436, -0.35877532, -0.92946505,  1.36076053,
#         -0.50337254,  0.31118011,  1.76363596, -0.55800502],
#        [-1.13700667, -1.77633756, -0.88703163,  1.11375929,  0.51258845,
#          0.27536585,  1.28389416, -2.12965461,  1.06628903],
#        [ 1.38760237, -1.91586126, -1.03073046,  0.20377425,  1.3200953 ,
#          1.06901408,  0.13826922,  2.59829215,  0.08325974],
#        [-0.10694834, -3.32098199, -0.90141141,  1.35811628,  0.50719054,
#         -1.14807559,  1.35759578,  0.14131858,  0.08517415],
#        [-1.60904769, -0.56884858, -1.37201081, -0.5656238 , -1.43557289,
#          2.02678028,  1.45460686, -0.01497641,  1.29506987],
#        [ 0.28985702,  0.12488409, -0.80248135, -0.5586918 ,  0.52969201,
#         -2.19573987,  1.53942036,  1.16216084,  1.5473746 ],
#        [ 2.16532176, -2.51650444, -2.35318534,  0.92014355,  0.53302144,
#          0.43230933, -0.24314595, -0.42656871,  1.05742809],
#        [ 0.91899575, -1.85249952, -1.394296  , -0.89623417,  1.26043681,
#          1.76191747, -1.08742435,  1.12534517, -0.26945662]]), array([[-1.50722767,  0.05950315, -0.86763994, -3.11685555, -2.50337395,
#         -0.10964682,  0.35859482,  0.22492311,  1.25064157],
#        [ 1.47952466,  1.30317734, -1.16482398, -0.66572951,  0.10431976,
#         -1.29099544, -1.01004488,  1.64435574, -0.27329744],
#        [ 1.34257523, -0.7330269 , -1.61816173, -0.4655065 , -0.19469397,
#         -0.95081695,  1.19187383,  1.04309373,  0.53421575],
#        [ 1.48580442,  0.66309705, -1.65334629,  1.00388478,  0.57857857,
#          1.60942369, -2.94346319,  1.80620381,  2.28409746],
#        [ 0.26233738,  0.99326288, -0.4370177 , -2.13982685, -0.08765243,
#         -0.12372558,  0.60889608,  1.31839318,  1.33882861],
#        [ 2.14245348,  0.67788754, -1.3568991 , -1.01058651, -1.46014832,
#         -0.37697152, -0.82738634, -0.12388531,  1.35111636],
#        [ 2.17325365, -2.32375901, -1.19422821, -1.75036675, -0.11990323,
#         -0.95128562, -1.739285  ,  1.0496489 ,  0.39198551],
#        [ 1.88091099,  1.17545793,  1.21934788, -0.88146843,  0.32905332,
#         -1.19243229,  0.63766804, -1.35589407,  1.2656365 ],
#        [ 2.13800406,  1.35109552,  0.11320667, -1.05263778,  0.55545974,
#          0.39684915,  0.48980465, -1.29638186,  1.23472198]]), array([[ 1.28942684],
#        [ 0.17004614],
#        [ 0.79264143],
#        [-0.49656504],
#        [ 0.81843213],
#        [ 1.40735868],
#        [ 1.35618226],
#        [ 1.21658725],
#        [-0.41115517]])]
