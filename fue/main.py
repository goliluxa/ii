import numpy as np
from bd import add, get, gg
import pprint
import time
from threading import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def what(proc, t, l, base, boo=False):
    s = list()
    s1 = list()

    for i in base.split("\n"):
        k = list()
        f = True
        for j in i.split():
            h = 1 if float(j) > proc else 0
            if f:
                s1.append(h)
                f = False
            else:
                k.append(h)
        s.append(k)
    s = s[:-1]
    # print(s1)
    # pprint.pprint(s)


    training_inputs = np.array(s)

    training_outputs = np.array([s1]).T

    np.random.seed(1)

    synaptic_weights = 2 * np.random.random((9, 1)) - 1

    # print("Случайные инициализирующие веса: ")
    # print(synaptic_weights)

    for i in range(100000):
        # Метод обратного распространения
        input_laver = training_inputs
        outputs = sigmoid(np.dot(input_laver, synaptic_weights))

        err = training_outputs - outputs

        arguments = np.dot(input_laver.T, err * (outputs * (1 - outputs)))

        synaptic_weights += arguments

    # print("веса")
    # print(synaptic_weights)
    # print("Результат")
    # print(outputs)
    k = list()
    for i in add(l, boo):
        h = 1 if float(i) > proc else 0
        k.append(h)
    new_inputs = np.array(k)  # новая ситуация
    output = sigmoid(np.dot(new_inputs, synaptic_weights))

    print(f"То что выше {t}    ", output[0])

text = input()
a = ""
for i in range(9):
    a += text
    text = input()
t = time.time()
a += text
l = gg(a)[0]
base = get()
what(1.5, "1.5", l, base)
what(2, "2", l, base)
what(3.5, "3.5", l, base)
what(5, "5", l, base)
what(10, "10", l, base, True)
# t1 = Thread(target=what, args=(1.5, "1.5", l, base))
# t2 = Thread(target=what, args=(2, "2", l, base))
# t3 = Thread(target=what, args=(5, "5", l, base))
# t4 = Thread(target=what, args=(10, "10", l, base, True))
# t1.start()
# t2.start()
# t3.start()
# t4.start()
# t1.join()
# t2.join()
# t3.join()
# t4.join()
print(time.time() - t)
