import numpy as np
from list import base_l
from main import gg
import pprint


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


s = list()
s1 = list()

for i in base_l.split("\n"):
    k = list()
    f = True
    for j in i.split():
        if f:
            s1.append(round(float(j) / 10, 3))
            f = False
        else:
            k.append(round(float(j) / 10, 3))
    s.append(k)
print(s1)
pprint.pprint(s)


training_inputs = np.array(s)

training_outputs = np.array([s1]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((9, 1)) - 1

# print("Случайные инициализирующие веса: ")
# print(synaptic_weights)

for i in range(20000):
    # Метод обратного распространения
    input_laver = training_inputs
    outputs = sigmoid(np.dot(input_laver, synaptic_weights))

    err = training_outputs - outputs

    arguments = np.dot(input_laver.T, err * (outputs * (1 - outputs)))

    synaptic_weights += arguments

print("веса")
print(synaptic_weights)
print("Результат")
print(outputs)

for i in gg():
    k = list()
    for j in i:
        k.append(float(j) / 10)
    s.append(k)
new_inputs = np.array(k)  # новая ситуация
output = sigmoid(np.dot(new_inputs, synaptic_weights))

print("Новая ситуация:")
print(output)
