import numpy as np
from list import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


training_inputs = np.array([[0, 1, 1],
                            [0, 0, 1],
                            [1, 0, 1],
                            [0, 0, 1]])

training_outputs = np.array([[1,
                              1,
                              0,
                              1]]).T

# np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1

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
