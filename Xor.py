import numpy as np
import math
import matplotlib.pyplot as plt

# Initialisation

#Output is 2*1
Y = np.zeros((2, 1), dtype=float)

#Number of neurons in layer1
n1 = 10
#Number of neurons in layer2
n2 = 5
a = 1
epoc = 100000

# initializing the weights and biases

W1 = (np.random.rand(n1, 3) - 0.5 * np.ones((n1, 3)))
B1 = a * (np.random.rand(n1, 1) - 0.5 * np.ones((n1, 1)))
W2 = a * (np.random.rand(n2, n1) - 0.5 * np.ones((n2, n1)))
B2 = a * (np.random.rand(n2, 1) - 0.5 * np.ones((n2, 1)))
W3 = a * (np.random.rand(2, n2) - 0.5 * np.ones((2, n2)))
B3 = a * (np.random.rand(2, 1) - 0.5 * np.ones((2, 1)))
E = np.zeros(epoc)

Lr = 0.06

#   PROCESS DATA
X = np.array([[0, 1, 0, 1, 0, 1, 0, 1],
              [0, 0, 1, 1, 0, 0, 1, 1],
              [0, 0, 0, 0, 1, 1, 1, 1]])
# Why 2 lines for the output: Why not each column in X correspend to each column to Yd
Yd = np.array([[0, 1, 1, 0, 1, 0, 0, 1],
               [1, 0, 0, 1, 0, 1, 1, 0]]);

def sig(x):
    y = 1 / (1 + np.exp(-x))
    return y


def dsig(x):
    y = sig(x) * (1 - sig(x))
    return y


# START THE LEARNING PROCESS



for j in range(epoc):
    error = 0
    #8 correspends to the examples
    for i in range(8):
        #  ANN Feedforward
        x = X[:, i].reshape(3, 1)
        yd = Yd[:, i].reshape(2, 1)

        #Layer 1
        Y1 = sig(np.dot(W1, x) + B1)   
        #Layer 2
        Y2 = sig(np.dot(W2, Y1) + B2)
        #Final Layer   
        Y = (np.dot(W3, Y2) + B3)

        #   BackPropagation:    Weights and biases update
        #            T            T
        #    DB is dE/dB    DW= dE/dW

        DB3 = (Y - yd)  # *dsig(np.dot(W3,Y2)+B3)
        DW3 = np.dot(DB3, np.transpose(Y2))
        DB2 = (np.dot(np.transpose(W3), DB3)) * dsig(np.dot(W2, Y1) + B2)
        DW2 = np.dot(DB2, np.transpose(Y1))
        DB1 = (np.dot(np.transpose(W2), DB2)) * dsig(np.dot(W1, x) + B1)
        DW1 = np.dot(DB1, np.transpose(x))

        B1 = B1 - Lr * DB1
        W1 = W1 - Lr * DW1
        B2 = B2 - Lr * DB2
        W2 = W2 - Lr * DW2
        B3 = B3 - Lr * DB3
        W3 = W3 - Lr * DW3

        error = error + np.dot(np.transpose(Y - yd), (Y - yd))

    E[j] = error

plt.plot(E)
plt.show()