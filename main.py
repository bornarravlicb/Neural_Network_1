import numpy as np
from model import DenseLayer, Activations
from julia import Main
Main.include("julia/ops.jl")

X = np.random.randn(5, 100)
Y = np.random.randn(3, 100)


w1 = np.random.randn(3, 5)
b1 = np.zeros((3, 1))
layer1 = DenseLayer(w1, b1)

epochs = 100
lr = 0.01

for epoch in range(epochs):
    Y_pred = layer1.forward(X)

    loss = Main.Losses.mse(Y_pred, Y)

    dY = Main.Losses.dmse(Y_pred, Y)

    dx = layer1.backward(dY)

    layer1.w -= lr * (dx @ X.T) / X.shape[1]
    layer1.b -= lr * np.sum(dY, axis=1, keepdims=True)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

X_test = np.random.randn(5, 10)
Y_test_pred = layer1.forward(X_test)
print("Test outputs:", Y_test_pred)
