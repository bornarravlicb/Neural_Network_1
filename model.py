import numpy as np
from model import DenseLayer
from julia import Main

from tensorflow.keras.datasets import mnist
(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = train_X.reshape(-1, 28*28).T / 255.0
test_X = test_X.reshape(-1, 28*28).T / 255.0

num_classes = 10
train_Y = np.eye(num_classes)[train_y].T
test_Y = np.eye(num_classes)[test_y].T

input_dim = 28*28
output_dim = num_classes
w = np.random.randn(output_dim, input_dim) * 0.01
b = np.zeros((output_dim, 1))

lr = 0.1
layer = DenseLayer(w, b)

epochs = 5
batch_size = 64
num_samples = train_X.shape[1]

for ep in range(epochs):
    perm = np.random.permutation(num_samples)
    X_shuf = train_X[:, perm]
    Y_shuf = train_Y[:, perm]

    epoch_loss = 0.0
    for i in range(0, num_samples, batch_size):
        Xb = X_shuf[:, i:i+batch_size]
        Yb = Y_shuf[:, i:i+batch_size]

        preds = layer.forward(Xb)

        loss = Main.Losses.mse(preds, Yb)
        epoch_loss += loss * Xb.shape[1]

        dloss = Main.Losses.dmse(preds, Yb)

        layer.backward(dloss)

        dw = dloss @ Xb.T / Xb.shape[1]
        db = np.sum(dloss, axis=1, keepdims=True) / Xb.shape[1]
        layer.w -= lr * dw
        layer.b -= lr * db

    epoch_loss /= num_samples
    print(f"Epoch {ep+1}/{epochs}, Loss: {epoch_loss}")

test_preds = layer.forward(test_X)
pred_labels = np.argmax(test_preds, axis=0)
accuracy = np.mean(pred_labels == test_y)
print("Test Accuracy:", accuracy)
