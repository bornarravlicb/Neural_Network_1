import numpy as np
from tensorflow.keras.datasets import mnist
from model import DenseLayer, Activations, Losses, SGD

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X.reshape(-1, 28*28).T / 255.0
test_X = test_X.reshape(-1, 28*28).T / 255.0

num_classes = 10
train_Y = np.eye(num_classes)[train_y].T
test_Y = np.eye(num_classes)[test_y].T

input_dim = 784
hidden_dim = 128
output_dim = 10

layer1 = DenseLayer(np.random.randn(hidden_dim, input_dim) * 0.01,
                    np.zeros((hidden_dim, 1)),
                    optimizer=SGD(lr=0.1))
layer2 = DenseLayer(np.random.randn(output_dim, hidden_dim) * 0.01,
                    np.zeros((output_dim, 1)),
                    optimizer=SGD(lr=0.1))

epochs = 5
batch_size = 128
num_samples = train_X.shape[1]

for ep in range(epochs):
    perm = np.random.permutation(num_samples)
    X_shuf = train_X[:, perm]
    Y_shuf = train_Y[:, perm]

    epoch_loss = 0.0
    for i in range(0, num_samples, batch_size):
        Xb = X_shuf[:, i:i+batch_size]
        Yb = Y_shuf[:, i:i+batch_size]

        Z1 = layer1.forward(Xb)
        A1 = Activations.relu(Z1)

        Z2 = layer2.forward(A1)
        A2 = Z2

        loss = Losses.mse(A2, Yb)
        dA2 = Losses.dmse(A2, Yb)

        dZ1 = layer2.backward(dA2)
        dA1 = dZ1 * (A1 > 0)
        layer1.backward(dA1)

        epoch_loss += loss * Xb.shape[1]

    epoch_loss /= num_samples
    print(f"Epoch {ep+1}/{epochs}, Loss: {epoch_loss:.4f}")

Z1_test = layer1.forward(test_X)
A1_test = Activations.relu(Z1_test)
Z2_test = layer2.forward(A1_test)
pred_labels = np.argmax(Z2_test, axis=0)
accuracy = np.mean(pred_labels == test_y)
print(f"Test Accuracy: {accuracy:.4f}")
