import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# ------------------------------------------------------------
# 1. Load the MNIST dataset
# ------------------------------------------------------------
# MNIST has 60,000 training images and 10,000 test images.
# Each image is 28x28 = 784 pixels, labeled 0–9.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten 28x28 images into vectors of size 784
x_train = x_train.reshape(x_train.shape[0], 784).astype(np.float32)
x_test  = x_test.reshape(x_test.shape[0], 784).astype(np.float32)

# Normalize pixel values to [0, 1]
x_train /= 255.0
x_test  /= 255.0

# Convert labels to one-hot vectors
y_train = to_categorical(y_train, 10)  # shape: (60000, 10)
y_test  = to_categorical(y_test, 10)   # shape: (10000, 10)

# ------------------------------------------------------------
# 2. Hyperparameters
# ------------------------------------------------------------
input_size = 784   # 28x28
hidden_size = 128
output_size = 10   # digits 0–9
learning_rate = 0.01
epochs = 3
batch_size = 128

# For simplicity, let's train on the full set in batches.
# This might take a while in pure NumPy; consider smaller subsets or fewer epochs.

# ------------------------------------------------------------
# 3. Initialize weights
# ------------------------------------------------------------
# We'll use small random numbers for weights, and zeros for biases.
np.random.seed(42)
W1 = 0.01 * np.random.randn(input_size, hidden_size).astype(np.float32)
b1 = np.zeros((1, hidden_size), dtype=np.float32)

W2 = 0.01 * np.random.randn(hidden_size, output_size).astype(np.float32)
b2 = np.zeros((1, output_size), dtype=np.float32)

# ------------------------------------------------------------
# 4. Define activation functions
# ------------------------------------------------------------
def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    # derivative of ReLU: 1 if z>0 else 0
    return (z > 0).astype(z.dtype)

def softmax(z):
    # numerically stable softmax
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# ------------------------------------------------------------
# 5. Define loss (cross-entropy) and its derivative
# ------------------------------------------------------------
def cross_entropy_loss(pred, true):
    # pred: softmax predictions (N, 10)
    # true: one-hot labels (N, 10)
    eps = 1e-15
    return -np.mean(np.sum(true * np.log(pred + eps), axis=1))

def cross_entropy_deriv(pred, true):
    # derivative w.r.t. softmax input
    # pred - true, then average over batch
    return (pred - true) / pred.shape[0]

# ------------------------------------------------------------
# 6. Training (mini-batch gradient descent)
# ------------------------------------------------------------
num_samples = x_train.shape[0]
num_batches = num_samples // batch_size

for epoch in range(epochs):
    # Shuffle the data at the start of each epoch
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    for i in range(num_batches):
        # Get batch
        start = i * batch_size
        end   = start + batch_size
        x_batch = x_train[start:end]
        y_batch = y_train[start:end]

        # ---------- Forward pass ----------
        # 1) Hidden layer
        z1 = x_batch.dot(W1) + b1      # shape: (batch_size, 128)
        a1 = relu(z1)                  # ReLU

        # 2) Output layer
        z2 = a1.dot(W2) + b2          # shape: (batch_size, 10)
        a2 = softmax(z2)              # Softmax probabilities

        # ---------- Compute loss ----------
        loss = cross_entropy_loss(a2, y_batch)

        # ---------- Backward pass ----------
        # derivative of cross-entropy w.r.t. z2
        d_z2 = cross_entropy_deriv(a2, y_batch)  # shape: (batch_size, 10)

        # gradient w.r.t. W2 and b2
        dW2 = a1.T.dot(d_z2)                     # shape: (128, 10)
        db2 = np.sum(d_z2, axis=0, keepdims=True)

        # derivative w.r.t. a1
        d_a1 = d_z2.dot(W2.T)                    # shape: (batch_size, 128)

        # derivative w.r.t. z1
        d_z1 = d_a1 * relu_deriv(z1)

        # gradient w.r.t. W1 and b1
        dW1 = x_batch.T.dot(d_z1)                # shape: (784, 128)
        db1 = np.sum(d_z1, axis=0, keepdims=True)

        # ---------- Update weights ----------
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

        # (Optional) print loss every 100 batches
        if (i + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{num_batches}, Loss: {loss:.4f}")

# ------------------------------------------------------------
# 7. Evaluation on test data
# ------------------------------------------------------------
def predict(X):
    # Only forward pass
    z1_ = X.dot(W1) + b1
    a1_ = relu(z1_)
    z2_ = a1_.dot(W2) + b2
    a2_ = softmax(z2_)
    return np.argmax(a2_, axis=1)

y_test_labels = np.argmax(y_test, axis=1)
y_pred = predict(x_test)

accuracy = np.mean(y_pred == y_test_labels)
print(f"\nTest accuracy: {accuracy * 100:.2f}%")
