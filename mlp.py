"""
Low-level MLP (Multi-Layer Perceptron) using only NumPy.
Architecture: 784 -> 256 -> 128 -> 10
Training with mini-batch SGD + cross-entropy loss.
"""

import numpy as np


# ── Activations ──────────────────────────────────────────────────────────────

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ── Loss ─────────────────────────────────────────────────────────────────────

def cross_entropy(probs, y_onehot):
    return -np.mean(np.sum(y_onehot * np.log(probs + 1e-9), axis=1))


# ── MLP ──────────────────────────────────────────────────────────────────────

class MLP:
    def __init__(self, layer_sizes, lr=0.01, seed=42):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # He init
            W = rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
            b = np.zeros(fan_out)
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        """Returns (probs, cache) where cache has pre/post activations."""
        cache = []
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            if i < len(self.weights) - 1:   # hidden layer
                a_next = relu(z)
            else:                            # output layer
                a_next = softmax(z)
            cache.append((a, z))
            a = a_next
        return a, cache

    def backward(self, probs, y_onehot, cache):
        """Computes gradients and updates weights."""
        n = probs.shape[0]
        # gradient at output (softmax + cross-entropy combined)
        delta = (probs - y_onehot) / n

        for i in reversed(range(len(self.weights))):
            a_prev, z = cache[i]
            dW = a_prev.T @ delta
            db = delta.sum(axis=0)
            if i > 0:
                _, z_prev = cache[i - 1]
                delta = (delta @ self.weights[i].T) * relu_grad(z_prev)
            # update
            self.weights[i] -= self.lr * dW
            self.biases[i]  -= self.lr * db

    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

    def train(self, X, y_onehot, epochs=100, batch_size=64):
        n = X.shape[0]
        for epoch in range(1, epochs + 1):
            # shuffle
            idx = np.random.permutation(n)
            X, y_onehot = X[idx], y_onehot[idx]

            total_loss = 0.0
            for start in range(0, n, batch_size):
                Xb = X[start:start + batch_size]
                yb = y_onehot[start:start + batch_size]
                probs, cache = self.forward(Xb)
                total_loss += cross_entropy(probs, yb) * len(Xb)
                self.backward(probs, yb, cache)

            if epoch % 10 == 0 or epoch == 1:
                avg_loss = total_loss / n
                preds = self.predict(X)
                acc   = np.mean(preds == np.argmax(y_onehot, axis=1))
                print(f"Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  train_acc={acc:.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Load data (shape: features x samples  ->  transpose to samples x features)
    X_train = np.loadtxt("train_samples.csv", delimiter=",").T   # (1000, 784)
    y_raw   = np.loadtxt("train_labels.csv")                     # (1000,)  values 1-10
    X_test  = np.loadtxt("test_samples.csv",  delimiter=",").T   # (5000, 784)

    # Normalize
    X_train = X_train / (X_train.max() + 1e-8)
    X_test  = X_test  / (X_test.max()  + 1e-8)

    # Labels to 0-indexed one-hot
    n_classes = 10
    y = y_raw.astype(int) - 1                                    # 0..9
    y_onehot = np.eye(n_classes)[y]                              # (1000, 10)

    # Build & train
    model = MLP(layer_sizes=[784, 256, 128, n_classes], lr=0.05)
    model.train(X_train, y_onehot, epochs=200, batch_size=64)

    # Predict on test set
    test_preds = model.predict(X_test) + 1   # back to 1-indexed

    # Save submission
    with open("submission.csv", "w") as f:
        f.write("Id,Prediction\n")
        for i, p in enumerate(test_preds, start=1):
            f.write(f"{i},{p}\n")

    print(f"\nSubmission saved to submission.csv  ({len(test_preds)} predictions)")


if __name__ == "__main__":
    main()

