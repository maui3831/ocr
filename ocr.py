import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class PerceptronOCR:
    def __init__(self, input_file=None):
        self.input_file = input_file
        self.df = pd.read_csv(input_file)

        self.X, self.Y = self._preprocess_csv()
        logging.info(f"Features shape: {self.X.shape}")
        logging.info(f"Labels shape: {self.Y.shape}")

        # weights and biases
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        # training history
        self.train_history = []

    # internal functions
    def _preprocess_csv(self):
        features = []
        labels = []
        unique_labels = sorted(self.df["label"].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        for _, row in self.df.iterrows():
            pixel_values = row.iloc[:-1].values.astype(np.float32)
            label = row["label"]
            grid = pixel_values.reshape(7, 5)
            row_sums = np.sum(grid, axis=1)
            col_sums = np.sum(grid, axis=0)
            feature_vector = np.concatenate([row_sums, col_sums])
            features.append(feature_vector)
            labels.append(self.label_to_idx[label])

        return np.array(features), np.array(labels).astype(np.int32).reshape(-1, 1)

    def preprocess(self):
        # Only CSV supported
        return self._preprocess_csv()

    # activation functions
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def mse(self, pred, true):
        return np.mean((pred - true) ** 2)

    def accuracy(self, pred, true):
        pred_class = np.argmax(pred, axis=1)
        true_class = true.flatten()
        return np.mean(pred_class == true_class)

    def cross_entropy(self, pred, true):
        m = true.shape[0]
        p = pred[range(m), true.flatten()]
        log_likelihood = -np.log(p + 1e-8)
        return np.mean(log_likelihood)

    # training function
    def train(
        self,
        input_size=None,
        learning_rate=0.01,
        hidden_size=16,
        epochs=1000,
    ):
        if input_size is None:
            input_size = self.X.shape[1]  # Auto-detect input size

        output_size = len(self.label_to_idx)  # Number of classes

        # He initialization for ReLU layers
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        # Feature normalization for CSV data
        self.feature_mean = np.mean(self.X, axis=0)
        self.feature_std = np.std(self.X, axis=0) + 1e-8
        X_normalized = (self.X - self.feature_mean) / self.feature_std

        for epoch in range(epochs):
            z1 = X_normalized @ self.W1 + self.b1
            a1 = self.relu(z1)
            z2 = a1 @ self.W2 + self.b2
            a2 = self.softmax(z2)  # Softmax output
            loss = self.cross_entropy(a2, self.Y)

            # Backward pass
            dz2 = a2.copy()  # Use a copy to avoid modifying a2 in-place
            dz2[range(len(self.Y)), self.Y.flatten()] -= 1
            dz2 /= len(self.Y)
            dW2 = a1.T @ dz2
            db2 = np.sum(dz2, axis=0, keepdims=True)

            dz1 = (dz2 @ self.W2.T) * self.relu_deriv(z1)
            dW1 = X_normalized.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            # Gradient clipping to prevent explosion
            grad_norm = np.sqrt(
                np.sum(dW1**2) + np.sum(db1**2) + np.sum(dW2**2) + np.sum(db2**2)
            )
            if grad_norm > 1.0:
                dW1 = dW1 / grad_norm
                db1 = db1 / grad_norm
                dW2 = dW2 / grad_norm
                db2 = db2 / grad_norm

            # Update weights
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2

            # Accuracy calculation (no denormalization)
            accuracy = self.accuracy(a2, self.Y)

            # Store training history
            self.train_history.append(
                {"epoch": epoch, "loss": loss, "accuracy": accuracy}
            )

            if epoch % 100 == 0:  # Print every 100 epochs
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        print(f"Final - Epoch {epochs - 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        # Log final weights and biases
        logging.info(f"Final W1: {self.W1}")
        logging.info(f"Final b1: {self.b1}")
        logging.info(f"Final W2: {self.W2}")
        logging.info(f"Final b2: {self.b2}")

    def predict(self, X):
        X_normalized = (X - self.feature_mean) / self.feature_std
        z1 = X_normalized @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        probs = self.softmax(z2)
        pred_class = np.argmax(probs, axis=1)
        return pred_class

    def sample_predict(self):
        preds = self.predict(self.X)
        y_true = self.Y.flatten()
        print("Sample predictions for digit recognition:")
        for i, (pred, true) in enumerate(zip(preds, y_true)):
            pred_label = self.idx_to_label[pred]
            true_label = self.idx_to_label[true]
            print(f"Sample {i + 1}: predicted '{pred_label}', actual '{true_label}'")

    def history(self):
        if not self.train_history:
            print("No training history available.")
            return

        history_df = pd.DataFrame(self.train_history)
        print(history_df)
        return history_df
