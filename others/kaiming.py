import pandas as pd
import numpy as np


class KaimingOCR:
    def __init__(self, input_file=None):
        self.input_file = input_file
        self.df = pd.read_csv(input_file)

        self.X, self.Y = self._preprocess_csv()
        print(f"Features shape: {self.X.shape}")
        print(f"Labels shape: {self.Y.shape}")

        # weights and biases
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        # training history
        self.train_history = []

    # internal functions
    def _preprocess_csv(self):
        """Process CSV data and extract 12-element feature vectors"""
        features = []
        labels = []

        for _, row in self.df.iterrows():
            # Extract the 35 pixel values
            pixel_values = row.iloc[:-1].values.astype(
                np.float32
            )  # All columns except 'label'
            label = row["label"]

            # Reshape to 7x5 grid (7 rows, 5 columns)
            grid = pixel_values.reshape(7, 5)

            # Compute row sums (7 values) and column sums (5 values)
            row_sums = np.sum(grid, axis=1)  # Sum each row (7 values)
            col_sums = np.sum(grid, axis=0)  # Sum each column (5 values)

            # Combine into 12-element feature vector
            feature_vector = np.concatenate([row_sums, col_sums])

            features.append(feature_vector)
            labels.append(label)

        return np.array(features), np.array(labels).astype(np.float32).reshape(-1, 1)

    def preprocess(self):
        # Only CSV supported
        return self._preprocess_csv()

    # activation functions
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

    def sigmoid_deriv(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def mse(self, pred, true):
        return np.mean((pred - true) ** 2)

    def accuracy(self, pred, true):
        # For regression, check if rounded prediction matches true ASCII value
        pred_ascii = np.round(pred).astype(int)
        true_ascii = true.astype(int)
        return np.mean(pred_ascii == true_ascii)

    # training function
    def train(
        self,
        input_size=None,
        learning_rate=0.01,
        hidden_size=26,
        epochs=1000,
    ):
        if input_size is None:
            input_size = self.X.shape[1]  # Auto-detect input size

        output_size = 1  # Single output for regression

        # Kaiming He initialization for ReLU activation functions
        # For W1 (input_size -> hidden_size)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        # For W2 (hidden_size -> output_size)
        # Assuming linear output, so Kaiming is still appropriate based on the previous layer's ReLU
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))


        # Feature normalization for CSV data
        self.feature_mean = np.mean(self.X, axis=0)
        self.feature_std = np.std(self.X, axis=0) + 1e-8
        X_normalized = (self.X - self.feature_mean) / self.feature_std

        # Normalize targets to help with training stability
        self.target_mean = np.mean(self.Y)
        self.target_std = np.std(self.Y) + 1e-8
        Y_normalized = (self.Y - self.target_mean) / self.target_std

        for epoch in range(epochs):
            # Forward pass
            z1 = X_normalized @ self.W1 + self.b1
            a1 = self.relu(z1)  # Use ReLU for CSV data
            z2 = a1 @ self.W2 + self.b2
            a2 = z2  # Linear output for regression
            loss = self.mse(a2, Y_normalized)

            # Backward pass
            dz2 = (a2 - Y_normalized) / len(X_normalized)  # MSE derivative
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

            # Denormalize predictions for accuracy calculation
            a2_denorm = a2 * self.target_std + self.target_mean
            accuracy = self.accuracy(a2_denorm, self.Y)

            # Store training history
            self.train_history.append(
                {"epoch": epoch, "loss": loss, "accuracy": accuracy}
            )

            if epoch % 100 == 0:  # Print every 100 epochs
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        print(f"Final - Epoch {epochs - 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    def predict(self, X):
        # Normalize input
        if hasattr(self, "feature_mean"):
            X_normalized = (X - self.feature_mean) / self.feature_std
        else:
            X_normalized = X

        z1 = X_normalized @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2

        # Denormalize the output for regression
        output_normalized = z2
        if hasattr(self, "target_mean"):
            output = output_normalized * self.target_std + self.target_mean
        else:
            output = output_normalized
        return output

    def sample_predict(self):
        # For CSV regression
        preds = self.predict(self.X)
        y_true = self.Y.flatten()

        print("Sample predictions for digit recognition:")
        for i, (pred, true) in enumerate(zip(preds.flatten(), y_true)):
            pred_rounded = int(np.round(pred))
            pred_char = chr(pred_rounded) if 32 <= pred_rounded <= 126 else "?"
            true_char = chr(int(true))
            print(
                f"Sample {i + 1}: predicted ASCII {pred_rounded} ('{pred_char}'), actual ASCII {int(true)} ('{true_char}')"
            )

    def history(self):
        if not self.train_history:
            print("No training history available.")
            return

        history_df = pd.DataFrame(self.train_history)
        print(history_df)
        return history_df