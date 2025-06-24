import pandas as pd
import numpy as np

class PerceptronOCR:
    def __init__(self, input_excel_file=None):
        self.input_excel_file = input_excel_file
        self.df = pd.read_excel(input_excel_file, header=None)

        self.X = self.preprocess()[0]  # Features
        self.Y = self.preprocess()[1]  # Labels
        print(f"Features shape: {self.X.shape}")
        print(f"Labels shape: {self.Y.shape}")

        # weights and biases
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None


    # internal functions
    def _preprocess_excel(self):
        characters = []
        labels = []

        for i in range(2, len(self.df), 10):
            block = self.df.iloc[i:i+7, 1:6]   
            label_cell = self.df.iloc[i, 9]    

            if pd.isna(label_cell):
                print(f"⚠️ Missing label at row {i}")
                continue
            block = block.fillna(0)      
            flat = block.values.flatten().astype(np.float32)

            if flat.shape[0] != 35:
                print(f"⚠️ Skipping malformed block at row {i}")
                continue
            characters.append(flat)
            labels.append(label_cell)
    
        return np.array(characters), np.array(labels)
    
    def _label_encode(self, y_raw):
        classes = sorted(set(y_raw))
        label_to_idx = {label: i for i, label in enumerate(classes)}
        y = np.array([label_to_idx[c] for c in y_raw])
        return y, classes

    def _one_hot(self, y, num_classes):
        result = np.zeros((len(y), num_classes))
        result[np.arange(len(y)), y] = 1
        return result
    
    def preprocess(self):
        X, y_raw = self._preprocess_excel()
        y, classes = self._label_encode(y_raw)
        y_one_hot = self._one_hot(y, len(classes)   )
        return X, y_one_hot

    # activation functions
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def cross_entropy(self, pred, true):
        return -np.sum(true * np.log(pred + 1e-9)) / len(true)
    
    def accuracy(self, pred, true):
        return np.mean(np.argmax(pred, axis=1) == np.argmax(true, axis=1))
    
    # training function
    def train(
            self,
            input_size=35,
            learning_rate=0.01,

            # 0.67 * (input_size + output_size)
            hidden_size=26,
            epochs=1000,
        ):
        output_size = self.Y.shape[1]
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

        for epoch in range(epochs):
            # Forward
            z1 = self.X @ self.W1 + self.b1
            a1 = self.sigmoid(z1)

            z2 = a1 @ self.W2 + self.b2
            a2 = self.softmax(z2)

            # Loss
            loss = self.cross_entropy(a2, self.Y)

            # Backward
            dz2 = a2 - self.Y                       
            dW2 = a1.T @ dz2 / len(self.X)
            db2 = np.sum(dz2, axis=0, keepdims=True) / len(self.X)

            dz1 = (dz2 @ self.W2.T) * self.sigmoid_deriv(z1)
            dW1 = self.X.T @ dz1 / len(self.X)
            db1 = np.sum(dz1, axis=0, keepdims=True) / len(self.X)

            # Update
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2

            
            print(f"Epoch {epoch}, Loss: {loss:.4f} , Accuracy: {self.accuracy(a2, self.Y):.4f}")

    def predict(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.sigmoid(z1)

        z2 = a1 @ self.W2 + self.b2
        a2 = self.softmax(z2)
        return np.argmax(a2, axis=1)

    def sample_predict(self):
        preds = self.predict(self.X)
        y_true = np.argmax(self.Y, axis=1)

        # Rebuild the classes from the original labels
        _, classes = self._label_encode(self._preprocess_excel()[1])  # just to get class names

        for i, (pred_idx, true_idx) in enumerate(zip(preds, y_true)):
            print(f"Sample {i+1}: predicted {classes[pred_idx]}, actual {classes[true_idx]}")


