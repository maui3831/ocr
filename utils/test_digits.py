#!/usr/bin/env python3
"""
Test script for digit recognition using the modified OCR system.
This script demonstrates the new CSV-based approach with 12-element feature vectors.
"""

import numpy as np
from ocr import PerceptronOCR


def main():
    print("🔢 Digit Recognition with 12-Element Feature Vectors")
    print("=" * 60)

    # Initialize the model with CSV data
    try:
        model = PerceptronOCR("digits_data.csv")
        print("✅ Model initialized successfully!")
        print(
            f"📊 Input features: {model.X.shape[1]} dimensions (row sums + column sums)"
        )
        print(
            f"🎯 Output targets: ASCII values {int(model.Y.min())} to {int(model.Y.max())}"
        )
        print()

        # Show feature extraction example
        print("📈 Feature Extraction Example:")
        print("Original 5x7 grid for digit '0':")
        sample_grid = model.df.iloc[0, :-1].values.reshape(7, 5)
        for row in sample_grid:
            print("".join("✅" if x else "❌" for x in row))

        row_sums = np.sum(sample_grid, axis=1)
        col_sums = np.sum(sample_grid, axis=0)
        print(f"Row sums (7 values): {row_sums}")
        print(f"Column sums (5 values): {col_sums}")
        print(
            f"Combined feature vector (12 values): {np.concatenate([row_sums, col_sums])}"
        )
        print()

        # Train the model
        print("🚀 Training the neural network...")
        print("Architecture: 12 → 16 → 1 (with ReLU activation)")
        model.train(input_size=12, hidden_size=16, learning_rate=0.1, epochs=100)
        print()

        # Show sample predictions
        print("🔍 Sample Predictions:")
        model.sample_predict()
        print()

        # Test with custom input
        print("🧪 Custom Test:")
        # Create a simple test pattern for digit '1'
        test_pattern = np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
            ]
        )

        print("Test pattern for '1':")
        for row in test_pattern:
            print("".join("✅" if x else "❌" for x in row))

        # Extract features
        row_sums = np.sum(test_pattern, axis=1)
        col_sums = np.sum(test_pattern, axis=0)
        test_features = np.concatenate([row_sums, col_sums]).reshape(1, -1)

        prediction = model.predict(test_features)
        pred_ascii = int(np.round(prediction.flatten()[0]))
        pred_char = chr(pred_ascii) if 32 <= pred_ascii <= 126 else "?"

        print(f"Predicted ASCII: {pred_ascii}")
        print(f"Predicted character: '{pred_char}'")
        print("Expected: '1' (ASCII 49)")

        if pred_char == "1":
            print("✅ Correct prediction!")
        else:
            print("❌ Incorrect prediction")

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    return True


if __name__ == "__main__":
    main()
