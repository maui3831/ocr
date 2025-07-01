#!/usr/bin/env python3
"""
Test script for digit recognition using the modified OCR system.
This script demonstrates the new CSV-based approach with 12-element feature vectors.
"""

import numpy as np
from glorot import GlorotOCR
from kaiming import KaimingOCR
from ocr import PerceptronOCR


def run_test_for_model(model_name, model_class, data_path="data/digits_data.csv"):
    """
    Runs a test for a given OCR model class.
    """
    print(f"\nTesting {model_name} Model")
    print("=" * (20 + len(model_name)))

    try:
        model = model_class(data_path)
        print(f"✅ {model_name} initialized successfully!")
        print(
            f"📊 Input features: {model.X.shape[1]} dimensions (row sums + column sums)"
        )
        print(
            f"🎯 Output targets: ASCII values {int(model.Y.min())} to {int(model.Y.max())}"
        )
        print()

        # Show feature extraction example (common for all models as preprocessing is similar)
        if model_name == "PerceptronOCR": # Only show this once as it's data preprocessing related
            print("📈 Feature Extraction Example (from original CSV):")
            print("Original 5x7 grid for digit '0':")
            sample_grid = model.df.iloc[0, :-1].values.reshape(7, 5)
            for row in sample_grid:
                print("".join("✅" if x else "❌" for x in row))

            row_sums = np.sum(sample_grid, axis=1)
            col_sums = np.sum(sample_grid, axis=0)
            print(f"Row sums (7 values): {row_sums}")
            print(f"Column sums (5 values): {col_sums}")
            print(
                f"Combined features (12 values): {np.concatenate([row_sums, col_sums])}"
            )
            print()

        # Train the model
        print(f"🚀 Training {model_name}...")
        model.train(input_size=12, hidden_size=16, learning_rate=0.1, epochs=1000) # Increased epochs for better training
        print()

        # Show sample predictions
        print(f"🔍 Sample Predictions for {model_name}:")
        model.sample_predict()
        print()

        # Test with custom input (digit '1')
        print(f"🧪 Custom Test for {model_name}:")
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

        print(f"Predicted ASCII: {pred_ascii} ('{pred_char}')")
        print(f"Raw output: {prediction.flatten()[0]:.2f}")
        print("-" * (20 + len(model_name)))

    except FileNotFoundError:
        print(f"❌ Error: Data file '{data_path}' not found for {model_name}.")
        print("Please ensure 'data/digits_data.csv' exists.")
    except Exception as e:
        print(f"❌ An error occurred during {model_name} test: {e}")


def main():
    print("🔢 Digit Recognition with 12-Element Feature Vectors - All Models")
    print("=" * 60)

    # Path to your dataset
    data_file = "data/digits_data.csv"

    # Run tests for each model
    run_test_for_model("PerceptronOCR", PerceptronOCR, data_file)
    run_test_for_model("GlorotOCR", GlorotOCR, data_file)
    run_test_for_model("KaimingOCR", KaimingOCR, data_file)


if __name__ == "__main__":
    main()