"""
Digit Visualization Script for OCR Data

This script reads the digits_data.csv file and visualizes the digit patterns
as grids to help understand the data structure and patterns.
"""

import pandas as pd
import numpy as np


def load_digit_data(csv_path):
    """Load the digit data from CSV file."""
    df = pd.read_csv(csv_path)
    return df


def ascii_to_digit(ascii_val):
    """Convert ASCII value to actual digit (48->0, 49->1, etc.)."""
    return ascii_val - 48


def print_digit_as_ascii(pixel_data, digit_label):
    """Print a digit pattern using ASCII characters."""
    grid = np.array(pixel_data).reshape(7, 5)
    print(f"\nDigit {digit_label}:")
    print("-" * 7)
    for row in grid:
        line = ""
        for pixel in row:
            line += "✅" if pixel == 1 else "❌"
        print(f"|{line}|")
    print("-" * 7)


def main():
    # Load the data
    print("Loading digit data from CSV...")
    df = load_digit_data("digits_data.csv")

    print(f"Loaded {len(df)} digit examples")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Convert ASCII labels to digits
    df["digit"] = df["label"].apply(ascii_to_digit)

    # Show data distribution
    print("\nDigit distribution:")
    digit_counts = df["digit"].value_counts().sort_index()
    for digit, count in digit_counts.items():
        print(f"Digit {digit}: {count} examples")

    # Get pixel columns
    pixel_cols = [col for col in df.columns if col.startswith("pixel_")]
    print(f"\nNumber of pixel features: {len(pixel_cols)}")

    # Print all examples as ASCII art
    print("\nAll examples as ASCII art:")
    for i in range(len(df)):
        row = df.iloc[i]
        pixel_data = row[pixel_cols].values
        digit_label = row["digit"]
        print_digit_as_ascii(pixel_data, digit_label)

    for digit in range(10):
        digit_examples = df[df["digit"] == digit]
        if not digit_examples.empty:
            first_example = digit_examples.iloc[0]
            pixel_data = first_example[pixel_cols].values


if __name__ == "__main__":
    main()
