# ANN Training - Optical Character recognition

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)

For Introduction to Artificial Intelligence
by AI Group 2

## Project Description

This project implements a web-based Optical Character Recognition (OCR) neural network trainer using Streamlit. It allows users to train and do inference on the multilayer perceptron model, employing a distinct weight initialization strategy through He initialization. 

**KaimingOCR (Kaiming He Initialization):** Implements Kaiming He 
initialization, specifically designed for ReLU activations.

The application processes 7x5 pixel character data by extracting 12-element feature vectors (7 row sums and 5 column sums). It provides a visual interface for drawing characters, inputting custom pixel data, training the models, and visualizing their performance metrics and predictions.

## Features

* Interactive Drawing Grid: Draw 7x5 pixel characters directly in the web UI.

* Custom Pixel Input: Provide character pixel data as a comma-separated string (35 pixels + 1 ASCII label).


* Comparative Training Plots: Visualize and contrast the training loss and accuracy of all over epochs using Plotly graphs.

* Detailed Training History: View and download the full training history (loss, accuracy per epoch) for each model.

* Sample Predictions: See how each trained model performs on a sample of the training data.


* Feature Analysis: Display the 12-element feature vector extracted from the input.

* Data Source Options: Use pre-included CSV datasets or upload your own.

## How to Run

### Prerequisites:

- Python 3.x
- pip (Python package installer) or uv

### Installation
Clone the repository (if applicable):

```bash
git clone <repository_url>
cd <repository_name>
```

### Create a virtual environment (recommended):

```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### Install the required packages:

Create a requirements.txt file in the root directory of your project with the following content:

streamlit
numpy
pandas
plotly

Then, install the dependencies:

```bash
pip install -r requirements.txt
# Or using uv:
uv sync
```

### Data Setup

- The application expects CSV data files in a data/ directory.

- data/digits_data.csv: A smaller dataset for digit recognition.

- data/more_digits_data.csv: A larger dataset (if available).

- Ensure you have a data folder in the same directory as main.py, and place your CSV files inside it. The format for CSV files should be 35 pixel values (0 or 1) followed by a single ASCII label.

**Example row in CSV:** `0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,0,1,1,1,0,48` (35 pixels for '0', followed by 48 which is ASCII for '0').

### Running the Application
Ensure your virtual environment is active.

Run the Streamlit application:

```bash
streamlit run main.py
```