# deep-learning-challenge

## Overview
This project aims to build a binary classifier using machine learning and neural networks. The classifier predicts the success of applicants for funding from the nonprofit foundation Alphabet Soup.

## Project Structure
- `charity_data.csv`: Dataset file containing more than 34,000 past applications.
- `AlphabetSoupCharity.ipynb`: Jupyter notebook for preprocessing and model training.
- `AlphabetSoupCharity.h5`: Saved model after training.

## Requirements
- Python 3.x
- TensorFlow
- Pandas
- scikit-learn

## Setup and Installation
1. Clone the repository to your local machine.
2. Install the required dependencies:


## Data Preprocessing
- The `charity_data.csv` file is loaded into a Pandas DataFrame.
- Non-beneficial ID columns (`EIN`, `NAME`) are dropped.
- Categorical variables are encoded using one-hot encoding.
- The dataset is split into features and target arrays, then into training and testing datasets.
- Feature scaling is applied to the training and testing datasets.

## Model Architecture
- The model is a deep neural network with multiple layers.
- The input data features determine the number of neurons in the input layer.
- The model includes hidden layers with ReLU activation functions and an output layer with a sigmoid activation function.

## Training and Evaluation
- The model is compiled with a binary crossentropy loss function and an Adam optimizer.
- It is trained on the training data for a specified number of epochs.
- Model performance (loss and accuracy) is evaluated using the test data.

## Usage
- To use the model, run Google Colab, which includes steps for training and evaluation.
- The trained model can be used to make predictions on new data formatted similarly to `charity_data.csv`.



