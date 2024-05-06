Fuel Consumption Linear Regression
This script performs linear regression using TensorFlow to predict highway fuel consumption based on city fuel consumption. It reads a dataset from a CSV file (FuelConsumptionCo2.csv), trains the model, and plots the loss over epochs to visualize the training process.

Dependencies
Python 3.x
TensorFlow
NumPy
Pandas
Matplotlib
Usage
Make sure you have all the dependencies installed.
Download the dataset file FuelConsumptionCo2.csv.
Run the script fuel_consumption_linear_regression.py.
Dataset
The dataset contains information about fuel consumption and CO2 emissions for various vehicles. It includes the following columns:

FUELCONSUMPTION_CITY: City fuel consumption in liters per 100 kilometers.
FUELCONSUMPTION_HWY: Highway fuel consumption in liters per 100 kilometers.
Model
The script defines a linear regression model with two parameters (a and b) and uses gradient descent to optimize these parameters based on the mean squared error loss function.
