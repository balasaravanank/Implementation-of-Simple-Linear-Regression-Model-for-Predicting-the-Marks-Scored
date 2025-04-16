# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load Dataset
df = pd.read_csv('student_scores.csv')
print("Full DataFrame:\n", df)
print("\nHeader Only:\n", df.head(0))
print("\nFirst 5 Rows:\n", df.head())
print("\nLast 5 Rows:\n", df.tail())

# Prepare Features and Target
x = df.iloc[:, :-1].values  # Hours
y = df.iloc[:, 1].values    # Scores
print("\nFeatures (x):\n", x)
print("\nTarget (y):\n", y)

# Split into Training and Testing Sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

# Train Linear Regression Model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict Test Set Results
y_pred = regressor.predict(x_test)
print("\nPredicted Scores:\n", y_pred)
print("\nActual Scores:\n", y_test)

# Plot Training Set Results
plt.scatter(x_train, y_train, color='black')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Plot Testing Set Results
plt.scatter(x_test, y_test, color='black')
plt.plot(x_train, regressor.predict(x_train), color='red')  # Same line as training
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Evaluate the Model
mse = mean_absolute_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nEvaluation Metrics:")
print("MSE =", mse)
print("MAE =", mae)
print("RMSE =", rmse)


Developed by: BALA SARAVANAN K
RegisterNumber:  212224230031
*/
```

## Output:



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
