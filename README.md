# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start 2.Input dataset → load data (Hours studied, Marks scored) 3.Preprocess data → separate input (X = Hours) and output (Y = Marks) 4.Split dataset → training set and testing set 5.Train model → fit a Linear Regression model on training data 6.Calculate slope 𝑚 and intercept 𝑐 7.Equation: 𝑌=𝑚𝑋+𝑐 8.Predict marks using test data (X_test) 9.Visualize regression line with actual data and Predict new values (e.g., marks for 7.5 hours studied) 10.End
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: NARESH KUMAR R
RegisterNumber: 212224040213
*/
# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Load dataset from location
df = pd.read_csv("C:/Users/YourName/Documents/student_scores.csv")

# Step 3: Split data
X = df[['Hours']]   # Feature
y = df['Marks']     # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict
y_pred = model.predict(X_test)

# Step 6: Evaluation
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Step 7: Visualization
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.legend()
plt.show()

# Predict marks for 7.5 hours
hours = np.array([[7.5]])
predicted_marks = model.predict(hours)
print(f"Predicted marks for 7.5 hours: {predicted_marks[0]:.2f}")

```

## Output:
<img width="897" height="687" alt="image" src="https://github.com/user-attachments/assets/fceea11a-6bd0-457d-9cda-274d8d741977" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
