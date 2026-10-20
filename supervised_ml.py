import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Prepare the dataset using Python lists
hours_studied = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
exam_scores = [30, 40, 45, 55, 65, 70, 80, 85, 90, 95]

#Convert the list to numpy array and resahpe it to a 2D
x = np.array(hours_studied).reshape(-1, 1)

# Separate input features (X) and target values (y)
X = np.array(hours_studied).reshape(-1, 1)
y = np.array(exam_scores)

# Create a Linear Regression model
model = LinearRegression()

# train the model using fit()
model.fit(X, y)

# Allow the user to enter the number of hours studied using input()
user_input = float(input("Enter hours studied: "))

# use predict() to estimate the examination score
prediction = model.predict([[user_input]])
    
# Display the prediction rounded to two decimal places
print(f"Predicted exam score: {prediction[0]:.2f}")
    
# BONUS TASK: Use Matplotlib to create a graph ---
# Generate predictions for the trendline
trendline_predictions = model.predict(X)

plt.figure(figsize=(8, 5))
# Plot actual data points
plt.scatter(X, y, color='blue', label='Actual Scores')
# Plot regression line
plt.plot(X, trendline_predictions, color='red', linewidth=2, label='Regression Line')

# Chart formatting
plt.title('Study Hours vs. Examination Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Display the graph
plt.show()
