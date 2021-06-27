# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# Import the csv file
df = pd.read_csv('data.csv')

# prints the top 5 rows
print(df.head())

# Prepare the training set
x_train = df['Father'].values[:, np.newaxis]
y_train = df['Son'].values

lm = LinearRegression()

# Train the model
lm.fit(x_train, y_train)

# Prepare the test data
x_test = [[72.8], [61.1], [67.4], [70.2], [75.6], [60.2], [65.3], [59.2]]

# Test the model
predictions = lm.predict(x_test)
print(predictions)

# Plot the training data
plt.scatter(x_train, y_train, color='b')

# Plot the best fit line using predicted value
plt.plot(x_test, predictions, color='black', linewidth=3)
plt.xlabel('Father height in inches')
plt.ylabel('Son height in inches')
plt.show()
