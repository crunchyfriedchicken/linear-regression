# Install packages ---------------------------------------------------------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Open data and explore data -----------------------------------------------------------------------------
df = pd.read_csv("./data/USA_Housing.csv")
df.info()
df.head()
df.describe()

# Remove categorical data (not working with NLP this project)---------------------------------------------
df.drop(labels="Address",axis=1, inplace=True)
df.info()

# Visualise the data -------------------------------------------------------------------------------------
# Set style
sns.set_style("whitegrid")

# See relationship between all variables graphically
sns.pairplot(df)
plt.show()

# Check distribution of price
sns.histplot(df["Price"])
plt.show()

# See correlation between variables with a heatmap - can see Area Income has a high correlation with Price
sns.heatmap(df.corr(),annot=True)
plt.show()

# Plot relationship between Area Income and Price
sns.scatterplot(df, x = "Avg. Area Income", y="Price")
plt.show()

# Check for missing data ---------------------------------------------------------------------------------
sns.heatmap(df.isnull())
plt.show()
# no missing data 

# Split data ---------------------------------------------------------------------------------------------
df.columns

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y = df["Price"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.4, random_state=101)

# Create and train model ---------------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)

# Evaluate linear regression model -----------------------------------------------------------------------
# Find intercept
lr.intercept_

# Find coefficients and evaluate them
lr.coef_
X_train.columns

pd.DataFrame(lr.coef_,X_train.columns, columns=["Coefficient"])
"""
Interpreting the coefficients (note: doesn't make sense because the data is fake):
- Holding all other features fixed, a 1 unit increase in Avg. Area Income is associated with an increase of $21.52.
- Holding all other features fixed, a 1 unit increase in Avg. Area House Age is associated with an increase of $164883.28.
- Holding all other features fixed, a 1 unit increase in Avg. Area Number of Rooms is associated with an increase of $122368.67.
- Holding all other features fixed, a 1 unit increase in Avg. Area Number of Bedrooms is associated with an increase of $2233.80.
- Holding all other features fixed, a 1 unit increase in Area Population is associated with an increase of $15.15.
"""

# Predict data using X_test ------------------------------------------------------------------------------
predictions = lr.predict(X_test)

# plot to check relationship between y_test and predictions
sns.scatterplot(x = y_test, y =predictions)
plt.show()
# line similar to y=x shows model is quite accurate at predicting

sns.histplot((y_test-predictions), bins=50)
plt.show()
# shows difference of 0 is the most occurent meaning model is pretty good

# Assess the accuracy of the model -----------------------------------------------------------------------
from sklearn import metrics
# loss functions which we hope to minimize

# Mean absolute error is mean of absolute values - easiest metric to understand average error
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

# Mean squared error is mean of squared values - punishes large errors which is useful in real world
print('MSE:', metrics.mean_squared_error(y_test, predictions))

# Root mean squared error is root of MSE - most popular because it interpretable in "y" units
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))