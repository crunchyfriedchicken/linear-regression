# Install packages ---------------------------------------------------------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Open data and explore data -----------------------------------------------------------------------------
from sklearn.datasets import fetch_california_housing
california_housing = fetch_california_housing()

# get list of attributes
dir(california_housing)

# Read about the dataset
print(california_housing.DESCR)

# Check out each attribute
california_housing.data
california_housing.feature_names

california_housing.target
california_housing.target_names

# Create a DataFrame from the data and target
feature = pd.DataFrame(data=california_housing.data, columns=california_housing.feature_names)
target = pd.DataFrame(data=california_housing.target, columns=['MedHouseVal'])

# Concatenate data and target along columns to create the final DataFrame
df = pd.concat([feature, target], axis=1)

# Now, df contains the data in a Pandas DataFrame
df.head()
df.info()
df.describe()

# Visualise the data -------------------------------------------------------------------------------------
# Set style
sns.set_style("whitegrid")

# See relationship between all variables graphically
#sns.pairplot(df)
#plt.show()

# Check distribution of price - interesting that there is a massive surge at the most expensive price
sns.histplot(df["MedHouseVal"])
plt.show()

# See correlation between variables with a heatmap - high positive correlation with MedInc and MedHouseVal
sns.heatmap(df.corr(),annot=True)
plt.show()
# MedInc is median income in block group so makes sense 

# Plot relationship between MedInc and MedHouseVal
sns.scatterplot(df, x = "MedInc", y="MedHouseVal")
plt.show()

# Check for missing data ---------------------------------------------------------------------------------
sns.heatmap(df.isnull())
plt.show()
# no missing data 

# Split data ---------------------------------------------------------------------------------------------
df.columns

X = feature
y = df["MedHouseVal"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=101)

# Create and train model ---------------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)

# Evaluate linear regression model -----------------------------------------------------------------------
# Find intercept
lr.intercept_

# Find coefficients and evaluate them
lr.coef_
X.columns

pd.DataFrame(lr.coef_[0],X.columns, columns=["Coefficient"])
# Holding each other feature fixed, a 1 unit increase in MedInc leads to an increase in price unit of 0.43

# Predict data using X_test ------------------------------------------------------------------------------
predictions = lr.predict(X_test)

# plot to check relationship between y_test and predictions
sns.scatterplot(x=y_test, y=predictions)
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