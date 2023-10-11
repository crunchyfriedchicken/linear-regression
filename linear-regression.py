# Install packages --------------------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Open data and explore data ----------------------------------------
df = pd.read_csv("./data/USA_Housing.csv")
df.info()
df.head()
df.describe()

# Remove categorical data (not working with NLP this project)-------
df.drop(labels="Address",axis=1, inplace=True)
df

# Visualise the data -----------------------------------------------
