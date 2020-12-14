import numpy as np
import plotly.express as px
import plotly.graph_objects as go 
from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.read_csv("ShampooData.csv")
X = df.values.reshape(-1,1)


model = LinearRegression()

model.fit(X, df)


x_range = np.linspace(X.min(), X.max(), 100)
y_range = model.predict(x_range.reshape(-1,1))

fig = px.scatter(df, x="sales", y="month", opacity = 0.65)
fig.add_traces(go.Scatter(x=x_range, y=y_range, name = 'Regression Fit'))
fig.show