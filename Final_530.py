import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom
import seaborn as sns
import numpy as np
from scipy.stats import pareto
from numpy import cov
from sklearn.linear_model import LinearRegression

# put data into dataframe
df = pd.read_csv("C:/Users/Matt/Desktop/SNAP_history.csv")

print(df.head(1))

#plot histograms along with mean, median and mode
print(df["Fiscal Year"].plot(kind='hist'))

print(df["Average Participation"].plot(kind='hist'))
print(df["Average Benefit Per Person"].plot(kind='hist'))
print(df["Total Benefits(M)"].plot(kind='hist'))
print(df["Other Costs"].plot(kind='hist'))
print(df["Total Costs(M)"].plot(kind='hist'))

#PMF
var = df["Total Benefits(M)"]
var2 = df["Average Participation"]
plt.ylabel('PMF')
sns.histplot(var, stat="probability", bins=20)

#CDF
#number of data points
n = 52
var3 = df["Average Benefit Per Person"]
count, bins_count = np.histogram(var3, bins=15)
sum = df["Average Benefit Per Person"].sum()
p = var3 / sum
cdf = np.cumsum(p)
plt.plot( var3)
plt.plot(cdf)

#Pareto Distribution

num = pareto.numargs
x, y = 4, 3
p = pareto(x, y)
q = np.arange(1, 2, 1)
r = pareto.rvs(x, y)
r2 = pareto.pdf(x, y, q)
dist = np.linspace(1,r2)
plot = plt.plot(dist, r.pdf(dist))

# Two Scatter Plots

df.plot.scatter(x="Average Participation", y="Fiscal Year", title = "Average Participation over time ")

df.plot.scatter(x="Total Costs(M)", y="Fiscal Year", title="Costs over Time")

#Testing Covariance
x2 = df["Average Participation"]
y2 = df["Total Costs(M)"]
covariance = cov(x2, y2)
print(covariance)


#Linear Regression

model = LinearRegression()
model.fit(x2,y2)
model = LinearRegression().fit(x2, y2)
print(model)




