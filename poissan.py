#!/usr/bin/env python3
# correlation between number of boats and manta presence at sunset or moonlight 

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load dataset
data = pd.read_csv("Documents/pythonprojects/Mantadata1.csv")

# columns for presence of boats, no = boat presence 1, yes = boat presence 2 
data['yboat'] = data['Boats']
data['yboat'].replace({1:1, 2:0}, inplace=True)
data.head()

data['nboat'] = data['Boats']
data['nboat'].replace({1:0, 2:2}, inplace=True)
data.head()

# columns for sunset verses moonlight manta tours 
data['sunset'] = data['Tour']
data['sunset'].replace({1:1, 2:0}, inplace=True)
data.head()

data['moonlight'] = data['Tour']
data['moonlight'].replace({1:0, 2:2}, inplace=True)
data.head()

data['high'] = data["High tide"]
data.head()

data['low'] = data["Low tide"]
data.head()

data['plankton5'] = data['Plankton']
data['plankton5'].replace({1:0, 2:0, 3:0, 4:0, 5:5}, inplace=True)
data.head()

data['plankton4'] = data['Plankton']
data['plankton4'].replace({1:0, 2:0, 3:0, 4:4, 5:0}, inplace=True)
data.head()

data['plankton3'] = data['Plankton']
data['plankton3'].replace({1:0, 2:0, 3:3, 4:0, 5:0}, inplace=True)
data.head()

#print(data.head())

data['high'] = data['high'].fillna(0)
data['Mantas'] = data['Mantas'].fillna(0)
data['plankton5'] = data['plankton5'].fillna(0)
data['plankton4'] = data['plankton4'].fillna(0)
data['plankton3'] = data['plankton3'].fillna(0)

#X = df[['Boats', 'Plankton', 'Moon Phase', 'High tide', 'Low tide']]
x_train, x_test, y_train, y_test = train_test_split(data[["nboat" , "high", 'low', 'moonlight','plankton5', 'plankton4', 'plankton3']], data[["Mantas"]], test_size=0.2, shuffle=True)

pois = PoissonRegressor()
pois.fit(x_train, y_train)

train2 = pd.concat([x_train, y_train], axis=1)
train2.head()

pois_reg = smf.poisson("Mantas ~ nboat + moonlight + high + low + plankton5 + plankton4 +plankton3", data=train2).fit()

print ('coefficient: ', pois.coef_)
print ('intercept: ', pois.intercept_)
#print(data.isnull().sum())

y_pred = pois.predict(x_test) # need to find the error, difference between actual and predict
y_test = y_test.to_numpy()
error = np.subtract(y_test, y_pred)
print ("error:" ,error.mean())
print ('Summary: ',pois_reg.summary())


plt.plot(error)
plt.xlabel('Residuals')
plt.axhline(y=0, color='black', linestyle='-')
plt.show()
# = .144, off by .1 or so #!/usr/bin/env python3





