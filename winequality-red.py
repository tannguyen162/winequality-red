#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')

dataset = pd.read_csv('winequality-red.csv')
dataset.shape
dataset.describe()
dataset.isnull().any()
dataset = dataset.fillna(method='ffill') #fixed null
X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']].values
y = dataset['quality'].values

# Let's check the average value of the "quanlity" column
# plt.figure(figsize=(15,10))
# plt.tight_layout()  #khit layout
# seabornInstance.distplot(dataset['quality'])

#split 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

# # train model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# show coefficients (he so toi uu) has choesen
# coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
# coeff_df

# predict
y_pred = regressor.predict(X_test)

# check between the actual value & predicted value
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)
df1


#Let' plot comparison of Actual and Predicted values
df1.plot(kind='bar', figsize=(10,8))
plt.grid(which='major', linestyle=':', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle='-', linewidth='0.5', color='black')
plt.show()


# MAE, MSE, RMSE
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:





# In[ ]:




