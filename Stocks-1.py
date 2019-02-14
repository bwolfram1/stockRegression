# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 16:55:55 2019

@author: brand
"""
import quandl
import pandas as pd
import numpy as np
import matplotlib as plt
import datetime

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm

tsla = quandl.get("WIKI/KO", start_date="2015-01-01", end_date="2018-03-27")

tsla = tsla[['Adj. Close']]

#plt.pyplot.plot(tsla)

forecast_out = int(1)
tsla['Prediction'] = tsla[['Adj. Close']].shift(-forecast_out)
x = np.array(tsla.drop(['Prediction'], 1))
x = preprocessing.scale(x)
x_forecast = x[-forecast_out:]
x = x[:-forecast_out]

y = np.array(tsla['Prediction'])
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size = 0.2)

clf = LinearRegression()
clf.fit(X_train,y_train)

confidence = clf.score(X_test, y_test)
print("confidence: ", confidence)

forecast_prediction = clf.predict(x_forecast)
print(forecast_prediction)