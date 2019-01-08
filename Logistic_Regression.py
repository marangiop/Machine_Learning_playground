# -*- coding: utf-8 -*-
"""
In this project we are working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. Our aim is to create a model that will predict whether or not they will click on an ad based off the features of that user.

This data set contains the following features:
* 'Daily Time Spent on Site': consumer time on site in minutes
* 'Age': cutomer age in years
* 'Area Income': Avg. Income of geographical area of consumer
* 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
* 'Ad Topic Line': Headline of the advertisement
* 'City': City of consumer
* 'Male': Whether or not consumer was male
* 'Country': Country of consumer
* 'Timestamp': Time at which consumer clicked on Ad or closed window
* 'Clicked on Ad': 0 or 1 indicated clicking on Ad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ad_data = pd.read_csv('advertising_dataset.csv') #Read dataset into a Pandas dataframe

#Part 1: Exploratory Data Analysis
ad_data['Age'].plot.hist(bins=30) #Create a histogram of the age
sns.jointplot(x='Age', y='Area Income', data=ad_data) #Create a jointplot showing Area Income versus Age
sns.jointplot(x='Age', y= 'Daily Time Spent on Site', data=ad_data, kind='kde', color='red') #Create a jointplot showing the kde distributions of Daily Time spent on side vs. Age
sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data= ad_data) #Create a joint plot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'

sns.pairplot(ad_data, hue='Clicked on Ad') # Create a pairplot of all the dataset features, with a hue that allows to devide dataset based on whether the user clicked on the ad or not 

"""
Conclusions
From the last jointplot Daily Internet Usage against Daily Time Spent on Site, we can see that there are two clusters in our data: one with compatively low Daily Time Spent on Site and low Daily Internet Usage, and one with high Daily Time Spent on Site and high Daily Internet Usage
"""

#Part 2: Creation of Training and Testing Data 
y = ad_data['Clicked on Ad'] # This is what we are trying to predict
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']] # This are the numerical features we use for prediction; we only include columns holding numerical features
               
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) #Split the data into training and testing sets

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression() #Create an instance  of a LogisticRegression() model
logmodel.fit(X_train, y_train) #Train logmodel on the training data

predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

"""
Conclusions
We get pretty good precision, recall and accuracy. We only have a few mislabelled points, which is acceptable given the size of our datases.
""




