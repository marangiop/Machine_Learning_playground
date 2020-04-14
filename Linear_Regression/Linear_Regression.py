# -*- coding: utf-8 -*-

"""
Case study
You just got some contract work with an Ecommerce company based in New York City that sells clothing online but they also have in-store style and clothing advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist, then they can go home and order either on a mobile app or website for the clothes they want.
The company is trying to decide whether to focus their efforts on their mobile app experience or their website. They've hired you on contract to help them figure it out!
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

customers = pd.read_csv('Ecommerce Customers.csv') #Read dataset into a dataframe   

# Part 1: Exploratory Data Analysis

sns.jointplot(data=customers, x='Time on Website', y='Yearly Amount Spent') #Create a jointplot to compare the Time on Website and Yearly Amount Spent
sns.jointplot(data=customers, x='Time on App', y='Yearly Amount Spent') #Create a jointplot to compare the Time on App and Yearly Amount Spent
sns.pairplot(customers) #Explore pairwise relationships across the entire dataset
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers) # Create a linear model plot of Yearly Amount Spent against Length of Membership
plt.show()
"""
Conclusions
There seems to be a relationship between Time on App and Yearly Amount Spent, while it does not appear that there is a relationship between Time on Website and Yearly Amount Spent. 
Length of memberships appears to be the most correlated featured with Yearly Amount Spent. 
"""

#Part 2: Creation of Training and Testing Data 
 
y= customers['Yearly Amount Spent'] # This is what we are trying to predict
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']] # This are the numerical features we use for prediction; we only include columns holding numerical features
               
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) #Split the data into training and testing sets


#Part 3: Train the Model 

from sklearn.linear_model import LinearRegression

lm=LinearRegression()#Create an instance  of a LinearRegression() model
lm.fit(X_train, y_train) #Train lm on the training data

#Part 4a: Evaluate the Performance of the model by predicting Test Data

predictions = lm.predict(X_test) #Use the trained model to predict off the X_test set; the result is passed to an array called predictions

plt.scatter(y_test, predictions)
plt.xlabel('Y test (True Values)')
plt.ylabel('Predicted Values')

#Part 5a: Evaluate the Performance by calculating metrics and residual

from sklearn import metrics

print('MAE ', metrics.mean_absolute_error(y_test, predictions)) #Mean Absolute Error
print('MSE ', metrics.mean_squared_error(y_test, predictions)) #Mean Squared Error
print('RMSE ', np.sqrt(metrics.mean_squared_error(y_test, predictions))) #Root Mean Squared Error
print('R^2 ', metrics.explained_variance_score(y_test, predictions)) #Percentage of variance that is explained by the model


sns.distplot((y_test-predictions), bins=50) #Plot a histogram of the residuals and make sure it is normally distributed

cdf = pd.DataFrame(lm.coef_, X.columns, columns=['Coeff']) #Each feature is associated with a different increase in the Yearly Amount Spent. The strength of each feature is conveyed by its coefficient.



"""
Conclusion: Length of membership is associated with the highest increase in Yearly Amount Spent. Since Time on App is the second strongest feature, which means that the aApp is working much better than the website, it makes more sense to focuse more efforts on the website.
   
"""       