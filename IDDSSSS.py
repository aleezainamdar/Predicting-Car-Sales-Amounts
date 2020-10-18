# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# Importing the dataset
car_df = pd.read_csv('Car_Purchasing_Data.csv' , encoding='ISO-8859-1')
x = car_df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis = 1)
y = car_df['Car Purchase Amount'] 

sns.boxplot(x=x['Annual Salary'])
sns.boxplot(x=x['Credit Card Debt'])
sns.boxplot(x=x['Net Worth'])

gender = [0,1]
LABELS = ["Female", "Male"]
plt.bar(x['Gender'], y)
plt.xticks(gender, LABELS)
plt.xlabel("Gender")
plt.ylabel("Car Purchase Amount")
plt.show()


plt.bar(x['Age'], y)
plt.xlabel("Age")
plt.ylabel("Car Purchase Amount")
plt.show()

y = y.values.reshape(-1,1)
  
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling , MinMaxScaler(feature_range = (0, 1)) will transform each value in the column proportionally within the range [0,1]. ... StandardScaler() will transform each value in the column to range about the mean 0 and standard deviation 1, ie, each value will be normalised by subtracting the mean and dividing by standard deviation
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

print('Variance score: %.2f' % regressor.score(X_test, y_test))

y_pred.mean()
#Hypothesis Testingimport pandas as pd
#h0="the average price of the predicted and the actual price is similar"
#h1="the average price of the predicted and the actual price is not similar"

from statsmodels.stats import weightstats as stests
ztest ,pval = stests.ztest(y_pred, x2=y_train, value=0,alternative='two-sided')
print(float(pval))
if pval<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")