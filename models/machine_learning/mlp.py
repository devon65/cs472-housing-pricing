import pandas as pd 
import numpy as np 
from numpy.polynomial.polynomial import polyfit 
from data.loader import HousingDataset as hd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn import linear_model 
from itertools import product
#from sklearn.ensemble import GradientBoostingClassifier                        
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#df = data.data.groupby(["YEAR",'AREA']).HOUSING_INDEX.sum().to_frame()

occ_code = "00-0000"
data = hd([occ_code])

df = data.data
lst = ['YEAR', 'AREA', 'TOT_EMP', 'EMP_PRSE', 'H_MEAN', 'A_MEAN', 'MEAN_PRSE', 'H_PCT10', 'H_PCT25', 'H_MEDIAN', 'H_PCT75', 'H_PCT90', 'A_PCT10', 'A_PCT25', 'A_MEDIAN', 'A_PCT75', 'A_PCT90']

X = df[['YEAR','AREA']]
y = np.array(df[['HOUSING_INDEX']]).ravel()

for yr in range(2004,2013):#first year 2004, last 2018, in 5-year increments,
    df_tmp = df.loc[df['YEAR'].between(yr, yr+5)] #get 5-year window of data
#    df_tmp = df_tmp[df_tmp[lst].notnull()]
    X = df_tmp[lst]
    y = np.array(df_tmp.HOUSING_INDEX).ravel()
    test = df.loc[(df['YEAR'] == yr+6)]
    X_train, y_train, X_test, y_test = X,y,test[lst],np.array(test.HOUSING_INDEX).ravel()

    model = MLPRegressor(learning_rate='adaptive',max_iter=1000,hidden_layer_sizes=(40,40))

    model.fit(X_train,y_train)
    print(f"For model starting at {yr}, accuracy is: {model.score(X_test,y_test)}")

yr = 2004
df_tmp = df.loc[df['YEAR'].between(yr, yr+13)] #get 5-year window of data
#    df_tmp = df_tmp[df_tmp[lst].notnull()]
X = df_tmp[lst]
y = np.array(df_tmp.HOUSING_INDEX).ravel()
test = df.loc[(df['YEAR'] == yr+14)]
X_train, y_train, X_test, y_test = X,y,test[lst],np.array(test.HOUSING_INDEX).ravel()

model = MLPRegressor(learning_rate='adaptive',max_iter=1000,hidden_layer_sizes=(40,40))

model.fit(X_train,y_train)
predicted = model.predict(X_test)
for i in range(len(predicted)):
    diff = predicted[i]-y_test[i]
    print("diff is ", diff/y_test[i])
    print(f"predicted {predicted[i]} real was {y_test[i]}")
print("predicted is ", predicted)
print(f"For model over all years, accuracy is: {model.score(X_test,y_test)}")



#y_pred = model.predict(X_test)

#print(y_pred.shape, X_test.shape,y_test.shape)
#plt.scatter(X_test[0],y_test,color='b')
#plt.scatter(X_test[0],y_pred,color='k',alpha=.5)
#plt.show()
