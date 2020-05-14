from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import dataFormatter as dataform
from sklearn import ensemble
from sklearn import tree
from sklearn import svm
from sklearn.multioutput import RegressorChain
predictionLabels = ["PCT_CHG","AftMrkt_CHG","HO_CHG"]
def findNextTick(df, type):
    nextStrings = []
    #Creating a column for next value (This is what we are predicting)
    for i in predictionLabels:
        nextString = "next" + str(i)
        df[nextString] = df[i].shift(-1)
        nextStrings.append(nextString)

    X_pred = df[-1:].drop(nextStrings, axis=1) #Setting up a variable for prediction.
    df = df[0:-1] #Taking all but the last value for training
    X = df.drop(nextStrings, axis=1) #Dropping the answers
    y = df[nextStrings] #Creating an answer list
    r1 = LinearRegression(n_jobs=-1)
    r2 = tree.DecisionTreeRegressor()
    r3 = ensemble.RandomForestRegressor(n_jobs=-1)
    estimators = [
       ('r1', r1),
       ('r2', r2),
       ('r3', r3)
    ]
    if(type == 0):
        regressor = ensemble.StackingRegressor(
            estimators=estimators,
            final_estimator=ensemble.RandomForestRegressor(n_estimators=100,
                                                  random_state=42, n_jobs=-1)
        )
        
    elif(type == 1):
        regressor = ensemble.VotingRegressor(
            estimators=estimators
        )
        print("I got here!")
    regressor = RegressorChain(regressor)
    regressor.fit(X, y) #training the algorithm
    y_pred = list(regressor.predict(X_pred))

    y_pred.insert(0,X_pred.iloc[0][predictionLabels])
    y_pred = np.asarray(y_pred)
    x_predTime = list(X_pred.index)
    x_predTime.append(x_predTime[0] + 1)
    x_predTime = np.asarray(x_predTime)
    print(y_pred)
    print(x_predTime)
    return {"Y":y_pred,"X":x_predTime}

df = pd.read_csv("./csv/TSLA.csv")
df = dataform.formatData(df)
#df = dataform.addData(df.copy())
df = df.replace([np.inf,-np.inf,np.nan], 0)

df = df.drop(["Close", "Volume", "High", "Low", "Adj Close", "Open", "Date", "prevVolume", "prevClose"], axis=1)
df.to_csv("test.csv")

df=(df-df.mean())/df.std()
df.to_csv("./test.csv")
X_predTime = []
Y_pred = []
fig, ax = plt.subplots(nrows=3, ncols=1)
for i in range(len(predictionLabels)):
    ax[i].plot(df[predictionLabels[i]][-20:])

#.plot()

predictions = [[],[],[]]
#actuals = list(df[predictionLabel][-20:])
indices = list(df[-5:].index)
for i in range(5):
    for b in range(3):
        if(b == 2):
            #Averaging the values.
            predictions[2].append((np.array(predictions[0]) + np.array(predictions[1])) / 2)
            break
        data = findNextTick(df[0:-(i + 1)].copy(),b)
        if(b == 0):
            predictions[0].append(data["Y"][1])

            for c in range(len(predictionLabels)):
                ax[c].plot(data["X"], [data["Y"][0][c],data["Y"][1][c]], color="green")
        elif(b == 1):
            predictions[1].append(data["Y"][1])
            for c in range(len(predictionLabels)):
                ax[c].plot(data["X"], [data["Y"][0][c],data["Y"][1][c]], color="red")

final_predictions = [[],[],[]]
for i in predictions[2]:

    index = 0
    final_predictions[0].append(i[0][0])
    final_predictions[1].append(i[0][1])
    final_predictions[2].append(i[0][2])
    index += 1
for i in range(len(final_predictions)):
    final_predictions[i] = final_predictions[i][::-1] #Reversing the averages array
    ax[i].scatter(indices,final_predictions[i], color="orange")
    
plt.show()