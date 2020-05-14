from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import dataFormatter as dataform
#import daal4py.sklearn
#daal4py.sklearn.patch_sklearn()
#from vispy import scene, app, use
from sklearn import ensemble
from sklearn import tree
from sklearn import svm
#use('PyQt5')
predictionLabel = "PCT_CHG"
def findNextTick(df, type):
    df["nextClose"] = df[predictionLabel].shift(-1) #Creating a column for next value (This is what we are predicting)
    #df["nextIndex"] = df.index #Creating a new index column
    #df["nextIndex"] = df["nextIndex"].shift(-1) #Shifting new index column back
    #df.at[len(df)-1, 'nextIndex'] = df.iloc[len(df) - 2]["nextIndex"] + 1
    #df = df[0:len(df) - 2] #
    X_pred = df[-1:].drop(["nextClose"], axis=1) #Setting up a variable for prediction.
    df = df[0:-1] #Taking all but the last value for training
    X = df.drop(["nextClose"], axis=1) #Dropping the answers
    y = df["nextClose"] #Creating an answer list
    r1 = LinearRegression(n_jobs=-1)
    r2 = tree.DecisionTreeRegressor()
    r3 = ensemble.RandomForestRegressor(n_jobs=-1)
    #r4 = svm.LinearSVR()
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
    regressor.fit(X, y) #training the algorithm
    y_pred = list(regressor.predict(X_pred))
    y_pred.insert(0,X_pred.iloc[0][predictionLabel])
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
#df = df.sort_values(["Date"])
#df = df.reset_index(drop=True)
df = df.drop(["Close", "Volume", "High", "Low", "Adj Close", "Open", "Date", "prevVolume", "prevClose"], axis=1)
df.to_csv("test.csv")

#df = df[["PCT_CHG","HL_CHG", "HC_CHG","LC_CHG","HO_CHG","LO_CHG","AftMrkt_CHG","VOL_CHG","DAYOFWEEK","IS_START_OF_MONTH","IS_END_OF_MONTH","IS_QUARTER_START","IS_QUARTER_END","MONTH","DAY"]]
df=(df-df.mean())/df.std()
df.to_csv("./test.csv")
X_predTime = []
Y_pred = []

df[predictionLabel][-20:].plot()
#plt.scatter(df.index,df["PCT_CHG"])
predictions = [[],[],[]]
actuals = list(df[predictionLabel][-20:])
indices = list(df[-20:].index)
for i in range(5):
    for b in range(3):
        if(b == 2):
            #Averaging the values.
            predictions[2].append((predictions[0][len(predictions[0]) - 1] + predictions[1][len(predictions[1]) - 1]) / 2)
            print("AVERAGE:")
            print(predictions[0][len(predictions[0]) - 1])
            print(predictions[1][len(predictions[1]) - 1])
            print((predictions[0][len(predictions[0]) - 1] + predictions[1][len(predictions[1]) - 1]) / 2)
            break
        data = findNextTick(df[0:-(i + 1)].copy(),b)
        if(b == 0):
            predictions[0].append(data["Y"][1])
            plt.plot(data["X"], data["Y"], color="green")
        elif(b == 1):
            predictions[1].append(data["Y"][1])
            plt.plot(data["X"], data["Y"], color="red")
predictions[2] = predictions[2][::-1] #Reversing the averages array

plt.scatter(indices,predictions[2], color="orange")
#positions3 = np.array(list(zip(dfCopy.nextIndex, dfCopy.nextClose)))


#plot = scene.Line(positions3, parent=graph.view.scene,antialias=True,color="green")
plt.show()