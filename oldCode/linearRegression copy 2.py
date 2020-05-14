from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from vispy import scene, app, use
from sklearn import ensemble
from sklearn import tree
use('PyQt5')

def findNextTick(df, type):
    df["nextClose"] = df["High"].shift(-1)
    #df["nextTime"] = df["time"].shift(-1)
    df["nextIndex"] = df.index
    df["nextIndex"] = df["nextIndex"].shift(-1)
    df.at[len(df)-1, 'nextIndex'] = df.iloc[len(df) - 2]["nextIndex"] + 1
    df = df[0:len(df) - 2]
    #df.to_csv("test3.csv")
    X_pred = df[-1:].drop(["nextClose"], axis=1)
    print(X_pred)
    df = df[0:-1]
    X = df.drop(["nextClose"], axis=1)
    #X.to_csv("test4.csv")
    y = df["nextClose"]
    r1 = LinearRegression(n_jobs=-1)
    r2 = tree.DecisionTreeRegressor()
    r3 = ensemble.RandomForestRegressor(n_jobs=-1)
    r4 = ensemble.AdaBoostRegressor()
    r5 = ensemble.BaggingRegressor(n_jobs=-1)
    r6 = ensemble.GradientBoostingRegressor()
    estimators = [
       ('r1', r1),
       ('r2', r2),
       ('r3', r3),
       ('r4', r4),
       ('r5', r5),
       ('r6',r6)
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
    y_pred.insert(0,X_pred.iloc[0]["High"])
    y_pred = np.asarray(y_pred)
    x_predTime = list(X_pred.index)
    x_predTime.append(x_predTime[0] + 1)
    x_predTime = np.asarray(x_predTime)
    print(y_pred)
    print(x_predTime)
    return {"Y":y_pred,"X":x_predTime}
def formatData(df):
    df["PCT_CHG"] = (df["Close"] - df["Open"]) / df["Open"]
    df["HL_CHG"] = (df["High"] - df["Low"]) / df["Low"]
    #df["VOLSHIFT"] = df["volume"].shift(1)
    #df["VOL_CHG"] = (df["volume"] - df["VOLSHIFT"]) / df["VOLSHIFT"]
    #df = df[1:]
    return df
df = pd.read_csv("./AMD2.csv")
#df = df.sort_values(["Date"])
#df = df.reset_index(drop=True)
df = formatData(df)
df=(df-df.mean())/df.std()
df["PctStd"] = df["PCT_CHG"].rolling(4).std()
df["HLStd"] = df["HL_CHG"].rolling(4).std()
df["PctMean"] = df["PCT_CHG"].rolling(4).mean()
df["HLMean"] = df["HL_CHG"].rolling(4).mean()
df = df[5:-30]
df = df[["Close", "High", "Low", "PCT_CHG","HL_CHG", "Volume", "Date"]]
df.to_csv("./test.csv")
#df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
#df.to_csv("test.csv")
X_predTime = []
Y_pred = []
graph = Graph()
positions = np.array(list(zip(df.index, df["High"])))
plot = scene.Line(positions, parent=graph.view.scene,antialias=True,width=4)
for i in range(20):
    for b in range(2):
        data = findNextTick(df[0:-(i + 1)].copy(),b)
        positions2 = np.array(list(zip(data["X"], data["Y"])))
        if(b == 0):
            plot = scene.Line(positions2, parent=graph.view.scene,antialias=True,color="red")
        elif(b == 1):
            plot = scene.Line(positions2, parent=graph.view.scene,antialias=True,color="blue")
        elif(b == 2):
            plot = scene.Line(positions2, parent=graph.view.scene,antialias=True,color="green")
        elif(b == 3):
            plot = scene.Line(positions2, parent=graph.view.scene,antialias=True,color="purple")
        elif(b == 4):
            plot = scene.Line(positions2, parent=graph.view.scene,antialias=True,color="orange")
        elif(b == 5):
            plot = scene.Line(positions2, parent=graph.view.scene,antialias=True,color="yellow")
        elif(b == 6):
            plot = scene.Line(positions2, parent=graph.view.scene,antialias=True,color="pink")


#positions3 = np.array(list(zip(dfCopy.nextIndex, dfCopy.nextClose)))


#plot = scene.Line(positions3, parent=graph.view.scene,antialias=True,color="green")
graph.draw()