import pandas as pd
def formatData(df):
    df["PCT_CHG"] = (df["Close"] - df["Open"]) / df["Open"]
    df["HL_CHG"] = (df["High"] - df["Low"]) / df["Low"]
    df["HC_CHG"] = (df["High"] - df["Close"]) / df["Close"]
    df["LC_CHG"] = (df["Low"] - df["Close"]) / df["Close"]
    df["HO_CHG"] = (df["High"] - df["Close"]) / df["Close"]
    df["LO_CHG"] = (df["Low"] - df["Close"]) / df["Close"]
    #df["ADJ_CHG"] = (df["Close"] - df["Adj Close"]) / df["Adj Close"]
    df["prevClose"] = df["Close"].shift(1)
    df["prevVolume"] = df["Volume"].shift(1)
    df = df[1:]
    df["AftMrkt_CHG"] = (df["Open"] - df["prevClose"]) / df["prevClose"]
    df["VOL_CHG"] = (df["Volume"] - df["prevVolume"]).divide(df["prevVolume"])
    df['Date'] = pd.to_datetime(df['Date'])
    df['DAYOFWEEK'] = df['Date'].dt.dayofweek.astype(int)
    df["IS_START_OF_MONTH"] = df['Date'].dt.is_month_start.astype(int)
    df["IS_END_OF_MONTH"] = df["Date"].dt.is_month_end.astype(int)
    df["IS_QUARTER_START"] = df["Date"].dt.is_quarter_start.astype(int)
    df["IS_QUARTER_END"] = df["Date"].dt.is_quarter_end.astype(int)
    df["MONTH"] = df["Date"].dt.month.astype(int)
    df["DAY"] = df["Date"].dt.day.astype(int)
    return df

def addData(main_df):
    df1 = pd.read_csv("./csv/NQ=F.csv")
    df2 = pd.read_csv("./csv/YM=F.csv")
    df3 = pd.read_csv("./csv/ES=F.csv")
    dfs = [df1,df2,df3]
    index = 0
    for i in dfs:
        i = i.dropna()
        i = i.reset_index(drop=True)
        i = formatData(i.copy())
        i = i[["Date","PCT_CHG","HL_CHG", "HC_CHG","LC_CHG","HO_CHG","LO_CHG","AftMrkt_CHG","VOL_CHG","DAYOFWEEK","IS_START_OF_MONTH","IS_END_OF_MONTH","IS_QUARTER_START","IS_QUARTER_END","MONTH","DAY"]]
        i = i.add_suffix("_" + str(index))
        #i = i[-200:]
        dfs[index] = i
        index += 1
    df = pd.merge(left=dfs[2], left_on='Date_2',
            right=dfs[1], right_on='Date_1')
    df = pd.merge(left=dfs[0], left_on='Date_0',
            right=df, right_on='Date_1')
   
    df.to_csv("./test3.csv")
    main_df = pd.merge(left=main_df, left_on='Date',
            right=df, right_on='Date_1')
    main_df = main_df.reset_index(drop=True)
    main_df = main_df.drop(["Date_0", "Date_1", "Date_2"], axis=1)
    return main_df
    

