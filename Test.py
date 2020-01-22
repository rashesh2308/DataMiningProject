import numpy as np
import  pandas as pd
import pickle
from collections import Counter
from sklearn import  svm, neighbors
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

def process_data_for_label(ticker):
#We're predicting prices for the 3 months.

    hm_month= 3
#everything in df
    df = pd.read_csv('SP500CombinedData.csv' , index_col= 0)
#just the column names in tickers
    tickers = df.columns.values.tolist()
    df.fillna(0,inplace= True)
#This will calculate the % of shift happened in each month for 3 months
    for i in range(1, hm_month + 1):
        df['{}_{}M'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0,inplace=True)
    print(df.head())
    return tickers,df

#process_data_for_label('BAC')

#This is where we're creating a buckets for Buy Sell and Hold. We're setting 7%
def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.10
    for col in cols :
        # print(col)
        if col > requirement:
            return 1 #buy
        if col < -requirement:
            return -1 #sell
    return 0  #hold



def extract_featuresets(ticker):
    tickers ,df =  process_data_for_label(ticker)
    # print(tickers)
    # print(df.head())

    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
                                          df['{}_1M'.format(ticker)],
                                          df['{}_2M'.format(ticker)],
                                          df['{}_3M'.format(ticker)]
                                        ))


    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data Spread' , Counter(str_vals))

    df.fillna(0, inplace=True)

    df = df.replace([np.inf, -np.inf],np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change().replace([np.inf, -np.inf],0)
    # df_vals = df.replace([np.inf, -np.inf],0)
    df_vals.fillna(0, inplace= True)

    X  = df_vals.values
    y =  df['{}_target'.format(ticker)].values
    print(y)

    return  X,y,df

#extract_featuresets('BAC')

#('MMM')

def do_ml(ticker):
    X,y, df  = extract_featuresets(ticker)
    X_train , X_test ,y_train, y_test = train_test_split(X,y,test_size=0.25)

    clf = neighbors.KNeighborsClassifier(weights='distance')

    clf.fit(X_train,y_train)

    print("\n\n")
    print("Parameters of Kneighbors",clf.get_params())
    confidence = clf.score(X_test,y_test)
    print("Accuracy of Kneighbors", confidence)
    predicition = clf.predict(X_test)
    print("Predicted Spread of Kneighbors:" , Counter(predicition))
    print("\n\n")


    print("Decision Tree")
    clf1 = DecisionTreeClassifier(max_depth=4)
    clf1.fit(X_train, y_train)
    print("Parameters of Decision Tree" , clf1.get_params())
    print("Accuracy of Decision Tree" , clf1.score(X_test,y_test))
    print("Predicted Spread of Decision Tree" , Counter(clf1.predict(X_test)))
    print("\n\n")


    print("RandomForest")
    clf2 = RandomForestClassifier()
    clf2.fit(X_train,y_train)
    print("Parameters of RandomForest", clf2.get_params())
    print("Accuracy of RandomForest", clf2.score(X_test, y_test))
    print("Predicted Spread of RandomForest", Counter(clf2.predict(X_test)))


    print("Ensemble")
    clfn = VotingClassifier([('lsvc', svm.LinearSVC()) ,
                             ('knn', neighbors.KNeighborsClassifier() ),
                             ('rfor', RandomForestClassifier() )])

    clfn .fit(X_train, y_train)
    confidence = clfn.score(X_test, y_test)
    print("Accuracy of Ensembles", confidence)
    predicition = clfn.predict(X_test)

    print("Predicted Spread of ensembles:", Counter(predicition))



    return confidence

do_ml('BAC')
#    rf_cv_score = cross_val_score(estimator=rf, X=xtrain, y=xtest, cv=5)















