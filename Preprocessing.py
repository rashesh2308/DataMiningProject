import numpy as np
import  pandas as pd
import pickle
from collections import Counter
from sklearn import  svm, neighbors
from sklearn.model_selection import cross_val_score,train_test_split

from sklearn.ensemble import VotingClassifier, RandomForestClassifier

def process_data_for_label(ticker):
    hm_month= 3
    df = pd.read_csv('SP500CombinedData.csv' , index_col= 0)
    tickers = df.columns.values.tolist()
    df.fillna(0,inplace= True)

    for i in range(1, hm_month + 1):
        df['{}_{}M'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0,inplace=True)
    print(df.columns)
    return tickers,df


def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.07
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

    # print(df.columns)
    # print(df.CERN_6d.value_counts())
    # print(df.CERN_target.value_counts())

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

# extract_featuresets('XOM')

#('MMM')

def do_ml(ticker):
    X,y, df  = extract_featuresets(ticker)
    X_train , X_test ,y_train, y_test = train_test_split(X,y,test_size=0.25)

    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train,y_train)
    print(clf.get_params())
    confidence = clf.score(X_test,y_test)
    print("Accuracy of Kneighbors", confidence)
    predicition = clf.predict(X_test)

    print("Predicted Spread of Kneighbors:" , Counter(predicition))


    #
    clfn = VotingClassifier([('lsvc', svm.LinearSVC()) ,
                             ('knn', neighbors.KNeighborsClassifier() ),
                             ('rfor', RandomForestClassifier() )])

    clfn .fit(X_train, y_train)
    confidence = clfn.score(X_test, y_test)
    print("Accuracy of ensembles", confidence)
    predicition = clfn.predict(X_test)

    print("Predicted Spread of ensembles:", Counter(predicition))



    return confidence

do_ml('BAC')
#    rf_cv_score = cross_val_score(estimator=rf, X=xtrain, y=xtest, cv=5)















