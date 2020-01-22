import bs4 as bs
import requests
import datetime as dt
import time
import quandl
quandl.ApiConfig.api_key = "VYsf7X4RBp3sgNRYCB61"
from pandas.tseries.offsets import BMonthEnd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import  numpy as np


# Need to install lxml

def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S&P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        #ticker_name = row.findAll('td')[1].text
        #ticker_sliced = "WIKI/" + ticker.rstrip('\r\n') + "\n"
        ticker_sliced =  ticker
        tickers.append(ticker_sliced)

    file1 = open("SP500List.txt",'w')
    for f in tickers:
        file1.write(f)

    print(tickers)
    return tickers


# save_sp500_tickers()
#
# Run the first function first, and then run the second get_data_from_quandl()
# For few companies there'll be errors. So just go to SP500List.txt and delete that entry
#Run the code again

def get_data_from_quandl(reload_sp500 = False):
    tickers = []
    not_wanted_stocks = ["AMCR\n", "ANET\n", "BKR\n", "BRK.B\n", "BF.B\n", "CPRI\n", "CDW\n", "CTVA\n", "EVRG\n", "GL\n", "PEAK\n", "KEYS\n", "LHX\n", "LW\n", "LIN\n", "NLOK\n"]
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        file2 = open("SP500List.txt", 'r')
    for f in file2:
        if f not in not_wanted_stocks:
            tickers.append(f)


    print(tickers)

#'Open','High','Low','Close','Volume'

    for ticker in tickers:
            print(ticker)
        # time.sleep(1)
            df = quandl.get("WIKI/"+ticker.rstrip("\n") , start_date="2010-01-01", end_date="2019-11-05", collapse="monthly")
            df.to_csv('{}.csv'.format(ticker.rstrip("\n")))

# get_data_from_quandl()
# At first, tried to take daily data, but i makes sense to take monthly data for such large dataset

def compile_data():
    tickers = []
    not_wanted_stocks = ["AMCR\n", "ANET\n", "BKR\n", "BRK.B\n", "BF.B\n", "CPRI\n", "CDW\n", "CTVA\n", "EVRG\n",
                         "GL\n", "PEAK\n", "KEYS\n", "LHX\n", "LW\n", "LIN\n", "NLOK\n"]
    file1 =  open('sp500List.txt','r')
    for f in file1:
        if f not in not_wanted_stocks:
            print(f)
            tickers.append(f[:-1])

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv('{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df.rename(columns = {'Adj. Close': ticker}, inplace=True )
        df.drop(['Open', 'High' ,'Low' ,'Close' ,'Volume', 'Ex-Dividend','Split Ratio','Adj. Open','Adj. High','Adj. Low', 'Adj. Volume'],1 , inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count%10 == 0:
            print(count)

    print(main_df.head())
    main_df.to_csv('SP500CombinedData.csv')


# compile_data()

def visualize_data():
    df = pd.read_csv('SP500CombinedData.csv')

    df_corr = df.corr()
    # print(df_corr)

    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    heatmap = ax.pcolor(data, cmap = plt.cm.RdYlGn)
    fig.colorbar(heatmap)

    ax.set_xticks(np.arange(data.shape[0] + 0.5))
    ax.set_yticks(np.arange(data.shape[1] + 0.5))

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_label  = df_corr.columns
    row_label = df_corr.columns

    ax.set_xticklabels(column_label)
    ax.set_yticklabels(row_label)
    plt.xticks(rotation  = 90)
    heatmap.set_clim(-1,1)

    plt.tight_layout()
    plt.show()


visualize_data()




# compile_data()

def visualize_data():
    df = pd.read_csv('SP500CombinedData.csv')

    df_corr = df.corr()
    # print(df_corr)

    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    heatmap = ax.pcolor(data, cmap = plt.cm.RdYlGn)
    fig.colorbar(heatmap)

    ax.set_xticks(np.arange(data.shape[0] + 0.5))
    ax.set_yticks(np.arange(data.shape[1] + 0.5))

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_label  = df_corr.columns
    row_label = df_corr.columns

    ax.set_xticklabels(column_label)
    ax.set_xticklabels(row_label)
    plt.xticks(rotation  = 90)
    heatmap.set_clim(-1,1)

    plt.tight_layout()
    plt.show()


visualize_data()








