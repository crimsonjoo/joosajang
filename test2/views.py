from django.shortcuts import render
from urllib.request import urlopen, Request
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import numpy as np
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from mpl_finance import candlestick_ohlc
import mplfinance as mpf
import matplotlib.dates as mdates
import io
import urllib, base64

def index(request):
    return render(request,'test2/index.html')

def index2(request):
    return render(request,'test2/index2.html')

def question1(request):
    return render(request,'test2/question1.html')

def question2(request):
    return render(request, 'test2/question2.html')

def question3(request):
    return render(request, 'test2/question3.html')

def question4_0(request):
    return render(request, 'test2/question4_0.html')

def question4_1(request):
    return render(request, 'test2/question4_1.html')

def result(request,num):
    if num == 0:
        return render(request, 'test2/result0.html')
    elif num == 1:
        return render(request, 'test2/result1.html')
    elif num == 2:
        return render(request, 'test2/result2.html')
    elif num == 3:
        return render(request, 'test2/result3.html')
    elif num == 4:
        return render(request, 'test2/result4.html')
    elif num == 5:
        return render(request, 'test2/result5.html')

def detail(request,idx):  #여기선 결과물이 idx 0: 볼린저추세추종, 1:볼린저반전 ,2:삼중창(ema) ,3:삼중창(macd) , 4:골든&데드(5/20) ,5:골든&데드(20/60)를 의미한다.
    today = datetime.today()
    today_str = today.strftime("%Y-%m-%d")
    if idx == 0:
        queryDict=dict(request.GET)
        code = queryDict['code'][0]
        start_date = queryDict['date'][0]
        if start_date[4] != '-':
            start_date = str(start_date[:4]) + '-' + str(start_date[4:6]) + '-' + str(start_date[6:])

        df = fdr.DataReader(code,start_date)
        name_df = fdr.StockListing('KRX')
        name = name_df.loc[name_df['Symbol']==code,'Name'].values[0]

        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['stddev'] = df['Close'].rolling(window=20).std()
        df['upper'] = df['MA20'] + (df['stddev'] * 2)
        df['lower'] = df['MA20'] - (df['stddev'] * 2)
        df['PB'] = (df['Close'] - df['lower']) / (df['upper'] - df['lower'])
        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['PMF'] = 0
        df['NMF'] = 0
        for i in range(len(df.Close) - 1):
            if df.TP.values[i] < df.TP.values[i + 1]:
                df.PMF.values[i + 1] = df.TP.values[i + 1] * df.Volume.values[i + 1]
                df.NMF.values[i + 1] = 0
            else:
                df.NMF.values[i + 1] = df.TP.values[i + 1] * df.Volume.values[i + 1]
                df.PMF.values[i + 1] = 0
        df['MFR'] = (df.PMF.rolling(window=10).sum() /
                     df.NMF.rolling(window=10).sum())
        df['MFI10'] = 100 - 100 / (1 + df['MFR'])
        df = df[19:]

        plt.figure(figsize=(9, 8))
        plt.subplot(3, 1, 1)
        plt.title('Bollinger Band Trend Following')
        plt.plot(df.index, df['Close'], color='k', label='Close')
        plt.plot(df.index, df['upper'], 'r:', label='Upper band')
        plt.plot(df.index, df['MA20'], 'g-.', label='Moving average (20)')
        plt.plot(df.index, df['lower'], 'b:', label='Lower band')
        plt.fill_between(df.index, df['upper'], df['lower'], color='0.9')

        buy_list = []
        sell_list = []
        returns = 0
        own_price = 0
        for i in range(len(df.Close)):
            if df.PB.values[i] > 0.8 and df.MFI10.values[i] > 80:  # ①
                plt.plot(df.index.values[i], df.Close.values[i], 'r^')  # ②
                date = np.datetime_as_string(df.index.values[i], unit='D')
                cost = df.Close.values[i]
                if own_price == 0:
                    own_price += cost
                    buy_list.append([date, cost])
            elif df.PB.values[i] < 0.2 and df.MFI10.values[i] < 20:  # ③
                plt.plot(df.index.values[i], df.Close.values[i], 'bv')  # ④
                date = np.datetime_as_string(df.index.values[i], unit='D')
                cost = df.Close.values[i]
                if own_price != 0:
                    returns += ((cost / own_price) - 1) * 100
                    own_price = 0
                    sell_list.append([date, cost])
            if i == (len(df.Close) - 1):  # 매수했었는데 마지막 날까지 매도 신호가 안나왔다면 현재의 수익률을 환산
                date = np.datetime_as_string(df.index.values[i], unit='D')
                cost = df.Close.values[i]
                if own_price != 0:
                    returns += ((cost / own_price) - 1) * 100
                    own_price = 0
        plt.legend(loc='best')
        plt.subplot(3, 1, 2)
        plt.plot(df.index, df['PB'] * 100, 'g-', label='%B x 100')  # ⑤
        plt.yticks([-20, 0, 20, 40, 60, 80, 100, 120])  # ⑦
        for i in range(len(df.Close)):
            if df.PB.values[i] > 0.8 and df.MFI10.values[i] > 80:
                plt.plot(df.index.values[i], 0, 'r^')
            elif df.PB.values[i] < 0.2 and df.MFI10.values[i] < 20:
                plt.plot(df.index.values[i], 0, 'bv')
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.subplot(3, 1, 3)
        plt.plot(df.index, df['MFI10'], 'y-', label='MFI (10)')  # ⑥
        plt.grid(True)
        plt.legend(loc='upper left')

        fig = plt.gcf()
        buf = io.BytesIO()

        fig.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = 'data:image/png;base64,' + urllib.parse.quote(string)


        return render(request, 'test2/result0_2.html',{'uri':uri,'start':start_date,'today':today_str,'code':code,'name':name,'returns':int(returns),'buy_list':buy_list,'sell_list':sell_list,})
    elif idx == 1:
        queryDict = dict(request.GET)
        code = queryDict['code'][0]
        start_date = queryDict['date'][0]
        if start_date[4] != '-':
            start_date = str(start_date[:4]) + '-' + str(start_date[4:6]) + '-' + str(start_date[6:])

        df = fdr.DataReader(code, start_date)
        name_df = fdr.StockListing('KRX')
        name = name_df.loc[name_df['Symbol'] == code, 'Name'].values[0]

        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['stddev'] = df['Close'].rolling(window=20).std()
        df['upper'] = df['MA20'] + (df['stddev'] * 2)
        df['lower'] = df['MA20'] - (df['stddev'] * 2)
        df['PB'] = (df['Close'] - df['lower']) / (df['upper'] - df['lower'])

        df['II'] = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']
        df['IIP21'] = df['II'].rolling(window=21).sum() / df['Volume'].rolling(window=21).sum() * 100
        df = df.dropna()

        plt.figure(figsize=(9, 9))
        plt.subplot(3, 1, 1)
        plt.title('Bollinger Band Trend Reversals')
        plt.plot(df.index, df['Close'], 'k', label='Close')
        plt.plot(df.index, df['upper'], 'r:', label='Upper band')
        plt.plot(df.index, df['MA20'], 'g-.', label='Moving average (20)')
        plt.plot(df.index, df['lower'], 'b:', label='Lower band')
        plt.fill_between(df.index, df['upper'], df['lower'], color='0.9')

        buy_list = []
        sell_list = []
        returns = 0
        own_price = 0
        for i in range(0, len(df.Close)):
            if df.PB.values[i] < 0.05 and df.IIP21.values[i] > 0:  # ①
                plt.plot(df.index.values[i], df.Close.values[i], 'r^')  # ②
                date = np.datetime_as_string(df.index.values[i], unit='D')
                cost = df.Close.values[i]
                if own_price == 0:
                    own_price += cost
                    buy_list.append([date, cost])
            elif df.PB.values[i] > 0.95 and df.IIP21.values[i] < 0:  # ③
                plt.plot(df.index.values[i], df.Close.values[i], 'bv')  # ④
                date = np.datetime_as_string(df.index.values[i], unit='D')
                cost = df.Close.values[i]
                if own_price != 0:
                    returns += ((cost / own_price) - 1) * 100
                    own_price = 0
                    sell_list.append([date, cost])
            if i == (len(df.Close) - 1):  # 매수했었는데 마지막 날까지 매도 신호가 안나왔다면 현재의 수익률을 환산
                date = np.datetime_as_string(df.index.values[i], unit='D')
                cost = df.Close.values[i]
                if own_price != 0:
                    returns += ((cost / own_price) - 1) * 100
                    own_price = 0
        plt.legend(loc='best')

        plt.subplot(3, 1, 2)
        plt.plot(df.index, df['PB'], 'g', label='%B')
        plt.grid(True)
        plt.legend(loc='upper left')

        plt.subplot(3, 1, 3)
        plt.bar(df.index, df['IIP21'], color='y', label='II% (21)')
        for i in range(0, len(df.Close)):
            if df.PB.values[i] < 0.05 and df.IIP21.values[i] > 0:
                plt.plot(df.index.values[i], 0, 'r^')  # ⑤
            elif df.PB.values[i] > 0.95 and df.IIP21.values[i] < 0:
                plt.plot(df.index.values[i], 0, 'bv')  # ⑥
        plt.grid(True)
        plt.legend(loc='upper left')

        fig = plt.gcf()
        buf = io.BytesIO()

        fig.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = 'data:image/png;base64,' + urllib.parse.quote(string)

        return render(request, 'test2/result1_2.html',{'uri':uri,'start':start_date,'today':today_str,'code':code,'name':name,'returns':int(returns),'buy_list':buy_list,'sell_list':sell_list})
    elif idx == 2:
        queryDict = dict(request.GET)
        code = queryDict['code'][0]
        start_date = queryDict['date'][0]
        if start_date[4] != '-':
            start_date = str(start_date[:4]) + '-' + str(start_date[4:6]) + '-' + str(start_date[6:])

        df = fdr.DataReader(code, start_date)
        name_df = fdr.StockListing('KRX')
        name = name_df.loc[name_df['Symbol'] == code, 'Name'].values[0]

        ema60 = df.Close.ewm(span=60).mean()
        ema130 = df.Close.ewm(span=130).mean()
        macd = ema60 - ema130
        signal = macd.ewm(span=45).mean()
        macdhist = macd - signal
        df = df.assign(ema130=ema130, ema60=ema60, macd=macd, signal=signal, macdhist=macdhist).dropna()

        df['number'] = df.index.map(mdates.date2num)
        ohlc = df[['number', 'Open', 'High', 'Low', 'Close']]

        ndays_high = df.High.rolling(window=14, min_periods=1).max()
        ndays_low = df.Low.rolling(window=14, min_periods=1).min()

        fast_k = (df.Close - ndays_low) / (ndays_high - ndays_low) * 100
        slow_d = fast_k.rolling(window=3).mean()
        df = df.assign(fast_k=fast_k, slow_d=slow_d).dropna()

        plt.figure(figsize=(9, 9))
        p1 = plt.subplot(3, 1, 1)
        plt.title('Triple Screen Trading')
        plt.grid(True)
        candlestick_ohlc(p1, ohlc.values, width=.6, colorup='red', colordown='blue')
        p1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        p2 = plt.subplot(3, 1, 2)
        plt.grid(True)
        p2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.plot(df.number, df['ema130'], color='k', label='EMA (130)')

        buy_list = []
        sell_list = []
        returns = 0
        own_price = 0
        for i in range(1, len(df.Close)):
            if df.ema130.values[i - 1] < df.ema130.values[i] and \
                    df.slow_d.values[i - 1] >= 20 and df.slow_d.values[i] < 20:
                plt.plot(df.number.values[i], df.ema130.values[i], 'r^')
                date = np.datetime_as_string(df.index.values[i], unit='D')
                cost = df.Close.values[i]
                if own_price == 0:
                    own_price += cost
                    buy_list.append([date, cost])

            elif df.ema130.values[i - 1] > df.ema130.values[i] and \
                    df.slow_d.values[i - 1] <= 80 and df.slow_d.values[i] > 80:
                plt.plot(df.number.values[i], df.ema130.values[i], 'bv')
                date = np.datetime_as_string(df.index.values[i], unit='D')
                cost = df.Close.values[i]
                if own_price != 0:
                    returns += ((cost / own_price) - 1) * 100
                    own_price = 0
                    sell_list.append([date, cost])
            if i == (len(df.Close) - 1):  # 매수했었는데 마지막 날까지 매도 신호가 안나왔다면 현재의 수익률을 환산
                date = np.datetime_as_string(df.index.values[i], unit='D')
                cost = df.Close.values[i]
                if own_price != 0:
                    returns += ((cost / own_price) - 1) * 100
                    own_price = 0
        plt.legend(loc='upper left')

        p3 = plt.subplot(3, 1, 3)
        plt.grid(True)
        p3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.plot(df.number, df['slow_d'], color='g', linestyle='solid', label='%D')
        plt.yticks([0, 20, 80, 100])
        plt.legend(loc='upper left')

        fig = plt.gcf()
        buf = io.BytesIO()

        fig.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = 'data:image/png;base64,' + urllib.parse.quote(string)

        return render(request, 'test2/result2_2.html',{'uri':uri,'start':start_date,'today':today_str,'code':code,'name':name,'returns':int(returns),'buy_list':buy_list,'sell_list':sell_list})
    elif idx == 3:
        queryDict = dict(request.GET)
        code = queryDict['code'][0]
        start_date = queryDict['date'][0]
        if start_date[4] != '-':
            start_date = str(start_date[:4]) + '-' + str(start_date[4:6]) + '-' + str(start_date[6:])

        df = fdr.DataReader(code, start_date)
        name_df = fdr.StockListing('KRX')
        name = name_df.loc[name_df['Symbol'] == code, 'Name'].values[0]

        ema60 = df.Close.ewm(span=60).mean()
        ema130 = df.Close.ewm(span=130).mean()
        macd = ema60 - ema130
        signal = macd.ewm(span=45).mean()
        macdhist = macd - signal
        df = df.assign(ema130=ema130, ema60=ema60, macd=macd, signal=signal, macdhist=macdhist).dropna()

        df['number'] = df.index.map(mdates.date2num)
        ohlc = df[['number', 'Open', 'High', 'Low', 'Close']]

        ndays_high = df.High.rolling(window=14, min_periods=1).max()
        ndays_low = df.Low.rolling(window=14, min_periods=1).min()

        fast_k = (df.Close - ndays_low) / (ndays_high - ndays_low) * 100
        slow_d = fast_k.rolling(window=3).mean()
        df = df.assign(fast_k=fast_k, slow_d=slow_d).dropna()

        plt.figure(figsize=(9, 9))
        p1 = plt.subplot(3, 1, 1)
        plt.title('Triple Screen Trading')
        plt.grid(True)
        candlestick_ohlc(p1, ohlc.values, width=.6, colorup='red', colordown='blue')
        p1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        p2 = plt.subplot(3, 1, 2)
        plt.grid(True)
        p2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.bar(df.number, df['macdhist'], color='c', label='MACD-Hist')
        plt.plot(df.number, df['macd'], color='k', label='MACD')
        plt.plot(df.number, df['signal'], 'y:', label='MACD-Signal')

        buy_list = []
        sell_list = []
        returns = 0
        own_price = 0
        for i in range(1, len(df.Close)):
            if df['macdhist'].values[i - 1] < df['macdhist'].values[i] and \
                    df.slow_d.values[i - 1] >= 20 and df.slow_d.values[i] < 20:
                plt.plot(df.number.values[i], df['macd'].values[i], 'r^')
                date = np.datetime_as_string(df.index.values[i], unit='D')
                cost = df.Close.values[i]
                if own_price == 0:
                    own_price += cost
                    buy_list.append([date, cost])

            elif df['macdhist'].values[i - 1] > df['macdhist'].values[i] and \
                    df.slow_d.values[i - 1] <= 80 and df.slow_d.values[i] > 80:
                plt.plot(df.number.values[i], df['macd'].values[i], 'bv')
                date = np.datetime_as_string(df.index.values[i], unit='D')
                cost = df.Close.values[i]
                if own_price != 0:
                    returns += ((cost / own_price) - 1) * 100
                    own_price = 0
                    sell_list.append([date, cost])
            if i == (len(df.Close) - 1):  # 매수했었는데 마지막 날까지 매도 신호가 안나왔다면 현재의 수익률을 환산
                date = np.datetime_as_string(df.index.values[i], unit='D')
                cost = df.Close.values[i]
                if own_price != 0:
                    returns += ((cost / own_price) - 1) * 100
                    own_price = 0
        plt.legend(loc='upper left')

        p3 = plt.subplot(3, 1, 3)
        plt.grid(True)
        p3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.plot(df.number, df['slow_d'], color='g', linestyle='solid', label='%D')
        plt.yticks([0, 20, 80, 100])
        plt.legend(loc='upper left')

        fig = plt.gcf()
        buf = io.BytesIO()

        fig.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = 'data:image/png;base64,' + urllib.parse.quote(string)

        return render(request, 'test2/result3_2.html',{'uri':uri,'start':start_date,'today':today_str,'code':code,'name':name,'returns':int(returns),'buy_list':buy_list,'sell_list':sell_list})
    elif idx == 4:  # 골든&데드 크로스 (5&20)
        today = datetime.today()
        today_str = today.strftime("%Y-%m-%d")
        queryDict = dict(request.GET)
        code = queryDict['code'][0]
        start_date = queryDict['date'][0]

        if start_date[4] != '-':
            start_date = str(start_date[:4]) + '-' + str(start_date[4:6]) + '-' + str(start_date[6:])

        price_df = fdr.DataReader(code, start_date)
        price_df = price_df.dropna()

        name_df = fdr.StockListing('KRX')
        name = name_df.loc[name_df['Symbol'] == code, 'Name'].values[0]

        price_df['ema5'] = price_df.Close.ewm(span=5).mean()
        price_df['ema20'] = price_df.Close.ewm(span=20).mean()
        price_df['st_rtn'] = (1 + price_df['Change']).cumprod()
        price_df = price_df.dropna()
        price_df['number'] = price_df.index.map(mdates.date2num)

        ohlc = price_df[['number', 'Open', 'High', 'Low', 'Close']]
        plt.figure(figsize=(9, 9))
        p1 = plt.subplot(2, 1, 1)
        plt.title('Golden Cross & Dead Cross Trade')
        plt.grid(True)
        candlestick_ohlc(p1, ohlc.values, width=.6, colorup='red', colordown='blue')
        p1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        p3 = plt.subplot(2, 1, 2)
        plt.grid(True)
        p3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.plot(price_df.number, price_df['ema5'], color='g', linestyle='solid', label='EMA 5')
        plt.plot(price_df.number, price_df['ema20'], color='k', linestyle='solid', label='EMA 20')

        buy_list = []
        sell_list = []
        returns = 0
        own_price = 0
        for i in range(1, len(price_df.Close)):
            if price_df['ema5'].values[i] > price_df['ema20'].values[i] and price_df['ema5'].values[i - 1] < \
                    price_df['ema20'].values[i - 1]:
                plt.plot(price_df.number.values[i], price_df['ema5'].values[i], 'r^')
                date = np.datetime_as_string(price_df.index.values[i], unit='D')
                cost = price_df.Close.values[i]
                if own_price == 0:
                    own_price += cost
                    buy_list.append([date, cost])
            elif price_df['ema5'].values[i] < price_df['ema20'].values[i] and price_df['ema5'].values[i - 1] > \
                    price_df['ema20'].values[i - 1]:
                plt.plot(price_df.number.values[i], price_df['ema5'].values[i], 'bv')
                date = np.datetime_as_string(price_df.index.values[i], unit='D')
                cost = price_df.Close.values[i]
                if own_price != 0:
                    returns += ((cost / own_price) - 1) * 100
                    own_price = 0
                    sell_list.append([date, cost])
            if i == (len(price_df.Close) - 1):  # 매수했었는데 마지막 날까지 매도 신호가 안나왔다면 현재의 수익률을 환산
                cost = price_df.Close.values[i]
                if own_price != 0:
                    returns += ((cost / own_price) - 1) * 100
                    own_price = 0
        plt.legend(loc='upper left')

        fig = plt.gcf()
        buf = io.BytesIO()

        fig.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = 'data:image/png;base64,' + urllib.parse.quote(string)

        return render(request, 'test2/result4_2.html',{'uri':uri,'start': start_date, 'today': today_str, 'code': code, 'name': name, 'returns': int(returns),'buy_list': buy_list, 'sell_list': sell_list})
    elif idx == 5:  # 골든&데드 크로스 (20&60)
        today = datetime.today()
        today_str = today.strftime("%Y-%m-%d")
        queryDict = dict(request.GET)
        code = queryDict['code'][0]
        start_date = queryDict['date'][0]

        if start_date[4] != '-':
            start_date = str(start_date[:4]) + '-' + str(start_date[4:6]) + '-' + str(start_date[6:])

        price_df = fdr.DataReader(code, start_date)
        price_df = price_df.dropna()

        name_df = fdr.StockListing('KRX')
        name = name_df.loc[name_df['Symbol'] == code, 'Name'].values[0]

        price_df['ema20'] = price_df.Close.ewm(span=20).mean()
        price_df['ema60'] = price_df.Close.ewm(span=60).mean()
        price_df['st_rtn'] = (1 + price_df['Change']).cumprod()
        price_df = price_df.dropna()
        price_df['number'] = price_df.index.map(mdates.date2num)

        ohlc = price_df[['number', 'Open', 'High', 'Low', 'Close']]
        plt.figure(figsize=(9, 9))
        p1 = plt.subplot(2, 1, 1)
        plt.title('Golden Cross & Dead Cross Trade')
        plt.grid(True)
        candlestick_ohlc(p1, ohlc.values, width=.6, colorup='red', colordown='blue')
        p1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        p3 = plt.subplot(2, 1, 2)
        plt.grid(True)
        p3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.plot(price_df.number, price_df['ema20'], color='g', linestyle='solid', label='EMA 20')
        plt.plot(price_df.number, price_df['ema60'], color='k', linestyle='solid', label='EMA 60')

        buy_list = []
        sell_list = []
        returns = 0
        own_price = 0
        for i in range(1, len(price_df.Close)):
            if price_df['ema20'].values[i] > price_df['ema60'].values[i] and price_df['ema20'].values[i - 1] < \
                    price_df['ema60'].values[i - 1]:
                plt.plot(price_df.number.values[i], price_df['ema20'].values[i], 'r^')
                date = np.datetime_as_string(price_df.index.values[i], unit='D')
                cost = price_df.Close.values[i]
                if own_price == 0:
                    own_price += cost
                    buy_list.append([date, cost])
            elif price_df['ema20'].values[i] < price_df['ema60'].values[i] and price_df['ema20'].values[i - 1] > \
                    price_df['ema60'].values[i - 1]:
                plt.plot(price_df.number.values[i], price_df['ema20'].values[i], 'bv')
                date = np.datetime_as_string(price_df.index.values[i], unit='D')
                cost = price_df.Close.values[i]
                if own_price != 0:
                    returns += ((cost / own_price) - 1) * 100
                    own_price = 0
                    sell_list.append([date, cost])
            if i == (len(price_df.Close) - 1):  # 매수했었는데 마지막 날까지 매도 신호가 안나왔다면 현재의 수익률을 환산
                cost = price_df.Close.values[i]
                if own_price != 0:
                    returns += ((cost / own_price) - 1) * 100
                    own_price = 0
        plt.legend(loc='upper left')

        fig = plt.gcf()
        buf = io.BytesIO()

        fig.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = 'data:image/png;base64,' + urllib.parse.quote(string)

        return render(request, 'test2/result5_2.html',{'uri':uri,'start': start_date, 'today': today_str, 'code': code, 'name': name, 'returns': int(returns),'buy_list': buy_list, 'sell_list': sell_list})
