from django.shortcuts import render
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import numpy as np
import mplfinance as mpf
import FinanceDataReader as fdr
import io
import urllib, base64


def index(request):
    return render(request,'test3/index.html')

def result(request):
    return render(request, 'test3/result.html')

def detail(request):
    today = datetime.today()
    today_str = today.strftime("%Y-%m-%d")
    # photo_str = today.strftime("%Y-%m-%d-%H%M%S")

    queryDict = dict(request.GET)
    code = queryDict['code'][0]
    start_date = queryDict['date'][0]
    if start_date[4] != '-':
        start_date = str(start_date[:4]) + '-' + str(start_date[4:6]) + '-' + str(start_date[6:])

    price_df = fdr.DataReader(code, start_date)
    price_df = price_df.dropna()
    name_df = fdr.StockListing('KRX')
    name = name_df.loc[name_df['Symbol'] == code, 'Name'].values[0]

    price_df['st_rtn'] = (1 + price_df['Change']).cumprod()

    historical_max = price_df['Close'].cummax()
    daily_drawdown = price_df['Close'] / historical_max - 1.0
    historical_dd = daily_drawdown.cummin()  ## 최대 낙폭

    MDD = historical_dd.min()
    simple_return = ((price_df['st_rtn'].values[-1]-1) * 100)
    CAGR = simple_return ** (252. / len(price_df.index)) - 1  # 연평균 복리 수익률
    VOL = np.std(price_df['Change']) * np.sqrt(252.)
    Sharpe = np.mean(price_df['Change']) / np.std(price_df['Change']) * np.sqrt(252.)

    # print('returns :', simple_return, '%')
    # print('CAGR :', round(CAGR, 2), '%')
    # print('Sharpe :', round(Sharpe, 2))
    # print('VOL :', round(VOL * 100, 2), '%')
    # print('MDD :', round(-1 * MDD * 100, 2), '%')

    df = price_df[['Open', 'High', 'Low', 'Close', 'Volume']]
    kwargs = dict(type='candle', volume=True)
    mc = mpf.make_marketcolors(up='r', down='b', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc)

    mpf.plot(df, **kwargs, style=s)
    fig = plt.gcf()
    buf = io.BytesIO()

    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)

    return render(request, 'test3/detail.html',{'uri':uri,'start':start_date,'today':today_str,'code':code,'name':name,'returns':int(simple_return),'CAGR':round(CAGR, 2),'SHARPE':round(Sharpe, 2),'VOL':round(VOL * 100, 2),'MDD':round(-1 * MDD * 100, 2)})
