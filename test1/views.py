from django.shortcuts import render
from urllib.request import urlopen, Request
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import mplfinance as mpf
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc
import io
import urllib, base64
# from django.http import HttpResponse



def index(request):
    return render(request,'test1/index.html')

def index2(request):
    return render(request,'test1/index2.html')

def payment(request,idx):
    if idx == 0:  # 마법공식의 결제 안내문구
        return render(request,'test1/payment0.html')
    if idx == 1:  # 듀얼 모멘텀의 결제 안내문구
        return render(request,'test1/payment1.html')


def payment_detail(request,idx):
    if idx == 0:
        return render(request,'test1/payment0_detail.html')
    if idx == 1:
        return render(request,'test1/payment1_detail.html')


def question1(request):
    return render(request,'test1/question1.html')

def question2(request):
    return render(request, 'test1/question2.html')

def question3(request):
    return render(request, 'test1/question3.html')

def question4_0(request):
    return render(request, 'test1/question4_0.html')

def question4_1(request):
    return render(request, 'test1/question4_1.html')

def result(request,num):
    if num == 0:
        return render(request, 'test1/result0.html')
    elif num == 1:
        return render(request, 'test1/result1.html')
    elif num == 2:
        return render(request, 'test1/result2.html')
    elif num == 3:
        return render(request, 'test1/result3.html')

def detail(request,idx):  # 여기선 결과물이  idx 0 : 종목 분석 결과물 , idx 1: 레버리지 결과물 , idx 2: 마법공식, 3:듀얼모멘텀 일때밖에 없다.
    if idx == 0:
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

        fig.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = 'data:image/png;base64,' + urllib.parse.quote(string)

        return render(request, 'test1/result0_2.html',
                      {'uri':uri,'start': start_date, 'today': today_str, 'code': code, 'name': name,
                       'returns': int(simple_return), 'CAGR': round(CAGR, 2), 'SHARPE': round(Sharpe, 2),
                       'VOL': round(VOL * 100, 2), 'MDD': round(-1 * MDD * 100, 2)})
    elif idx == 1: #  골든&데드 크로스 전략으로 분석한 결과
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

        return render(request, 'test1/result1_2.html', {'uri':uri,'start':start_date,'today':today_str,'code':code,'name':name,'returns':int(returns),'buy_list':buy_list,'sell_list':sell_list})
    elif idx == 2:  #마법공식
        def get_html_fnguide(ticker, gb):
            """
            :param ticker: 종목코드
            :param gb: 데이터 종류 (0 : 재무제표, 1 : 재무비율, 2: 투자지표)
            :return:
            """
            url = []

            url.append(
                "https://comp.fnguide.com/SVO2/ASP/SVD_Finance.asp?pGB=1&gicode=A" + ticker + "&cID=&MenuYn=Y&ReportGB=&NewMenuID=103&stkGb=701")
            url.append(
                "https://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A" + ticker + "&cID=&MenuYn=Y&ReportGB=&NewMenuID=104&stkGb=701")
            url.append(
                "https://comp.fnguide.com/SVO2/ASP/SVD_Invest.asp?pGB=1&gicode=A" + ticker + "&cID=&MenuYn=Y&ReportGB=&NewMenuID=105&stkGb=701")

            if gb > 2:
                return None

            url = url[gb]
            try:

                req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                html_text = urlopen(req).read()

            except AttributeError as e:
                return None

            return html_text
        def get_code_datas(code):
            for_profit = pd.read_html(get_html_fnguide(code, gb=0))[0]
            for_roa = pd.read_html(get_html_fnguide(code, gb=1))[0]
            for_per = pd.read_html(get_html_fnguide(code, gb=2))[0]

            profit_df = for_profit.loc[0].dropna()
            roa_df = for_roa.loc[19].dropna()
            per_df = for_per.loc[2].dropna()

            per_low = float(per_df.iloc[-1])
            per_high = float(per_df.iloc[-2])

            profit = float(profit_df.iloc[-1])  # 저장 해야할 매출액
            roa = float(roa_df.iloc[-1])  # 저장 해야할 ROA
            per = (per_low + per_high) / 2  # 저장 해야할 PER

            return profit, roa, per
        def sort_values(s_value, asc=True, standard=0):
            s_value_mask = s_value.mask(s_value < standard, np.nan)
            s_value_mask_rank = s_value_mask.rank(ascending=asc, na_option="bottom")

            return s_value_mask_rank
        ### 조회할 대상을 바꾸려면 total_list 의 대상을 KOSPI,KOSDAQ,KRX 중 선택해 바꾼다
        ###########################날짜###############################
        today = datetime.today()
        today_str = today.strftime("%Y-%m-%d")

        before365 = today - timedelta(days=365)
        before365_str = before365.strftime("%Y-%m-%d")
        ##########################코스닥 정보#################################
        total_list = fdr.StockListing('KOSDAQ')
        stock_code_list = total_list['Symbol'].values
        stock_name_list = total_list['Name'].values
        ##########################재무제표 정보################################
        rows = []
        stock_code_name_list = list(zip(stock_code_list, stock_name_list))
        columns = ['code', 'company', 'current_price', 'profit', 'returns', 'roa', 'per']
        for code, company in stock_code_name_list:
            try:
                profit, roa, per = get_code_datas(code)
                df = fdr.DataReader(code, before365_str, today_str)
                df = df.dropna()
                # old_price = df['Close'].values[0]
                current_price = df['Close'].values[-1]
                returns = (df['Change'] + 1).cumprod()
                returns = int((returns.values[-1]-1) * 100)
                rows.append([code, company, current_price, profit, returns, roa, per])
            except:
                continue

        df = pd.DataFrame(rows, columns=columns)
        per_rank = sort_values(df['per'], asc=True, standard=0)
        roa_rank = sort_values(df['roa'], asc=True, standard=0)

        result_rank = per_rank + roa_rank
        result_rank = sort_values(result_rank, asc=True)
        result_rank = result_rank.where(result_rank <= 10, 0)
        result_rank = result_rank.mask(result_rank > 0, 1)
        mf_df = df.loc[result_rank > 0, ['code', 'company', 'current_price', 'profit', 'returns']].copy()  # mf_df를 최종적으로 반환

        code_list = mf_df['code'].values
        name_list = mf_df['company'].values

        stocks = mf_df['code'].values.tolist()
        ex_df = pd.DataFrame()
        for s in stocks:
            ex_df[s] = fdr.DataReader(s, before365_str, today_str)['Close']

        daily_ret = ex_df.pct_change()
        annual_ret = daily_ret.mean() * 252
        daily_cov = daily_ret.cov()
        annual_cov = daily_cov * 252

        port_ret = []
        port_risk = []
        port_weights = []
        sharpe_ratio = []

        for _ in range(30000):
            weights = np.random.random(len(stocks))
            weights /= np.sum(weights)

            returns = np.dot(weights, annual_ret)
            risk = np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights)))

            port_ret.append(returns)
            port_risk.append(risk)
            port_weights.append(weights)
            sharpe_ratio.append(returns / risk)

        portfolio = {'Returns': port_ret, 'Risk': port_risk, 'Sharpe': sharpe_ratio}
        for i, s in enumerate(stocks):
            portfolio[s] = [weight[i] for weight in port_weights]

        montekarlo_df = pd.DataFrame(portfolio)
        montekarlo_df = montekarlo_df[['Returns', 'Risk', 'Sharpe'] + [s for s in stocks]]

        max_sharpe = montekarlo_df.loc[montekarlo_df['Sharpe'] == montekarlo_df['Sharpe'].max()]
        min_risk = montekarlo_df.loc[montekarlo_df['Risk'] == montekarlo_df['Risk'].min()]

        montekarlo_df.plot.scatter(x='Risk', y='Returns', c='Sharpe', cmap='viridis', edgecolors='k', figsize=(11, 7),
                                   grid=True)
        plt.scatter(x=max_sharpe['Risk'], y=max_sharpe['Returns'], c='r', marker='*', s=300)
        plt.scatter(x=min_risk['Risk'], y=min_risk['Returns'], c='r', marker='o', s=200)
        plt.xlabel('Risk')
        plt.ylabel('Expected Returns')

        fig = plt.gcf()
        buf = io.BytesIO()

        fig.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = 'data:image/png;base64,' + urllib.parse.quote(string)

        table = mf_df.values.tolist()
        names = name_list.tolist()
        codes = code_list.tolist()
        max_sharpe = max_sharpe.values.tolist()[0][3:]
        min_risk = min_risk.values.tolist()[0][3:]

        max_sharpe = [int(ratio * 100) for ratio in max_sharpe]
        min_risk = [int(ratio * 100) for ratio in min_risk]

        max_sharpe = list(zip(codes, names, max_sharpe))
        min_risk = list(zip(codes, names, min_risk))

        # ######################## 빠른 실행을 위한 임의식########################################
        # today = datetime.today()
        # today_str = today.strftime("%Y-%m-%d")
        # table= [['323990', '박셀바이오', 12675, 55850, 412.1771217712176], ['195990', '에이비프로바이오', 799, 2790, 341.91176470588243], ['019770', '서연탑메탈', 3540, 10200, 293.52517985611524], ['090710', '휴림로봇', 757, 2100, 277.7777777777777], ['299170', '더블유에스아이', 2010, 4990, 249.49999999999997], ['131220', '대한과학', 8580, 20800, 241.01969872537663], ['214310', '세미콘라이트', 1190, 2250, 237.84355179704005], ['114450', 'KPX생명과학', 7120, 16550, 237.44619799139167],['041190', '우리기술투자', 2390, 5600, 237.28813559322037], ['089230', 'THE E&M', 623, 1445, 233.8187702265373]]
        # max_sharpe = [0.814618470015201, 0.7343546622494926, 1.1092984247146227, 0.2936788778649645, 0.015598816469787221, 0.0010880476048793204, 0.06833393347141829, 0.003946573951989132, 0.20048834603134788, 0.023275781122569002, 0.06327322281533262, 0.10572480129580768, 0.22459159937190432][3:]
        # min_risk = [-2.614002011068698, 0.34437464630300624, -7.5905762492419555, 0.01906015790583676, 0.09287463622091505, 0.2195628069283207, 0.07213189892415349, 0.23067544605133547, 0.01020231260585002, 0.05947836483574131, 0.15711615156078387, 0.11386916902467774, 0.02502905594238551][3:]
        # codes = ['0001','0002','0003','0004','0005','0006','0007','0008','0009','0010']
        # names = ['마법 1번째 종목','마법 2번째 종목','3번째 종목','4번째 종목','5번째 종목','6번째 종목','7번째 종목','8번째 종목','9번째 종목','10번째 종목']
        #
        # max_sharpe = [int(ratio*100) for ratio in max_sharpe]
        # min_risk = [int(ratio * 100) for ratio in min_risk]
        #
        # max_sharpe = list(zip(codes,names,max_sharpe))
        # min_risk = list(zip(codes,names, min_risk))

        return render(request, 'test1/result2_2.html',{'uri':uri,'table':table,'max_sharpe':max_sharpe,'min_risk':min_risk,'names':names,'today':today_str})
    elif idx == 3:  #듀얼모멘텀
        ### 조회할 대상을 바꾸려면 total_list 의 대상을 KOSPI,KOSDAQ,KRX 중 선택해 바꾼다
        ###########################날짜###############################
        today = datetime.today()
        today_str = today.strftime("%Y-%m-%d")

        before120 = today - timedelta(days=180)
        before120_str = before120.strftime("%Y-%m-%d")

        before60 = today - timedelta(days=90)
        before60_str = before60.strftime("%Y-%m-%d")
        stock_first_count = 30
        stock_second_count = 10
        ################################################################
        total_list = fdr.StockListing('KOSDAQ')
        stock_code_list = total_list['Symbol'].values
        stock_name_list = total_list['Name'].values

        rows = []
        columns = ['code', 'company', 'old_price', 'new_price', 'returns']
        for code, company in zip(stock_code_list, stock_name_list):
            try:
                df = fdr.DataReader(code, before120_str, before60_str)
                df = df.dropna()
                old_price = df['Close'].values[0]
                new_price = df['Close'].values[-1]
                # df['daily_rtn']=df['Close'].pct_change()
                returns = (df['Change'] + 1).cumprod()
                returns = (returns.values[-1]-1) * 100
                rows.append([code, company, old_price, new_price, returns])
            except:
                continue

        df = pd.DataFrame(rows, columns=columns)
        df = df.sort_values(by='returns', ascending=False)
        df = df.head(stock_first_count)
        df.index = pd.Index(range(stock_first_count))

        code_list = df['code'].values
        name_list = df['company'].values
        second_rows = []
        for code, company in zip(code_list, name_list):
            df = fdr.DataReader(code, before60_str, today_str)
            df = df.dropna()
            try:
                old_price = df['Close'].values[0]
                new_price = df['Close'].values[-1]
                old_price = format(old_price,',d')
                new_price = format(new_price,',d')
                returns = (df['Change'] + 1).cumprod()
                returns = int((returns.values[-1]-1) * 100)
                second_rows.append([code, company, old_price, new_price, returns])
            except:
                continue
        final_df = pd.DataFrame(second_rows, columns=columns)
        final_df = final_df.sort_values(by='returns', ascending=False)
        final_df = final_df.head(stock_second_count)
        final_df.index = pd.Index(range(stock_second_count))

        fin_name_list = final_df['company'].values.tolist()
        fin_code_list = final_df['code'].values.tolist()

        stocks = final_df['code'].values.tolist()
        ex_df = pd.DataFrame()
        for s in stocks:
            ex_df[s] = fdr.DataReader(s, before60_str, today_str)['Close']

        daily_ret = ex_df.pct_change()
        annual_ret = daily_ret.mean() * 252
        daily_cov = daily_ret.cov()
        annual_cov = daily_cov * 252

        port_ret = []
        port_risk = []
        port_weights = []
        sharpe_ratio = []

        for _ in range(30000):
            weights = np.random.random(len(stocks))
            weights /= np.sum(weights)

            returns = np.dot(weights, annual_ret)
            risk = np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights)))

            port_ret.append(returns)
            port_risk.append(risk)
            port_weights.append(weights)
            sharpe_ratio.append(returns / risk)

        portfolio = {'Returns': port_ret, 'Risk': port_risk, 'Sharpe': sharpe_ratio}
        for i, s in enumerate(stocks):
            portfolio[s] = [weight[i] for weight in port_weights]

        montekarlo_df = pd.DataFrame(portfolio)
        montekarlo_df = montekarlo_df[['Returns', 'Risk', 'Sharpe'] + [s for s in stocks]]

        max_sharpe = montekarlo_df.loc[montekarlo_df['Sharpe'] == montekarlo_df['Sharpe'].max()]
        min_risk = montekarlo_df.loc[montekarlo_df['Risk'] == montekarlo_df['Risk'].min()]

        montekarlo_df.plot.scatter(x='Risk', y='Returns', c='Sharpe', cmap='viridis', edgecolors='k', figsize=(11, 7),
                                   grid=True)
        plt.scatter(x=max_sharpe['Risk'], y=max_sharpe['Returns'], c='r', marker='*', s=300)
        plt.scatter(x=min_risk['Risk'], y=min_risk['Returns'], c='r', marker='o', s=200)
        plt.xlabel('Risk')
        plt.ylabel('Expected Returns')

        fig = plt.gcf()
        buf = io.BytesIO()

        fig.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = 'data:image/png;base64,' + urllib.parse.quote(string)


        # 전달 해야할 것들 : final_df, max_sharpe , min_risk , today_str

        table=final_df.values.tolist()
        names = name_list.tolist()
        codes = code_list.tolist()
        max_sharpe = max_sharpe.values.tolist()[0][3:]
        min_risk = min_risk.values.tolist()[0][3:]

        max_sharpe = [int(ratio*100) for ratio in max_sharpe]
        min_risk = [int(ratio * 100) for ratio in min_risk]

        max_sharpe = list(zip(fin_code_list,fin_name_list,max_sharpe))
        min_risk = list(zip(fin_code_list,fin_name_list, min_risk))

        # ######################## 빠른 실행을 위한 임의식########################################
        # today = datetime.today()
        # today_str = today.strftime("%Y-%m-%d")
        # table= [['323990', '박셀바이오', 12675, 55850, 412.1771217712176], ['195990', '에이비프로바이오', 799, 2790, 341.91176470588243], ['019770', '서연탑메탈', 3540, 10200, 293.52517985611524], ['090710', '휴림로봇', 757, 2100, 277.7777777777777], ['299170', '더블유에스아이', 2010, 4990, 249.49999999999997], ['131220', '대한과학', 8580, 20800, 241.01969872537663], ['214310', '세미콘라이트', 1190, 2250, 237.84355179704005], ['114450', 'KPX생명과학', 7120, 16550, 237.44619799139167],['041190', '우리기술투자', 2390, 5600, 237.28813559322037], ['089230', 'THE E&M', 623, 1445, 233.8187702265373]]
        # max_sharpe = [0.814618470015201, 0.7343546622494926, 1.1092984247146227, 0.2936788778649645, 0.015598816469787221, 0.0010880476048793204, 0.06833393347141829, 0.003946573951989132, 0.20048834603134788, 0.023275781122569002, 0.06327322281533262, 0.10572480129580768, 0.22459159937190432][3:]
        # min_risk = [-2.614002011068698, 0.34437464630300624, -7.5905762492419555, 0.01906015790583676, 0.09287463622091505, 0.2195628069283207, 0.07213189892415349, 0.23067544605133547, 0.01020231260585002, 0.05947836483574131, 0.15711615156078387, 0.11386916902467774, 0.02502905594238551][3:]
        # codes = ['0001','0002','0003','0004','0005','0006','0007','0008','0009','0010']
        # names = ['1번째 종목','2번째 종목','3번째 종목','4번째 종목','5번째 종목','6번째 종목','7번째 종목','8번째 종목','9번째 종목','10번째 종목']
        #
        # max_sharpe = [int(ratio*100) for ratio in max_sharpe]
        # min_risk = [int(ratio * 100) for ratio in min_risk]
        #
        # max_sharpe = list(zip(codes,names,max_sharpe))
        # min_risk = list(zip(codes,names, min_risk))

        return render(request, 'test1/result3_2.html',{'uri':uri,'table':table,'max_sharpe':max_sharpe,'min_risk':min_risk,'names':names,'today':today_str})