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
# import stocker
# from keras.models import Sequential
# from keras.layers import LSTM,Dropout,Dense,Activation
from scipy import stats
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from bs4 import BeautifulSoup
import requests
import re
from wordcloud import WordCloud




def index(request):
    return render(request,'test4/index.html')

def payment(request,idx):
    if idx == 0:  # 클러스터링 결제 안내문구
        return render(request,'test4/payment0.html')

def payment_detail(request,idx):
    if idx == 0:   # 클러스터링 보고서 예시
        return render(request,'test4/payment0_detail.html')

def question(request):
    return render(request, 'test4/question.html')

def predict(request):
    return render(request, 'test4/predict.html')

def classify(request):
    return render(request, 'test4/classify.html')

def result_predict(request,num):  # 예측 카테고리로 이동
    if num == 0:
        return render(request, 'test4/result_predict0.html')
    if num == 1:
        return render(request, 'test4/result_predict1.html')
    if num == 2:
        return render(request, 'test4/result_predict2.html')

def detail_predict(request,idx):  # 예측 카테고리의 최종 결과값
    if idx == 0:
        return render(request, 'test4/result_predict0_2.html')
    elif idx == 1:
        return render(request, 'test4/result_predict1_2.html')
    elif idx ==2:
        return render(request, 'test4/result_predict1_2.html')
        # today = datetime.today()
        # today_str = today.strftime("%Y-%m-%d")
        # ##########################날짜###########################
        # ##########################################################
        # queryDict = dict(request.GET)
        # code = queryDict['code'][0]
        # df = fdr.DataReader(code)
        # dataset = df.dropna()
        # name_df = fdr.StockListing('KRX')
        # tag = ''
        # if code in name_df['Symbol'].values:
        #     name = name_df.loc[name_df['Symbol'] == code, 'Name'].values[0]
        #     tag = '.KS'
        # tomorrow_data = stocker.predict.tomorrow(code + tag)
        # tomorrow_price = tomorrow_data[0]
        # tomorrow_error = tomorrow_data[1]
        # tomorrow_date = tomorrow_data[2]
        #
        # ##########################################################
        # ##########################################################
        # close_prices = dataset['Close'].values
        # seq_len = 50
        # sequence_length = seq_len + 1
        #
        # result = []
        # for index in range(len(close_prices) - sequence_length):
        #     result.append(close_prices[index: index + sequence_length])
        #
        # normalized_data = []
        # window_mean = []
        # window_std = []
        #
        # for window in result:
        #     normalized_window = [((p - np.mean(window)) / np.std(window)) for p in window]
        #     normalized_data.append(normalized_window)
        #     window_mean.append(np.mean(window))
        #     window_std.append(np.std(window))
        #
        # result = np.array(normalized_data)
        #
        # # split train and test data
        # row = int(round(result.shape[0] * 0.9))
        # train = result[:row, :]
        # # shuffle을 해도, 51개의 묶음은 변하지 않으므로 상관없음
        # # 50일(x)로 1일(y) 예측
        # np.random.shuffle(train)
        #
        # x_train = train[:, :-1]
        # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        # y_train = train[:, -1]
        #
        # x_test = result[row:, :-1]
        # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        # y_test = result[row:, -1]
        #
        # model = Sequential()
        #
        # model.add(LSTM(20, return_sequences=True, input_shape=(50, 1)))
        # model.add(Dropout(0.5))
        # model.add(LSTM(20, return_sequences=False))
        # model.add(Dense(1, activation='linear'))
        # model.compile(loss='mse', optimizer='adam')
        # hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=10)
        #
        # lt = close_prices[-365:]
        # seq_len = 50
        # sequence_length = seq_len + 1
        #
        # result = []
        # for index in range(len(lt) - sequence_length):
        #     result.append(lt[index: index + sequence_length])
        # normalized_data = []
        # window_mean = []
        # window_std = []
        #
        # for window in result:
        #     normalized_window = [((p - np.mean(window)) / np.std(window)) for p in window]
        #     normalized_data.append(normalized_window)
        #     window_mean.append(np.mean(window))
        #     window_std.append(np.std(window))
        #
        # result = np.array(normalized_data)
        #
        # x_test = result[:, :-1]
        # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        # y_test = result[:, -1]
        # pred = model.predict(x_test)
        # pred_result = []
        # pred_y = []
        # for i in range(len(pred)):
        #     n1 = (pred[i] * window_std[i]) + window_mean[i]
        #     n2 = (y_test[i] * window_std[i]) + window_mean[i]
        #     pred_result.append(n1)
        #     pred_y.append(n2)
        #
        # plt.figure(facecolor='white', figsize=(9, 9))
        # plt.subplot(2, 1, 1)
        # plt.grid(True)
        # plt.plot(pred_y, label='Stock Price')
        # plt.plot(pred_result, label='AI predict')
        # plt.legend(loc='upper left')
        #
        # plt.subplot(2, 1, 2)
        # plt.grid(True)
        # plt.plot(hist.history['loss'], color='k', linestyle='solid', label='Loss')
        # plt.plot(hist.history['val_loss'], color='g', linestyle='dashed', label='Val_loss')
        # plt.legend(loc='upper left')
        #
        # fig = plt.gcf()
        # buf = io.BytesIO()
        #
        # fig.savefig(buf, format='png')
        # buf.seek(0)
        # string = base64.b64encode(buf.read())
        # uri = 'data:image/png;base64,' + urllib.parse.quote(string)
        # return render(request, 'test4/result_predict2_2.html',{'uri': uri, 'today': today_str, 'tomorrow': tomorrow_date, 'today_price': int(close_prices[-1]),'tomorrow_price': int(tomorrow_price), 'tomorrow_error': tomorrow_error, 'name': name, 'code': code})

def result_classify(request,num):  # 분류 카테고리로 이동
    if num == 0:
        return render(request, 'test4/result_classify0.html')
    if num == 1:
        return render(request, 'test4/result_classify1.html')
    if num == 2:
        return render(request, 'test4/result_classify2.html')

def detail_classify(request,idx):  # 분류 카테고리의 최종 결과값
    if idx == 0:
        today = datetime.today()
        today_str = today.strftime("%Y-%m-%d")
        ##########################날짜###########################
        ##########################################################
        queryDict = dict(request.GET)
        code1 = queryDict['code1'][0]
        code2 = queryDict['code2'][0]

        name_df = fdr.StockListing('KRX')
        name1 = name_df.loc[name_df['Symbol'] == code1, 'Name'].values[0]
        name2 = name_df.loc[name_df['Symbol'] == code2, 'Name'].values[0]
        #########################코드 & 이름#######################
        ##########################################################

        df1 = fdr.DataReader(code1)
        df2 = fdr.DataReader(code2)

        date1 = df1.iloc[0].name
        date2 = df2.iloc[0].name

        if date1 < date2:
            date = date2
        elif date1 > date2:
            date = date1
        elif date1 == date2:
            date = date1
        start_date = date.strftime("%Y-%m-%d")

        first_data = fdr.DataReader(code1, start_date)['Close']
        second_data = fdr.DataReader(code2, start_date)['Close']
        d = (first_data / first_data.iloc[0]) * 100
        k = (second_data / second_data.iloc[0]) * 100

        df = pd.DataFrame({'First': first_data, 'Second': second_data})
        df = df.dropna()
        regr = stats.linregress(df['First'], df['Second'])
        regr_line = f'y = {regr.slope:.2f}x +{int(regr.intercept)}'
        corr = df['First'].corr(df['Second'])
        r_squared = corr ** 2

        plt.figure(figsize=(9, 9))
        plt.subplot(2, 1, 1)
        plt.plot(d.index, d, 'g--',label=code1)
        plt.plot(k.index, k, 'k',label=code2)
        plt.title('Exponential Chart')
        plt.grid(True)
        plt.legend(loc='upper left')

        plt.subplot(2, 1, 2)
        plt.plot(df['First'], df['Second'], '.')
        plt.plot(df['First'], regr.slope * df['First'] + regr.intercept, 'r')
        plt.legend(['Scatter plot', regr_line])

        fig = plt.gcf()
        buf = io.BytesIO()

        fig.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = 'data:image/png;base64,' + urllib.parse.quote(string)
        return render(request, 'test4/result_classify0_2.html',{'uri':uri,'start':start_date,'today':today_str,'code1':code1,'code2':code2,'name1':name1,'name2':name2,'corr':round(corr,2),'corr_squar':round(r_squared,2),'line':regr_line})
    elif idx == 1:
        ### 조회할 대상을 바꾸려면 total_list 의 대상을 KOSPI,KOSDAQ,KRX 중 선택해 바꾼다
        today = datetime.today()
        today_str = today.strftime("%Y-%m-%d")
        start_date = today - timedelta(days=90)
        start_date = start_date.strftime("%Y-%m-%d")
        # ##########################날짜###########################
        queryDict = dict(request.GET)
        special_code = queryDict['code'][0] # 입력한 종목의 코드
        # ##########################################################
        name_df = fdr.StockListing('KRX')
        special_name = name_df.loc[name_df['Symbol'] == special_code, 'Name'].values[0]  # 입력한 종목의 이름
        total_list = fdr.StockListing('KOSDAQ')['Symbol'].values.tolist()

        prices_list = []
        for code in total_list:
            try:
                prices = fdr.DataReader(code, start_date)['Close']
                prices = pd.DataFrame(prices)
                prices.columns = [code]
                prices_list.append(prices)
            except:
                pass
        prices_df = pd.concat(prices_list, axis=1)
        prices_df.sort_index(inplace=True)

        df = prices_df.pct_change().iloc[1:].T
        df = df.dropna()

        normalize = Normalizer()
        array_norm = normalize.fit_transform(df)
        df_norm = pd.DataFrame(array_norm, columns=df.columns)
        final_df = df_norm.set_index(df.index)
        special_idx = list(final_df.index).index(special_code)  # 입력한 종목의 클러스터링에 쓰인 전체 코드중 위치한 순서
        companies = final_df.index

        kmeans = KMeans(n_clusters=100, n_init=10, random_state=42)  ### 일별 수익률을 이용한 비지도 분류
        kmeans.fit(final_df)

        special_label = kmeans.labels_[special_idx]  # 입력한 종목의 결과값 레이블
        same_sector_company = []
        for i in range(len(kmeans.labels_)):
            if special_label == kmeans.labels_[i]:
                same_sector_company.append(i)
        same_sector_company = np.array(same_sector_company)
        sorted_codes = list(companies[same_sector_company])
        sorted_codes.remove(special_code)  # 클러스터링한 묶음 리스트에서 입력한 종목 코드는 제거한다.

        returns_list = []
        start_price_list = []
        end_price_list = []
        for code in sorted_codes:
            df = fdr.DataReader(code, start_date)
            df = df.dropna()
            star_price = df['Close'].values[0]
            end_price = df['Close'].values[-1]
            returns = (df['Change'] + 1).cumprod()
            returns = (returns.values[-1]-1) * 100
            start_price_list.append(star_price)
            end_price_list.append(end_price)
            returns_list.append(int(returns))

        result = pd.DataFrame({'start_price': start_price_list, 'end_price': end_price_list, 'returns': returns_list},
                              index=sorted_codes)
        result.sort_values(['returns'])
        result = result.head(10)

        result_start_price = result['start_price'].values.tolist()  # 최종 종목 10개 과거가
        result_end_price = result['end_price'].values.tolist()  # 최종 종목 10개 현재가
        result_returns = result['returns'].values.tolist()  # 최종 종목 10개 수익률
        result_codes = result.index.values.tolist()  # 최종 종목코드 10개
        result_names = []  # 최종 종목명 10개
        for code in result_codes:
            if code in name_df['Symbol'].values:
                name = name_df.loc[name_df['Symbol'] == code, 'Name'].values[0]
                result_names.append(name)

        table = []
        for i in range(len(result_start_price)):
            container = []
            container.append(result_codes[i])
            container.append(result_names[i])
            container.append(result_start_price[i])
            container.append(result_end_price[i])
            container.append(result_returns[i])
            table.append(container)

        price_df = fdr.DataReader(special_code, start_date)
        price_df = price_df.dropna()

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



        return render(request, 'test4/result_classify1_2.html',{'uri':uri,'today':today_str,'code':special_code,'name':special_name,'table':table})
    elif idx == 2:
        ### 조회할 대상을 바꾸려면 total_list 의 대상을 KOSPI,KOSDAQ,KRX 중 선택해 바꾼다
        today = datetime.today()
        today_str = today.strftime("%Y-%m-%d")
        # ##########################날짜###########################
        queryDict = dict(request.GET)
        special_code = queryDict['code'][0] # 입력한 종목의 코드
        # ##########################################################
        total_list = []
        def crawler(company_code, maxpage):
            page = 1
            while page <= int(maxpage):
                url = 'https://finance.naver.com/item/news_news.nhn?code=' + \
                      str(company_code) + '&page=' + str(page)
                source_code = requests.get(url).text
                html = BeautifulSoup(source_code, "lxml")

                # 뉴스 제목
                titles = html.select('.title')

                for title in titles:
                    title = title.get_text()
                    title = re.sub('\n', '', title)
                    total_list.append(title)

                page += 1
        def convert_to_code(company, maxpage):

            data = pd.read_csv('./joosajang/test4/static/test4/company_list.txt', dtype=str, sep='\t')  # 종목코드 추출
            company_name = data['회사명']
            keys = [i for i in company_name]  # 데이터프레임에서 리스트로 바꾸기

            company_code = data['종목코드']
            values = [j for j in company_code]

            dict_result = dict(zip(keys, values))  # 딕셔너리 형태로 회사이름과 종목코드 묶기
            dict_result2 = dict(zip(values, keys))

            pattern = '[a-zA-Z가-힣]+'

            if bool(re.match(pattern, company)) == True:  # Input에 이름으로 넣었을 때
                company_code = dict_result.get(str(company))
                crawler(company_code, maxpage)
                return company, company_code

            else:  # Input에 종목코드로 넣었을 때
                company_name = dict_result2.get(str(company))
                company_code = str(company)
                crawler(company_code, maxpage)
                return company_name,company_code

        def clean_str(string):
            string = re.sub('([ㄱ-ㅎㅏ-ㅣ])+', " ", str(string))
            string = re.sub('<[^>]*>', " ", str(string))
            string = re.sub('[^\w\s]', " ", str(string))
            string = re.sub(re.compile(r'\s+'), " ", str(string))
            return string
        def tokenize_str(sentence_list):
            token_list = []
            for sentence in sentence_list:
                words = sentence.split(' ')
                for word in words:
                    word = word.rstrip().lstrip()
                    if word != '':
                        token_list.append(word)
            return token_list

        company_name, company_code =convert_to_code(special_code, 400)
        clean_list = [clean_str(sentence) for sentence in total_list]
        tokens = tokenize_str(clean_list)
        text = ""
        for token in tokens:
            text = text + token + " "

        wordcloud = WordCloud(font_path='./joosajang/test4/static/test4/KOTRA_BOLD.ttf',
                              max_font_size=45,
                              background_color='white').generate(text)
        fig = plt.gcf()
        plt.imshow(wordcloud, interpolation='lanczos')
        plt.axis('off')
        buf = io.BytesIO()

        fig.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = 'data:image/png;base64,' + urllib.parse.quote(string)



        count_dict = {}
        for token in tokens:
            if token in count_dict.keys():
                count_dict[token] += 1
            else:
                count_dict[token] = 1

        word = np.array(list(count_dict.keys()))
        count = np.array(list(count_dict.values()))
        index = count.argsort()[::-1][1:6]
        table = list(word[index])

        # print(today_str,company_name,company_code,table)


        return render(request, 'test4/result_classify2_2.html',{'uri':uri,'today':today_str,'name':special_code,'code':company_code,'table_0':table[0],'table_1':table[1],'table_2':table[2],'table_3':table[3],'table_4':table[4]})