#!/usr/bin/env python
# coding: utf-8

# ### 문제
# * 1시간별 일주일 예측 발전량
# * 일별 발전량
# * 발전소 총 발전량

# ### 태양광 발전량 예측의 필요성
# #### 전력 생산 균형을 맞추고, 전력 공급 계획을 효율적으로 세우기위해서
# #### 잘못 예측 시, 많은 연료와 운영비 소비
# 

# In[1]:


get_ipython().system('sudo apt-get install -y fonts-nanum')
get_ipython().system('sudo fc-cache -fv')
get_ipython().system('rm ~/.cache/matplotlib -rf')
# !apt-get update -qq
# !apt-get install fonts-nanum* -qq


# In[2]:


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # 폰트 관련 용도

# path = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
# fontName = fm.FontProperties(fname=path, size=9)

# fontName
# plt.rcParams['font.family'] = font_name
plt.rc('font', family='NanumBarunGothic')
plt.rcParams['axes.unicode_minus'] = False


# In[4]:


import tensorflow as tf
import pandas as pd
import numpy as np

import requests
import re
import json


# 차트 보여줄 시, 한글깨짐 방지
# plt.rcParams['font.family'] = 'NanumBarunGothic'
# plt.rcParams['font.family'] = 'NanumBarunGothic'
# plt.rcParams['font.family'] = 'Malgun Gothic'
import seaborn as sns

from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import LSTM
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler

# from pandas.core.indexes.datetimes import date
# from pandas.io.json import json_normalize

# from urllib.request import Request, urlopen
# from urllib.parse import urlencode, quote_plus

from zipfile import ZipFile


# In[5]:


solarDf = pd.read_csv('/content/sample_data/[006] [전라남도 강진군 신전면 영관리]-data-2022-02-21 16_10_32.csv', index_col= 'Time', parse_dates = True)

# solarDf.rename(columns={'deviceNum' : '디바이스ID', 'siteID' : '발전소ID', 'acA' : '교류전류량', 'acV' : '교류전압량' , 'capacity' : '전력출력량', 'cumulativePower' : '전력누적량', 'dailyPower' : '하루발전량', 'dcA' : '직류전류량', 'dcV':'직류전압', 'dcKW' : '직류전력량'}, inplace=True)

solarDf
# acA : 암페어(A) 단위로 나타낸 교류 전류량
# acV : 볼트(V) 단위로 나타낸 교류 전압량
# capacity : 최대전력 출력량
# cumulativePower : 전력 누적량
# dailyPower : 하루 발전량
# dcA : 암페어 단위로 나타낸 직류 전류량
# dcV : 볼트 단위로 나타낸 직류 전압
# dcKW: Kw 단위로 직류 전력을 표현


# In[ ]:


# plt.figure(figsize=(15,10))
solarCols = solarDf.columns.values.tolist()

solarDf.plot(subplots=True, figsize=(8,15))
# plt.plot(solarDf['dailyPower'], marker='.', label='하루 발전량', color = 'blue')


# ### 데이터 전처리

# ### 디바이스 번호별로 데이터 분리
# * 데이터프레임 및 그래프로 표시

# In[6]:


# 데이터 전처리 후 group by 진행하기
groups = solarDf.groupby('deviceNum') # 디바이스별로 그룹화

devices = []
print('그룹 길이', len(groups))
for index, group in enumerate(groups):
  print(index)
  globals()['device{}'.format(index+1)]= group[1]
  devices.append(group[1].resample(rule='H').last()) # 초별 => 1시간별로 리샘플링 작업

colors = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
]

plt.figure(figsize=(15,5))
for index, device in enumerate(devices):
  print(device)
  deviceNo = '디바이스' + str(index+1)
  plt.plot(device['dailyPower'], marker='.', label=deviceNo, color = colors[index])
plt.legend()


# In[18]:


solarBydevice1 = devices[0]

solarBydevice1.shape


# In[ ]:


solarDf.describe()


# ### 결측치 확인 및 제거

# In[ ]:


solarBydevice1.isnull().sum()


# In[19]:


solarBydev1Df = solarBydevice1.fillna(solarBydevice1.mean())

solarBydev1Df.isnull().sum()


# ### 선형 보간법

# In[ ]:


start_date = pd.to_datetime('2019-06-14 20:00')
end_date = pd.to_datetime('2020-01-01 7:00')

emptyPeriod = solarBydevice1.loc[start_date:end_date]

# 이동평균
# emptyPeriod =  device.rolling(30).mean()

# 보간법
# interpolatedDf = emptyPeriod.interpolate(method='linear')
# device.update(interpolatedDf)
# device.reset_index(inplace=True)

# plt.plot(device.index, device['dailyPower'], color='blue')


# ### 기상청 API 데이터를 csv로  저장

# In[ ]:


## 요청 예시 : http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList?serviceKey=인증키&numOfRows=10&pageNo=1&dataCd=ASOS&dateCd=HR&stnIds=108&endDt=20200310&endHh=01&startHh=01&startDt=20190120
domain = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList?"
serviceKey = "serviceKey=UgjoxFVmY0P3xDE0K5OAKlL2uA288oEZ7Vy535AW30B1M4YxUrsLQGkYC5BlWMZG0FBnf2L%2FQEsYQJalwUgZhg%3D%3D&"
option = "dataCd=ASOS&dateCd=HR&dataType=JSON&numOfRows=999&"
stnIds = "stnIds=259&"
date = "endDt=20220131&endHh=18&startHh=07&startDt=20190101&"
pageNo = "pageNo="


url =  domain + serviceKey + option + stnIds + date
climateApiResult = ''

# 데이터 총 갯수 확인(totalCount)
# response = requests.get(url)
# respnse.text

for i in range(1, 29):
    url =  domain + serviceKey + option + stnIds + date + pageNo + str(i)
    response = requests.get(url)
    response.content.decode('utf-8')
    climateApiResult = climateApiResult + response.text

climateApiResult

json_obj = json.loads(climateApiResult)

json_obj

climateData = json_obj['response']['body']['items']['item']


climateDf = pd.json_normalize(climateData) # json => df
climateDf
climateDf.tail() # 데이터 확인


# df2.to_csv('./climateApiData.csv') # CSV 저장


# ### 기상 데이터 csv로 조회
# * 2020-06-15 ~ 2020-12-31

# In[20]:


climateDf = pd.read_csv('/content/sample_data/climateApiData.csv',index_col= 'tm', parse_dates = True)

climateDf.head()
# climateDf.describe()


# In[ ]:


climateDf.shape


# ### 기상 데이터 기간 조회

# In[ ]:


# ctest = climateDf[climateDf.index.between('2020-06-14 20:00','2021-01-01 7:00')]
# deviceTest = devices[0][(devices[0]['Time'] <= '2020-06-14 20:00') &( devices[0]['Time'] >= '2021-01-01 7:00')]
start_date = '2020-06-14 20:00'
end_date = '2021-01-01 07:00'

# 값 확인
climateDf.index = pd.to_datetime(climateDf.index)
# c2 = climateDf[(climateDf.index >= '2020-06-14 20:00') &( climateDf.index <= '2021-01-01 7:00')]
# c2.shape
# selected_data = climateDf.loc[start_date:end_date]
# type(selected_data)

climateDf2 = climateDf[(climateDf.index <= start_date) | ( climateDf.index >= end_date)]
climateDf2

# dropedClimateDf = climateDf.drop(climateDf.index[start_date:end_date] , axis = 0, inplace = True)
# dropedClimateDf

# climateDf.loc[start_date:end_date]


# ### 기상 데이터 결측치 확인 및 제거

# In[21]:


climateDf.describe()
climateDf.isnull().sum()


# In[22]:


climateDf.drop(['hr3Fhsc'], axis=1, inplace=True)
climateDf.drop(['clfmAbbrCd'], axis=1, inplace=True)
climateDf.drop(['gndSttCd'], axis=1, inplace=True)
climateDf.drop(['dmstMtphNo'], axis=1, inplace=True)

climateDf.drop(labels='stnNm', axis=1, inplace=True)
climateDf.drop(labels='stnId', axis=1, inplace=True)


# In[23]:


climateDf = climateDf.fillna(climateDf.mean())

print(climateDf.isnull().sum())


# ### 데이터통합

# In[24]:


climateDf.index.name = 'Time'

df = pd.merge(solarBydev1Df, climateDf, on='Time', how="left")
# df2 = pd.merge(devices[0], climateDf, left_index=True, right_index=True, how = "inner")
df.head()
# plt.plot(df['dailyPower'], label='dailyPower')


# In[ ]:


# df.shape
df.shape


# In[ ]:


# df.isnull().sum()
df.isnull().sum()


# In[25]:


df = df.fillna(df.mean())

df.isnull().sum()


# In[26]:


df.hist(figsize=(15,20)) # https://www.boostcourse.org/ds112/lecture/60078?isDesc=false 참고


# ### 상관관계 분석

# In[ ]:


df_corr = df.corr()
df_corr_sort = df_corr.sort_values('dailyPower', ascending = False)
df_corr_sort['dailyPower']

# ss : 일조
# icsr : 일사
# ws : 풍속
# ts : 지면온도
# ta : 기온


# In[ ]:


# df.reset_index(drop=False)
dfCoulmns = ['Time', 'ss','icsr', 'ws','ts','ta','dailyPower']
df = df.reset_index(drop=False)
df = df[dfCoulmns]
df


# In[ ]:


# ss : 일조
# icsr : 일사
# ws : 풍속
# ts : 지면온도
# ta : 기온
labelDf = df

labelDf.rename(columns={'Time':'시간', 'ss' : '일조', 'icsr':'일사', 'ws':'풍속','ts':'지면온도','ta':'기온', 'dailyPower' : '하루 발전량'}, inplace = True)

df_corr = labelDf.corr()
plt.figure(figsize = (16,6))

sns.heatmap(df_corr, annot = True, linewidths=1, fmt='.2%', cmap='coolwarm')

plt.xticks(rotation='horizontal')


# In[ ]:


scaledDf.shape


# ### 데이터 정규화

# In[ ]:


X = df.drop(columns = ['dailyPower'])
y = df['dailyPower']


# In[ ]:


scaler = MinMaxScaler(feature_range = (0,1))

def normalize(X,y):
  scaledX =  X.copy()

  for name in X:
    temp = X[name].to_numpy().reshape(-1,1)
    print('name', name)
    scaledX[name] = scaler.fit_transform(temp)

  temp = y.to_numpy().reshape(-1,1)
  scaledY = scaler.fit_transform(temp)

  return scaledX, scaledY


# In[ ]:


scaledX, ydata = normalize(X,y)
xData = scaledX.to_numpy()

xData.shape, y


# In[ ]:


# tf.random.set_seed(13)

def multivariate_data(dataset, target, startNum, endIndex, steps, future, step, single_step=False):
  data = []
  labels = []

  startNum = startNum + steps
  if endIndex is None:
    endIndex = len(dataset) - future

  for i in range(startNum, endIndex):
    indices = range(i-steps, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+future])
    else:
      labels.append(target[i:i+future])

  return np.array(data), np.array(labels)


# In[ ]:


steps = 336 # 30일
future = 168 # 7일
STEP = 1

trainSplit = int(xData.shape[0]*0.7)


X_train, y_train = multivariate_data(xData, ydata, 0, trainSplit, steps, future, STEP, single_step = False )
X_test, y_test = multivariate_data(xData, ydata, trainSplit, None, steps, future, STEP, single_step = False )

X_train.shape, X_test.shape


# ### 데이터 모델 생성 및 학습
# * Sequentail Model => lstm
# * TimeseriesGenerator(Timeseries)

# In[ ]:


model = Sequential()

model.add(LSTM(units = 32,  return_sequences = True,  input_shape = X_train.shape[-2:])) # ,  return_sequences = True
model.add(LSTM(units = 32))
model.add(Dense(168))
model.summary()

print(X_train.shape[-2:])


# In[ ]:


modelRelu = Sequential()

modelRelu.add(LSTM(units = 32, input_shape = X_train.shape[-2:], activation = 'relu')) # ,  return_sequences = True
# model.add(LSTM(units = 32))
modelRelu.add(Dense(168))
modelRelu.summary()

print(X_train.shape[-2:])


# In[ ]:


modelSigmoid = Sequential()

modelSigmoid.add(LSTM(units = 32, input_shape = X_train.shape[-2:], activation = 'sigmoid')) # ,  return_sequences = True
# model.add(LSTM(units = 32))
modelSigmoid.add(Dense(168))
modelSigmoid.summary()

print(X_train.shape[-2:])


# In[ ]:


model.compile(loss='mse', optimizer = 'adam', metrics=['mae'])


# In[ ]:


modelRelu.compile(loss='mse', optimizer = 'adam', metrics=['mae'])
modelSigmoid.compile(loss='mse', optimizer = 'adam', metrics=['mae'])


# In[ ]:


BATCH_SIZE = 32
BUFFER_SIZE = 1000

trainData = tf.data.Dataset.from_tensor_slices((X_train, y_train)).cache().batch(BATCH_SIZE)
testData = tf.data.Dataset.from_tensor_slices((X_test, y_test)).cache().batch(BATCH_SIZE)


# In[ ]:


history = model.fit(trainData, epochs = 5, batch_size = BATCH_SIZE)


# In[ ]:


historyRelu = modelRelu.fit(trainData, epochs = 5, batch_size = BATCH_SIZE)
historySigmoid = modelSigmoid.fit(trainData, epochs = 5, batch_size = BATCH_SIZE)


# In[ ]:


ax1 = plt.subplot(3,1,1)
plt.title('tanh loss & mae')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['mae'], label='mae')
# plt.xticks(visible=False)

ax2 = plt.subplot(3,1,2)
plt.title('Relu loss & mae')
plt.plot(historyRelu.history['loss'], label='loss')
plt.plot(historyRelu.history['mae'], label='mae')


ax3 = plt.subplot(3,1,3)
plt.title('Sigmoid loss & mae')
plt.plot(historySigmoid.history['loss'], label='loss')
plt.plot(historySigmoid.history['mae'], label='mae')

plt.subplots_adjust(hspace=1)
plt.legend()
plt.show()


# In[ ]:


plt.plot(history.history['loss'], label='loss')
plt.title('loss & mae')
plt.plot(history.history['mae'], label='mae')
plt.legend()
plt.show()


# In[ ]:


testLoss = model.evaluate(X_test, y_test)


# ### 모델 예측

# In[ ]:


pred = model.predict(X_test)


# In[ ]:


print(pred)

print(X_test.shape)
print(y_test.shape)

print(y_test[0].shape)
# print(pred[::-steps])


# In[ ]:


print(pred[0].shape)
print(y_test[0].shape)

# print(pred[0])
print(y_test[0])


# In[ ]:


pred = pred[0]
actual = y_test[0]

actual = scaler.inverse_transform(y_test[0])
# predict = scaler.inverse_transform(pred[0])

actual
pred
# plt.title('comparison')

# plt.plot(pred, label='predict', color = 'red')
# plt.plot(actual, label='actual', color = 'blue')
# plt.legend()
# plt.show()


# In[ ]:





# In[ ]:


plt.title('comparison')


pred.shape

plt.plot(pred[0][-steps:], label='predict', color = 'red')
plt.plot(y_test[0][-steps + future : -steps + future], label='actual', color = 'blue')
plt.legend()
plt.show()


# ### 모델 생성 2
# * 참고 URL  : https://www.youtube.com/watch?v=uw6zYLbCGkY&t=588s

# In[ ]:


model = Sequential()

model.add(LSTM(units = 32, input_shape = [720,6],  return_sequences = True)) #
# model.add(LSTM(units = 32))
model.add(Dense(168))
model.summary()


# In[ ]:


model.compile(loss='mse', optimizer = 'adam', metrics=['mae'])

# earlyStop = EarlyStopping(monitor='val_loss', patience = 5)


# ### 모델 학습

# In[ ]:


# print(x.shape)
# print(y.shape)

history = model.fit(generatorTrain, batch_size = 32, epochs = 5, verbose = 1)


# In[ ]:


predictions = []
firstSequence = trainSet[-steps:]

currentSequence = firstSequence.reshape((1, steps, 6))


for i in range(len(testSet)):
  currentPred = model.predict(currentSequence)[0]
  predictions.append(currentPred)
  currentBatchStep = currentSequence[:,1:,:]
  currentSequence = np.append(currentBatchStep,[[currentPred]], axis = 1)


# In[ ]:


inversedPredictoin = scaler.inverse_transform(predictions)
inversedTestSet = scaler.inverse_transform(test)

plt.plot(inversedPredictoin[:,0])
plt.plot(inversedTestSet[:,0])


# ### 모델 실행 및  예측

# In[ ]:


loss = history.history['loss']
mae = history.history['mae']
epochs = range(1, len(loss) + 1)

plt.figure()
plt.plot(epochs, loss, 'b-', label='loss')
plt.title('loss')
plt.plot(epochs, mae, 'b-', label='mae', color = 'red')
plt.title('mae')
plt.legend()

plt.show()


# ### 모델 실행

# In[ ]:


predX = scaledDf[-steps:,] # 마지막 30일 데이터

testSet = np.reshape(predX, (1,steps,len(scaledDf[0,:])))

pred = model.predict(testSet)


# In[ ]:


df[-steps:,]


# ### 모델 평가

# In[ ]:


result = model.evaluate(testSet)


# In[ ]:





# In[ ]:


inversePred = pred * (max-min) + min


# In[ ]:


plt.figure(figsize=(15,5))
plt.plot(df[-7:-1], label='actual', color='black')
plt.plot(inversePred[0][0:7], label ='predict', color='blue')
plt.legend()

