# 스팸메일 분류 RNN(Vanila RNN)
# 다운로드 링크 : https://www.kaggle.com/uciml/sms-spam-collection-dataset


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request   # url 통해 바로 다운 받게 하는 기능/ 이미 다운 받은 것 있어서 사용 안 함
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data = pd.read_csv('../NLP/practice_data/spam.csv', encoding = 'latin1')

print('총 샘플의 수: ', len(data))
# 총 샘플의 수:  5572

print('상위 샘플: \n', data.head())                        #___________________________________ > 필요없는 특성
#       v1                                                 v2 Unnamed: 2 Unnamed: 3 Unnamed: 4
# 0   ham  Go until jurong point, crazy.. Available only ...        NaN        NaN        NaN
# 1   ham                      Ok lar... Joking wif u oni...        NaN        NaN        NaN
# 2  spam  Free entry in 2 a wkly comp to win FA Cup fina...        NaN        NaN        NaN
# 3   ham  U dun say so early hor... U c already then say...        NaN        NaN        NaN
# 4   ham  Nah I don't think he goes to usf, he lives aro...        NaN        NaN        NaN

# ------------------------------------------------------------------------------------
# 필요없는 unnamed 3개 삭제
del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']

# ham, spam 0/1로 변경
data['v1'] = data['v1'].replace(['ham', 'spam'], [0,1])

# 상위 샘플 봐서 확인
print('상위 샘플: \n', data.head())
# 확인 완료!

# 바뀐 데이터의 정보를 확인해보자
print(data.info())

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 5572 entries, 0 to 5571
# Data columns (total 2 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   v1      5572 non-null   int64        # 정수형
#  1   v2      5572 non-null   object       # 문자열
# dtypes: int64(1), object(1)
# memory usage: 87.2+ KB

# ------------------------------------------------------------------------------------
# Null값 있는지 확인
print(data.isnull().values.any())
# False : Null값 없음

# 중복값 있는지 확인(유일한 값 개수 찾기)
print(len(data['v2'].unique()))     # 5169 > 위 info에서 5572개 였으니 중복값이 403개
print(len(data['v1'].unique()))     # 2    > 중복값 없음

# 중복값 제거
data.drop_duplicates(subset=['v2'], inplace=True)
print('중복값 제거 후 총 샘플 수: ', len(data))
# 중복값 제거 후 총 샘플 수:  5169   (제거 전: 5572개)

# ------------------------------------------------------------------------------------
# spam, ham 레이블 값의 분포를 시각화
# data['v1'].value_counts().plot(kind='bar'); #plt.show()
# 레이블이 대부분 0에 편중되어있다.

# 수치로 확인해보자
print(data.groupby('v1').size().reset_index(name='count'))
#    v1  count
# 0   0   4516
# 1   1    653

# ------------------------------------------------------------------------------------
# x y 데이터 분리
x_data = data['v2']
y_data = data['v1']
print('메일 본문의 개수: {}'.format(len(x_data)))
print('스팸 레이블의 개수: {}'.format(len(y_data)))
# 메일 본문의 개수: 5169
# 스팸 레이블의 개수: 5169

# ------------------------------------------------------------------------------------
# 토큰화와 정수 인코딩 진행
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_data) # 5196개의 행을 가진 x의 각 행에 토큰화를 수행
sequences = tokenizer.texts_to_sequences(x_data)    # 단어를 숫자값으로 인덱싱하여 저장
print(sequences[:5])    # 인덱싱 확인
# [[47, 433, 4013, 780, 705, 662, 64, 8, 1202, 94, 121, 434, 1203, 142, 2712, 1204, 68, 57, 4014, 137],\
#  [49, 306, 1364, 435, 6, 1767],\
# [53, 537, 8, 20, 4, 1016, 934, 2, 220, 2713, 1365, 706, 2714, 2715, 267, 2716, 70, 2713, 2, 2717, 2, 359, 537, 604, 1205, 82, 436, 185, 707, 437, 4015],\
#  [6, 226, 152, 23, 347, 2718, 6, 138, 145, 56, 152],\
#  [935, 1, 97, 96, 69, 453, 2, 877, 69, 1768, 198, 105, 438]]

# 정수가 어떤 단어에 부여되었는지 확인
word_to_index = tokenizer.word_index
# print(word_to_index)
# {'i': 1, 'to': 2, 'you': 3, 'a': 4, 'the': 5, 'u': 6, 'and': 7, ... 생략

# ------------------------------------------------------------------------------------
# tokenizer.word_counts.items()를 이용해 각 단어에 대한 등장 빈도수 확인 가능
# 이를 이용해 빈도수가 낮은 단어들이 훈련데이터에서 얼만큼의 비중을 차지하는지 알아보자

threshold = 2   # threshold: 한계점
total_cnt = len(word_to_index)
rare_cnt = 0    # 등장 빈도수가 threshold 보다 작은 단어의 개수를 카운트
total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0   # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value
    
    # 단어의 등장 빈도수가 threshold(2) 보다 작은 단어의 수
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('등장 빈도가 %s번 이하인 중복 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합(vocabulary)에서 중복 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 중복 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)
# 등장 빈도가 1번 이하인 중복 단어의 수: 4908
# 단어 집합(vocabulary)에서 중복 단어의 비율: 55.02242152466368
# 전체 등장 빈도에서 중복 단어 등장 빈도 비율: 6.082538108811501

# 중복되어 등장하는 횟수가 4908번이나 되는데 막상 전체 단어에서 중복 단어의 비율은 6%뿐이라는 결론
# 미리 토큰화를 할 때 등장 빈도가 1회인 단어들은 제외 할 수 있다.
        # tokenizer = Tokenizer(num_words = total_cnt - rare_cnt +1 )
# 이번에는 우선 진행해보겠음

# ------------------------------------------------------------------------------------
# 단어 집합의 크기를 vocab_size에 저장 (패딩을 위한 토큰인 0번 단어를 고려해 +1 해야 함)
vocab_size = len(word_to_index) + 1
print('단어 집합의 크기: {}'.format((vocab_size)))
# 단어 집합의 크기: 8921

# ------------------------------------------------------------------------------------
# 훈련데이터와 테스트데이터 8:2로 분리

# 8:2 개수 계산
n_of_train = int(len(sequences) * 0.8)
n_of_test = int(len(sequences) - n_of_train)
print('훈련 데이터의 개수: ', n_of_train)
print('테스트 데이터의 개수: ', n_of_test)
# 훈련 데이터의 개수:  4135
# 테스트 데이터의 개수:  1034

# x_data로 토큰화>정수인코딩까지한 결과인 sequences를 X_data로 지정
X_data = sequences

# 전체 데이터의 길이가 가장 긴 메일과 전체 메일데이터의 길이 분포 확인
X_data = sequences
print('메일의 최대 길이 : %d' % max(len(l) for l in X_data))
print('메일의 평균 길이 : %f' % (sum(map(len, X_data))/len(X_data)))
# 메일의 최대 길이 : 189
# 메일의 평균 길이 : 15.610370

# ------------------------------------------------------------------------------------
# 패딩
max_len = 189   # 전체 데이터셋의 길이는 max_len으로 맞추고
data = pad_sequences(X_data, maxlen = max_len)
print('훈련 데이터 shape: ', data.shape)
# 훈련 데이터 shape:  (5169, 189)

# train, test 분리
x_train = data[:n_of_train]
x_test = data[n_of_train:]
y_train = np.array(y_data[:n_of_train])
y_test = np.array(y_data[n_of_train:])

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# (4135, 189)
# (1034, 189)
# (4135,)
# (1034,)

# ------------------------------------------------------------------------------------
# 모델 구성
# 전에 임베딩 레이어 설명 : Embedding()은 단어를 밀집 벡터로 만드는 역할
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense

model = Sequential()
model.add(Embedding(vocab_size, 32))        # (단어 집합의 크기, 임베딩 벡터의 차원)
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# 평가
print('\n acc: %.4f' % (model.evaluate(x_test, y_test)[1]))
#acc: 0.9778

# ------------------------------------------------------------------------------------
# 그래프로 그려보기
epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# 표 확인까지 완료!