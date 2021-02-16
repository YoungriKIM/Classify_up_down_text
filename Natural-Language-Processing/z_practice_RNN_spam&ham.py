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
data['v1'].value_counts().plot(kind='bar'); #plt.show()
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
total_cnt = len()
