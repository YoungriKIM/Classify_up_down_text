# # 태깅
# okt 말고
# kkma/ komoran로 바꾼 파일 

####### kkma.morphs 사용하기로 함!!!

# ------------------------------------------------------------
import numpy as np
import pandas as pd

# up, down 불러와서 합치자
# up
up_data = pd.read_csv('../NLP/save/up_data_01.csv', index_col=0)
print('load한 존대말 데이터: \n', up_data[-5:])
# down
down_data = pd.read_csv('../NLP/save/down_data_01.csv', index_col=0)
print('load한 반말 데이터: \n', down_data[-5:])
# 각 데이터의 수 확인
print('존댓말 데이터의 수: ', len(up_data), '\n반말 데이터의 수: ', len(down_data))
# 존댓말 데이터의 수:  1750
# 반말 데이터의 수:  2330

# 두 데이터 합하기
all_data = pd.concat([up_data, down_data])
print('shape of all_data: ', all_data.shape)
# shape of all_data:  (4080, 2)

# ------------------------------------------------------------
# 전처리 전에 pandas_profiling 으로 데이터 분석
# import pandas_profiling

# profile_all_data = all_data.profile_report()
# profile_all_data.to_file('../NLP/sample_data/profile_all_data.html')
# print('===== save complete =====')


# ------------------------------------------------------------
# 본격적인 전처리 시작

# 불용어 불러오기(import_stopword.py 파일 참고)
f = open('../NLP/sample_data/stopword_02.txt','rt',encoding='utf-8')  # Open file with 'UTF-8' 인코딩
text = f.read()
stopword = text.split('\n') 

# 품사 태깅으로 토큰화 (품사로 나눈 토큰화)
# from konlpy.tag import Kkma
# tokenizer = Kkma()
from konlpy.tag import Okt, Kkma, Komoran
okt = Okt()
kkma = Kkma()
komo = Komoran()

tag_data = []
for sentence in all_data['data']:
    temp_x = []
    temp_x = kkma.morphs(sentence)
    temp_x = [word for word in temp_x if not word in stopword]
    tag_data.append(temp_x)

# 확인용 출력
print('토큰화 된 샘플: ', tag_data[-5:])

# ------------------------------------------------------------
# 0번짜리 패딩과 OOV 토큰 고려하여 단어 집합의 크기에서 +2
# 단어 집합(vocabulary)의 크기 : 4998
vocab_size = 4998 + 2

# ------------------------------------------------------------
# 정수 인코딩
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()

# 인코딩 적용
tokenizer = Tokenizer(vocab_size, oov_token='OOV')
tokenizer.fit_on_texts(tag_data)
tag_data = tokenizer.texts_to_sequences(tag_data)

# 인코딩 확인을 위해 샘플 출력
print(tag_data[:3])
# [[1933, 509, 1133, 1934], [266, 1935], [267, 1134, 266, 3, 220, 365, 1936]]

# ------------------------------------------------------------
# x, y 로 지정해서 진행
x = tag_data
y = all_data['label']
# 한 번 확인
# print(len(x))     # 4080
# print(len(y))     # 4080

# ------------------------------------------------------------
# 패딩(서로 다른 길이의 샘플을 동일하게 맞추기)

# 길이 15으로 정해서 함수 돌리기
max_len = 15

# 패딩 적용
from tensorflow.keras.preprocessing.sequence import pad_sequences
x = pad_sequences(x, maxlen = max_len, padding='pre') 
# 확인
print(x[0])
# [   0    0    0    0    0    0    0    0    0    0    0 1933  509 1133  1934]

# ------------------------------------------------------------
# train, test 나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1)
# 확인
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)
# (3264, 15)
# (3264,)
# (816, 15)
# (816,)

# ------------------------------------------------------------
# 모델 구성
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 임베딩 벡터 차원 100으로 정하고 lstm 이용
model = Sequential()
model.add(Embedding(vocab_size, 50))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

# callbacks 정의
stop = EarlyStopping(monitor='val_loss', mode = 'min', verbose=1, patience=8)
file_path = '../NLP/modelcheckpoint/project_07kkma-2.h5'
mc = ModelCheckpoint(filepath= file_path, monitor='val_acc', mode = 'max', save_best_only=True, verbose=1)

# 컴파일, 훈련
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# history = model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.2, callbacks=[stop, mc])

# ------------------------------------------------------------
# 정확도가 가장 높은 가중치 가져와서 적용
loaded_model = load_model('../NLP/modelcheckpoint/project_07kkma-2.h5')
print('===== save complete =====')
print('loss: %.4f' % (loaded_model.evaluate(x_test, y_test)[0]), '\nacc: %.4f' % (loaded_model.evaluate(x_test, y_test)[1]))

# ------------------------------------------------------------
# 전에 predict에 넣을 것도 전처리 똑같이 해줘야 겠지?
import re

def sentiment_predict(new_sentence):
    # 전처리
    new_sentence = re.compile("[^ㄱ-ㅎㅏ-ㅣ가-힣]").sub(' ',new_sentence)
    new_sentence = kkma.morphs(new_sentence)  # 토큰화
    new_sentence = [word for word in new_sentence if not word in stopword] # 불용어 제거
    encoded = tokenizer.texts_to_sequences([new_sentence])  # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen = max_len, padding='pre')  # 패딩

    # 예측
    score = float(loaded_model.predict(pad_new))
    if(score > 0.5):
        print(new_sentence, '는 {:.2f} % 확률로 존댓말입니다.'.format(score * 100))
    else:
        print(new_sentence, '는 {:.2f} % 확률로 반말입니다.'.format((1 - score) * 100))


# =========================================================
# 예측하기

# project_07kkma.h5
# loss: 0.0481 
# acc: 0.9828

# ['겨울', '지나가', '었', '다'] 는 99.22 % 확률로 반말입니다.
# ['이렇', '게', '반말', '어도', '는', '모르', '겠', '지', 'ㅋㅋㅋ'] 는 91.41 % 확률로 반말입니다.
# ['머신', '알', '아서', '잘하'] 는 96.06 % 확률로 반말입니다.
# ['벌써', '금요일', '라니', '말', '도', '안', '되'] 는 62.46 % 확률로 반말입니다.
# --------------------------------------------
# ['겨울', '지나가', '었', '습니다'] 는 99.99 % 확률로 존댓말입니다.
# ['겨울', '은', '지나가', '었', '다', 'ㅂ시다'] 는 92.47 % 확률로 존댓말입니다.
# ['아버지', '가방', '들어가', '시', 'ㄴ다'] 는 70.35 % 확률로 존댓말입니다.
# ['배', '고프', '지', '않', '나요'] 는 96.87 % 확률로 반말입니다.
# ['이러', 'ㄹ', '배', '고프', '지요'] 는 78.70 % 확률로 존댓말입니다.
# ['이러', 'ㄹ', '배', '고프', '죠'] 는 63.41 % 확률로 존댓말입니다.
# ['는', '배', '고프', '아요'] 는 99.95 % 확률로 존댓말입니다.
# ['어서', '점심', '시키', '세요'] 는 100.00 % 확률로 존댓말입니다.


# project_07kkma.h5
# loss: 0.0306
# acc: 0.9914

# ['겨울', '지나', '갔', '다'] 는 99.25 % 확률로 반말입니다.
# ['내', '이렇', '게', '반말', '아도', '넌', '모르겠지ㅋㅋㅋ'] 는 94.57 % 확률로 반말입니다.
# ['머신', '알', '아서', '잘'] 는 99.59 % 확률로 반말입니다.
# ['벌써', '금요일', '라니', '말', '도', '안', '되'] 는 85.39 % 확률로 반말입니다.
# --------------------------------------------
# ['겨울', '지나', '갔', '습니다'] 는 99.97 % 확률로 존댓말입니다.
# ['겨울', '은', '지나', '갔', '다', 'ㅂ시다'] 는 64.15 % 확률로 존댓말입니다.
# ['아버지', '가방', '들어가', '시', 'ㄴ다'] 는 68.86 % 확률로 존댓말입니다.
# ['배', '고프', '지', '않', '나요'] 는 97.21 % 확률로 반말입니다.
# ['이렇', 'ㄹ', '배', '고프', '지요'] 는 78.84 % 확률로 반말입니다.
# ['이렇', 'ㄹ', '배', '고프', '죠'] 는 58.16 % 확률로 반말입니다.
# ['는', '배', '고프', '아요'] 는 99.94 % 확률로 존댓말입니다.
# ['어서', '점심', '시키', '시', '어요'] 는 99.99 % 확률로 존댓말입니다.


# =========================================================
# sentiment_predict('겨울이 지나갔다.')
# sentiment_predict('내가 이렇게 반말해도 넌 모르겠지ㅋㅋㅋ')
# sentiment_predict('머신아 알아서 잘하자?')
# sentiment_predict('벌써 금요일이라니 말도 안 돼!')
# print('--------------------------------------------')
# sentiment_predict('겨울이 지나갔습니다.')
# sentiment_predict('겨울은 지나갔다 합시다..')
# sentiment_predict('아버지가방에들어가신다.')
# sentiment_predict('배 고프지 않나요?')
# sentiment_predict('이럴때 배가 고프지요.')
# sentiment_predict('이럴때 배가 고프죠.')
# sentiment_predict('저는 배가 고파요.')
# sentiment_predict('어서 점심을 시키세요.')

