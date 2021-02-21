# 파라미터 튜닝을 위한 파일
# 기준 파일은 p_010

# ------------------------------------------------------------
import numpy as np
import pandas as pd

# up, down 불러와서 합치자
# up
up_data = pd.read_csv('../NLP/save/up_data_02.csv', index_col=0)
# print('load한 존대말 데이터: \n', up_data[-5:])
# down
down_data = pd.read_csv('../NLP/save/down_data_02.csv', index_col=0)
# print('load한 반말 데이터: \n', down_data[-10:-5])
# 각 데이터의 수 확인
# print('존댓말 데이터의 수: ', len(up_data), '\n반말 데이터의 수: ', len(down_data))
# 존댓말 데이터의 수:  1750 > 2830
# 반말 데이터의 수:  2330 > 2870

# 두 데이터 합하기
all_data = pd.concat([up_data, down_data])
# print('shape of all_data: ', all_data.shape)
# shape of all_data:  (5700, 2)

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
    temp_x = komo.morphs(sentence)
    temp_x = [word for word in temp_x if not word in stopword]
    nounlist = komo.nouns(sentence)
    temp_x = [noun for noun in temp_x if not noun in nounlist]
    tag_data.append(temp_x)

# 확인용 출력
# print('토큰화 된 샘플: ', tag_data[-10:-5])


# ------------------------------------------------------------
# p_09 에서 추가한 부분

# 등장 빈도수 확인하기 전 인코딩하여 단어에 숫자부여
# 정수 인코딩(텍스트를 숫자로)
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tag_data)
# print('단어에 정수 부여 확인\n', tokenizer.word_index)
count_item = tokenizer.word_counts.items()


# 등장 빈도수 1회인 단어의 비중 확인
threshold = 2                           # 횟수 제한을 위한 수 지정
total_cnt = len(tokenizer.word_index)   # 전체 단어 집합의 개수
rare_cnt = 0                            # 빈도수가 1회인 단어의 개수
total_freq = 0                          # 전체 등장 빈도 여기서는 1205번
rare_freq = 0                           # 빈도수가 1회인 단어의 빈도

# 단어-빈도수의 쌍을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value
    
    # 단어 등장 빈도수가 threshold(2) 미만이면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value
        
# print('전체 단어 집합의 개수 :',total_cnt)
# print('빈도수가 %s번 이하인 rare 단어의 개수: %s'%(threshold - 1, rare_cnt))
# print('단어 집합에서 rare 단어의 비율:', (rare_cnt / total_cnt)*100)
# print('전체 등장 빈도에서 rare 단어 등장 빈도율:', (rare_freq / total_freq)*100)
# 전체 단어 집합의 개수 : 1391
# 빈도수가 1번 이하인 rare 단어의 개수: 606
# 단어 집합에서 rare 단어의 비율: 43.56578001437815
# 전체 등장 빈도에서 rare 단어 등장 빈도율: 2.192078133478025

# 빈도수 1 이하인 단어 개수 제거하여 vocab_size 지정
# 하되 0번짜리 패딩 토큰과 OOV 토큰을 고려하여 +2 함
vocab_size = total_cnt - rare_cnt + 2
# print('빈도수1인 단어를 삭제한 전체 단어 집합의 개수: ', vocab_size)
# 빈도수1인 단어를 삭제한 전체 단어 +2를 한 집합의 개수:  787

# ------------------------------------------------------------
# 정수 인코딩
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()

# 인코딩 적용
tokenizer = Tokenizer(vocab_size, oov_token='OOV')  # OOV : Out-Of-Vocabulary 
tokenizer.fit_on_texts(tag_data)
tag_data = tokenizer.texts_to_sequences(tag_data)

# 인코딩 확인을 위해 샘플 출력
# print(tag_data[:3])
# [[19, 5, 356, 19, 11, 53], [3, 13, 134, 53], [49, 12, 1, 5, 53]]

# ------------------------------------------------------------
# x, y 로 지정해서 진행
x = tag_data
y = all_data['label']
# 한 번 확인
# print(len(x))       # 5700
# print(len(y))       # 5700

# ------------------------------------------------------------
# 길이 정해서 함수 돌리기
max_len = 18

# 패딩 적용
from tensorflow.keras.preprocessing.sequence import pad_sequences
x = pad_sequences(x, maxlen = max_len, padding='pre') 
# 확인
# print(x[0])
# # [   0    0    0    0   19 1261   23  374    1    5  477  627   19 1261 7   11    7   59]

# ------------------------------------------------------------
# train, test 나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, \
                                    shuffle=True, random_state=1)
# 확인
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)
# (4560, 18)
# (4560,)
# (1140, 18)
# (1140,)

# ------------------------------------------------------------
# 모델 구성
from tensorflow.keras.layers import Embedding, Dense, LSTM, SimpleRNN, GRU, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 임베딩 벡터 차원 100으로 정하고 lstm 이용
model = Sequential()
model.add(Embedding(vocab_size, input_length=18, output_dim=100))
model.add(LSTM(128))
model.add(Dense(64))
model.add(Dense(1, activation='sigmoid'))
# model.summary()

# callbacks 정의
stop = EarlyStopping(monitor='val_loss', mode = 'min', verbose=1, patience=12)
file_path = '../NLP/modelcheckpoint/project_011.h5'
mc = ModelCheckpoint(filepath= file_path, monitor='val_acc', mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode = 'min' ,patience=6, verbose=1, factor=0.8)

# 컴파일, 훈련
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_split=0.2, callbacks=[stop, mc, reduce_lr])

# ------------------------------------------------------------
# 정확도가 가장 높은 가중치 가져와서 적용
loaded_model = load_model('../NLP/modelcheckpoint/project_011.h5')
print('===== save complete =====')
print('loss: %.4f' % (loaded_model.evaluate(x_test, y_test)[0]), '\nacc: %.4f' % (loaded_model.evaluate(x_test, y_test)[1]))
# loss: 0.0243
# acc: 0.9939

# ------------------------------------------------------------
# 전에 predict에 넣을 것도 전처리 똑같이 해줘야 겠지?
import re

def do_predict(new_sentence):
    # 전처리
    new_sentence = re.compile("[^ㄱ-ㅎㅏ-ㅣ가-힣]").sub(' ',new_sentence)
    new_sentence = komo.morphs(new_sentence)  # 토큰화
    new_sentence = [word for word in new_sentence if not word in stopword] # 불용어 제거
    encoded = tokenizer.texts_to_sequences([new_sentence])  # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen = max_len, padding='pre')  # 패딩

    # 예측
    score = float(loaded_model.predict(pad_new))
    if(score > 0.5):
        print(new_sentence, '는 {:.2f} % 확률로 존댓말입니다.'.format(score * 100))
    else:
        print(new_sentence, '는 {:.2f} % 확률로 반말입니다.'.format((1 - score) * 100))


# # =========================================================
do_predict('이렇게 말해')
do_predict('이렇게 말하곤 해')
do_predict('이렇게 말해볼까')
do_predict('이렇게 말하니까')
do_predict('이렇게 말하는거 어떠냐')
do_predict('이렇게 말하는구나')
do_predict('이렇게 말하지마')
do_predict('이렇게 말하잖아')
do_predict('이렇게 말하더라')
do_predict('이렇게 말하잖니')
do_predict('이렇게 말하네')
do_predict('이렇게 말했어')
do_predict('이렇게 말하자')
do_predict('이렇게 말하면 안돼')
print('-------------------------------')
do_predict('이렇게 말해요')
do_predict('이렇게 말했어요')
do_predict('이렇게 말했다니까요')
do_predict('이렇게 말했나요?')
do_predict('이렇게 말하지요')
do_predict('이렇게 말했잖아요')
do_predict('이렇게 말해보세요')
do_predict('이렇게 말한 거에요')
do_predict('이렇게 말하지 마시오')
do_predict('이렇게 말하죠')
do_predict('이렇게 말했죠')
do_predict('이렇게 말했습니다')
do_predict('이렇게 말합시다')


# 튜닝 기록 https://docs.google.com/spreadsheets/d/17OxKDrjJH6KJs_J-Ng0AsvaZXp6svhGdKIjEbLP0Xhw/edit#gid=62567724
# 이 파일로 마무리!