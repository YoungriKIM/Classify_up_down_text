# p_08 기준으로 빈도수 작은 것 삭제해서 비교


# ------------------------------------------------------------
import numpy as np
import pandas as pd

# up, down 불러와서 합치자
# up
up_data = pd.read_csv('../NLP/save/up_data_02.csv', index_col=0)
print('load한 존대말 데이터: \n', up_data[-5:])
# down
down_data = pd.read_csv('../NLP/save/down_data_02.csv', index_col=0)
print('load한 반말 데이터: \n', down_data[-10:-5])
# 각 데이터의 수 확인
print('존댓말 데이터의 수: ', len(up_data), '\n반말 데이터의 수: ', len(down_data))
# 존댓말 데이터의 수:  1750 > 2830
# 반말 데이터의 수:  2330 > 2870

# 두 데이터 합하기
all_data = pd.concat([up_data, down_data])
print('shape of all_data: ', all_data.shape)
# shape of all_data:  (5700, 2)

# ------------------------------------------------------------
# 전처리 전에 pandas_profiling 으로 데이터 분석
# import pandas_profiling

# profile_all_data = all_data.profile_report()
# profile_all_data.to_file('../NLP/sample_data/profile_all_data_p_08.html')
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
    temp_x = komo.morphs(sentence)
    temp_x = [word for word in temp_x if not word in stopword]
    tag_data.append(temp_x)

# 확인용 출력
# print('토큰화 된 샘플: ', tag_data[-10:-5])

### 불용어 제거 , 토큰화 전 ###
#      label                        data
# 35      0                몇 비비가 있다 그랬나
# 36      0  내 친구는 뭐 컴활 이런 게 더 어렵다고 그랬나
# 37      0                  실추라고 뭐 그랬나
# 39      0                         그랬나
# 42      0          선훈이 오빠가 여동생 있다 그랬나

### 불용어 제거 , 토큰화 후 ###
# 토큰화 된 샘플:  [['비', '비', '가', '있다', '그리하', '였', '나'], 
                # ['내', '친구', '는', '뭐', '컴', '활', '이렇', 'ㄴ', '더', '어렵', '다고', '그리하', '였', '나'], 
                # ['실추', '라고', '뭐', '그리하', '였', '나'], 
                # ['그리하', '였', '나'], 
                # ['선훈', '오빠', '가', '여동생', '있다', '그리하', '였', '나']]

# ------------------------------------------------------------
# ------------------------------------------------------------
# p_09 에서 추가한 부분

# 등장 빈도수 확인을 위해 그래프 그리기

# # 정수 인코딩해서 넘기기
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
lemon = tokenizer.fit_on_texts(tag_data)
apple = pd.DataFrame(tokenizer.word_index.values())
# print('단어에 부여된 정수 확인\n: ', tokenizer.word_index)
# 어에 부여된 정수 확인
# :  {'어': 1, '아': 2, '가': 3, '는': 4, '았': 5, '게': 6, '었': 7, '되': 8, '어요': 9, '있': 10, '시': 11,

import pandas_profiling

aaa = apple.profile_report()
aaa.to_file('../NLP/sample_data/profile_all_data_p_shit.html')
print('===== save complete =====')


















'''
# 0번짜리 패딩과 OOV 토큰 고려하여 단어 집합의 크기에서 +2
# 단어 집합(vocabulary)의 크기 : 5702
vocab_size = 5702

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
# [[1331, 7, 20, 389, 2054, 5, 505, 661, 1331, 7, 8, 49], [4, 4, 1332, 110, 506, 332, 202, 110, 49], [113, 2055, 5, 20, 49]]

# ------------------------------------------------------------
# x, y 로 지정해서 진행
x = tag_data
y = all_data['label']
# 한 번 확인
print(len(x))       # 5700
print(len(y))       # 5700

# ------------------------------------------------------------
# 패딩(서로 다른 길이의 샘플을 동일하게 맞추기)
print('문장의 최대 길이: ', max(len(l) for l in x))
print('문장의 평균 길이: ', sum(map(len, x))/len(x))
# 문장의 최대 길이:  65
# 문장의 평균 길이:  6.606140350877193

# 패딩 길이 몇으로 할지 그래프로 확인
import matplotlib.pyplot as plt
plt.hist([len(s) for s in x], bins = 50)
plt.xlabel('lenth of samples')
plt.ylabel('number of samples')
# plt.show()

# 그래프를 보니 패딩 길이를 20로 하면 대부분의 샘플 커버 할 수 있을 듯
# 확인 해볼 함수 정의
def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if (len(s) <= max_len):
            cnt = cnt +1
    print('전체 샘플 중 길이가 %s 이하인 샘플 비율: %s' % (max_len, (cnt/len(nested_list))*100))

# 길이 정해서 함수 돌리기
max_len = 20
below_threshold_len(max_len, x)
# 전체 샘플 중 길이가 20 이하인 샘플 비율: 98.57894736842105

# 패딩 적용
from tensorflow.keras.preprocessing.sequence import pad_sequences
x = pad_sequences(x, maxlen = max_len, padding='pre') 
# 확인
print(x[0])
# [  0  0  0  0  0  0  0  0 1331  7  20  389 2054  5  505  661 1331    7    8   49]

# ------------------------------------------------------------
# train, test 나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1)
# 확인
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
# (4560, 20)
# (4560,)
# (1140, 20)
# (1140,)

# ------------------------------------------------------------
# 모델 구성
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 임베딩 벡터 차원 100으로 정하고 lstm 이용
model = Sequential()
model.add(Embedding(vocab_size, 100))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

# callbacks 정의
stop = EarlyStopping(monitor='val_loss', mode = 'min', verbose=1, patience=8)
file_path = '../NLP/modelcheckpoint/project_09.h5'
mc = ModelCheckpoint(filepath= file_path, monitor='val_acc', mode = 'max', save_best_only=True, verbose=1)

# 컴파일, 훈련
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.2, callbacks=[stop, mc])

# ------------------------------------------------------------
# 정확도가 가장 높은 가중치 가져와서 적용
loaded_model = load_model('../NLP/modelcheckpoint/project_09.h5')
print('===== save complete =====')
print('loss: %.4f' % (loaded_model.evaluate(x_test, y_test)[0]), '\nacc: %.4f' % (loaded_model.evaluate(x_test, y_test)[1]))

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


# =========================================================
# 예측하기

# project_08-3  > kkma
# loss: 0.0330
# acc: 0.9886

# project_08-4 > komoran
# loss: 0.0243
# acc: 0.9939
# kkma 보다 komoran이 더 정확. # 기준파일로 자정

# =========================================================
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


'''