# > 결과가 너무 안 좋음 맨 처음 파일 쓰기로

# p_01 에서 적용한 불용어 리스트에서 종미어구와 비슷한 키워드를 더 삭제함
# stopword_02.txt > stopword_03.txt 으로 변경하여 진행하고 성능 비교

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
f = open('../NLP/sample_data/stopword_03.txt','rt',encoding='utf-8')  # Open file with 'UTF-8' 인코딩
text = f.read()
stopword = text.split('\n') 

# 품사 태깅으로 토큰화 (품사로 나눈 토큰화)
# from konlpy.tag import Kkma
# tokenizer = Kkma()
from konlpy.tag import Okt
okt = Okt()

tag_data = []
for sentence in all_data['data']:
    temp_x = []
    temp_x = okt.morphs(sentence)
    # temp_x = [word for word in temp_x if not word in stopword]
    tag_data.append(temp_x)

# 확인용 출력
print('토큰화 된 샘플: ', tag_data[-5:])

### 불용어 제거 , 토큰화 전 ###
#      label               data
# 42      0  그럼 이 형한테 보여 달라 그래
# 45      0          그 사람들은 그래
# 47      0     블랙 좀 깔지 말라고 그래
# 48      0           아무도 안 그래
# 49      0   저기 저 모든 이큐가 다 그래

### 불용어 제거 , 토큰화 후 ###
# 토큰화 된 샘플: [['형', '한테', '보여', '달라', '그래'],\
                #['사람', '은', '그래'], \
                #['블랙', '깔지', '말', '라고', '그래'],\
                #['아무', '도', '안', '그래'],\
                #['모든', '큐', '다', '그래']]

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
# 결과 안 좋으면 불용어 제거할 때 명사도 제거해야 할 것 같은 느낌

# ------------------------------------------------------------
# x, y 로 지정해서 진행
x = tag_data
y = all_data['label']
# 한 번 확인
# print(len(x))     # 4080
# print(len(y))     # 4080

# ------------------------------------------------------------
# 패딩(서로 다른 길이의 샘플을 동일하게 맞추기)
print('문장의 최대 길이: ', max(len(l) for l in x))
print('문장의 평균 길이: ', sum(map(len, x))/len(x))
# 문장의 최대 길이:  29
# 문장의 평균 길이:  4.549019607843137

# 패딩 길이 몇으로 할지 그래프로 확인
import matplotlib.pyplot as plt
plt.hist([len(s) for s in x], bins = 50)
plt.xlabel('lenth of samples')
plt.ylabel('number of samples')
# plt.show()

# 그래프를 보니 패딩 길이를 15로 하면 대부분의 샘플 커버 할 수 있을 듯
# 확인 해볼 함수 정의
def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if (len(s) <= max_len):
            cnt = cnt +1
    print('전체 샘플 중 길이가 %s 이하인 샘플 비율: %s' % (max_len, (cnt/len(nested_list))*100))

# 길이 30으로 정해서 함수 돌리기
max_len = 15
below_threshold_len(max_len, x)
# 전체 샘플 중 길이가 15 이하인 샘플 비율: 98.99509803921569
# 문장의 최대 길이는 29이지만 15로 패딩을 해도 98% 커버 가능

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
file_path = '../NLP/modelcheckpoint/project_06-2.hdf5'
mc = ModelCheckpoint(filepath= file_path, monitor='val_acc', mode = 'max', save_best_only=True, verbose=1)

# 컴파일, 훈련
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.2, callbacks=[stop, mc])

# ------------------------------------------------------------
# 정확도가 가장 높은 가중치 가져와서 적용
loaded_model = load_model('../NLP/modelcheckpoint/project_06-2.hdf5')
print('===== save complete =====')
print('loss: %.4f' % (loaded_model.evaluate(x_test, y_test)[0]), '\nacc: %.4f' % (loaded_model.evaluate(x_test, y_test)[1]))

# =====================================
# p_01  > project_01.h5
# loss: 0.1747
# acc: 0.9265

# p_06  # 불용어 조금 변경 > project_06.hdf5
# loss: 0.1503
# acc: 0.9277   

# 06-2 불용어 제거 안 함
# loss: 0.2306
# acc: 0.9265       > 결과가 너무 안 좋음 맨 처음 파일 쓰기로