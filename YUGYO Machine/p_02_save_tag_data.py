# predict 할 데이터 전처리 할 떄
# tokenizer.fit_on_texts(tag_data)
# 해야 해서 tag_data 저장해야 함

# tag_data만 저장하기 위한 파일

# ------------------------------------------------------------
import numpy as np
import pandas as pd

# up, down 불러와서 합치자
# up
up_data = pd.read_csv('../NLP/save/up_data_02.csv', index_col=0)
print('load한 존대말 데이터: \n', up_data[-5:])
# down
down_data = pd.read_csv('../NLP/save/down_data_02.csv', index_col=0)
print('load한 반말 데이터: \n', down_data[-5:])
# 각 데이터의 수 확인
print('존댓말 데이터의 수: ', len(up_data), '\n반말 데이터의 수: ', len(down_data))
# 존댓말 데이터의 수:  1750 > 2830
# 반말 데이터의 수:  2330 > 2870

# 두 데이터 합하기
all_data = pd.concat([up_data, down_data])
print('shape of all_data: ', all_data.shape)
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
print('토큰화 된 샘플: ', tag_data[-10:-5])

### 불용어 제거 , 토큰화 전 ###
#      label                        data
# 35      0                몇 비비가 있다 그랬나
# 36      0  내 친구는 뭐 컴활 이런 게 더 어렵다고 그랬나
# 37      0                  실추라고 뭐 그랬나
# 39      0                         그랬나
# 42      0          선훈이 오빠가 여동생 있다 그랬나

### 불용어 제거 , 토큰화 후 ###
# 토큰화 된 샘플:  [['가', '있', '다', '그렇', '었', '나'], \
                # ['내', '는', '뭐', '화', 'ㄹ', '더', '어렵', '다고', '그렇', '었', '나'], \
                # ['라고', '뭐', '그렇', '었', '나'], \
                # ['그렇', '었', '나'], \
                # ['서', 'ㄴ', '가', '있', '다', '그렇', '었', '나']]


# ---------------------------------------------------
# ---------------------------------------------------
#tag_data만 저장해봅시다.
print(type(tag_data))
# <class 'list'>

# pickle 로 저장
import pickle

# 저장
pickle.dump(tag_data, open('../NLP/save/project_011_tag_data.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)

# 불러오기
load_tag_data = pickle.load(open('../NLP/save/project_011_tag_data.pickle', 'rb'))
print('======read complete=====')

print(load_tag_data[-5:])

# 손실없이 저장 완료!