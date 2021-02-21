# predict 할 데이터 전처리 할 떄
# tokenizer.fit_on_texts(tag_data)
# 해야 해서 tag_data 저장해야 함

# tag_data만 저장하기 위한 파일

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
# 본격적인 전처리 시작

# 불용어 불러오기(import_stopword.py 파일 참고)
f = open('../NLP/sample_data/stopword_02.txt','rt',encoding='utf-8')  # Open file with 'UTF-8' 인코딩
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
    temp_x = [word for word in temp_x if not word in stopword]
    tag_data.append(temp_x)

# 확인용 출력
print('토큰화 된 샘플: ', tag_data[-5:])
# 토큰화 된 샘플:  [['형', '한테', '보여', '달라', '그래'],
                # ['사람', '은', '그래'], 
                # ['블랙', '깔지', '말', '라고', '그래'], 
                # ['아무', '도', '안', '그래'], 
                # ['모든', '큐', '다', '그래']]


# ---------------------------------------------------
# ---------------------------------------------------
#tag_data만 저장해봅시다.
print(type(tag_data))
# <class 'list'>

# pickle 로 저장
import pickle

# 저장
pickle.dump(tag_data, open('../NLP/save/project_01_tag_data.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)

# 불러오기
load_tag_data = pickle.load(open('../NLP/save/project_01_tag_data.pickle', 'rb'))
print('======read complete=====')

print(load_tag_data[-5:])
# [['형', '한테', '보여', '달라', '그래'], 
# ['사람', '은', '그래'], 
# ['블랙', '깔지', '말', '라고', '그래'], 
# ['아무', '도', '안', '그래'], 
# ['모든', '큐', '다', '그래']]

# 손실없이 저장 완료!