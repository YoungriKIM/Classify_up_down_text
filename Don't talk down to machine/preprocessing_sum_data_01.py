# 전처리 어느정도 한 up과 down 데이터를 합해서 나머지 전처리를 하자

# [진행 한 전처리]
# 1) 데이터셋에서 필요 없는 부분 버리기
# 2) nan 값은 . 으로 변환
# 3) 두 개의 열로 나눠진 문장 하나로 합치기
# 4) 0과 1로 라벨링 부여하기
# 5) 열 이름 label, data로 변경
# 6) 정규식으로 한글 데이터만 남기기

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

# tokenizer (품사로 나눈 토큰화)
# from konlpy.tag import Kkma
# tokenizer = Kkma()
from konlpy.tag import Okt
okt = Okt()

token_data = []
for sentence in all_data['data']:
    temp_x = []
    temp_x = okt.morphs(sentence)
    temp_x = [word for word in temp_x if not word in stopword]
    token_data.append(temp_x)

# 확인용 출력
print('토큰화 된 샘플: ', token_data[-5:])

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
# 정수 인코딩(텍스트를 숫자로)
from tensorflow.keras.preprocessing.text import Tokenizer


### 인코딩하기 전에!!!!!! 길이를 몇으로 할지 정해야 하니까 네이버 149줄 참고























