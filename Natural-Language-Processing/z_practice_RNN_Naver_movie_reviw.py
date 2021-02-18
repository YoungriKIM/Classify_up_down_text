# https://wikidocs.net/44249
# 한글로 된 예제를 익힐 필요가 있어 진행함

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# urllib 이용해 링크 넣어서 데이터 다운로드
# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename = 'ratings_train.txt')
# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")
# 탐색기에 저장 된 것 확인!

# train, test 데이터 지정
train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')

# train에 존재하는 영화 리뷰 개수 확인
print('train 리뷰 개수: ', len(train_data))
# train 리뷰 개수:  150000
print(train_data.head())
#          id                                           document  label
# 0   9976970                                아 더빙.. 진짜 짜증나네요 목소리      0
# 1   3819312                  흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나      1
# 2  10265843                                  너무재밓었다그래서보는것을추천한다      0
# 3   9045019                      교도소 이야기구먼 ..솔직히 재미는 없다..평점 조정      0
# 4   6483659  사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서 늙어보이기만 했던 커스틴 ...      1

# id 는 사실상 필요없으니 드랍할 예정 / 1(긍정), 0(부정) 으로 라벨링 됨

# test 데이터 확인
print('test 리뷰 개수: ', len(test_data))
# test 리뷰 개수:  50000

# ------------------------------------------------------------
# train 데이터 전처리
 
# train 중복 데이터 확인
print(len(train_data['label'].unique()))        # 2 > 0,1 뿐이니까 ok
print(len(train_data['document'].unique()))     # 146183 > 150000-146183=3817

# document 열에서 중복 데이터 제거
train_data.drop_duplicates(subset=['document'], inplace=True) 

# 중복 제거된 document 열의 샘플 수 확인
print('중복 제거 후 샘플 수: ', len(train_data))
# 중복 제거 후 샘플 수:  146183     > 굿제거~

# 0,1 라벨링의 비율 확인
train_data['label'].value_counts().plot(kind='bar'); # plt.show()
# 비슷한 비율을 보임

# Null 값 있는지 확인
print(train_data.isnull().values.any())
# True >> Null 값이 있다는 뜻!

# Null 값 어느 열이 있는지 확인
print(train_data.isnull().sum())
# id          0
# document    1     > document 열에 1개 존재
# label       0
# dtype: int64

# Null 값을 가진 샘플 제거
train_data = train_data.dropna(how='any')
print(train_data.isnull().values.any()) #False > Null 값 없음!

# Null 제거된 document 열의 샘플 수 확인
print('Null 제거 후 샘플 수: ', len(train_data))
# 중복 제거 후 샘플 수:  146182     > 한개 제거 됨

# 한글과 공백을 제외하고 모두 제거하기
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
print(train_data[:5])
# 한글과 공백만 남은 것 확인

# 그런데 기존에 한글이 없는 리뷰였다면 빈값이 되었을텐데!
# train 에 빈 값을 가진 행이 있다면 Null 값으로 변경하자
train_data['document'].replace('' , np.nan, inplace=True)
print(train_data.isnull().sum())
# id            0
# document    391   > 391개의 빈값이 nan 으로 변경
# label         0

# nan 값 제거
train_data = train_data.dropna(how = 'any')
print('nan 제거 후 샘플 수: ', len(train_data))
# nan 제거 후 샘플 수:  145791

# ------------------------------------------------------------
# test 데이터 전처리

# 중복값 제거
test_data.drop_duplicates(subset = ['document'], inplace = True)

# 정규 표현식으로 한글만 남기기
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

# 공백 Null 값으로 변경
test_data['document'].replace('', np.nan, inplace=True)

# Null 값 제거
test_data = test_data.dropna(how='any')
print('전처리 후 test 샘플 수: ', len(test_data))
# 전처리 후 test 샘플 수:  48995

# ------------------------------------------------------------
# 토큰화 + 불용어 제거

# okt 불러오기
from konlpy.tag import Okt
okt = Okt()

# 불용어 리스트 정의
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으', '로', '자', '에', '와', '한', '하다']

# train >> 토큰화 + 불용어 제거 적용
x_train = []
for sentence in train_data['document']:
    temp_x = []
    temp_x = okt.morphs(sentence, stem=True) # 토큰화 / stem = 정규화하여 변환. 이런> 이렇다, 만드는 > 만들다.
    temp_x = [word for word in temp_x if not word in stopwords]
    x_train.append(temp_x)
print(x_train[:3])
# 잘 된 것 확인

# test >> 토큰화 + 불용어 제거 적용
x_test = []
for sentence in test_data['document']:
    temp_x = []
    temp_x = okt.morphs(sentence, stem=True)
    temp_x = [word for word in temp_x if not word in stopwords]
    x_test.append(temp_x)
print(x_test[:3])
# 확인

# 정수 인코딩(텍스트를 숫자로)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)
print('단어에 정수 부여 확인\n', tokenizer.word_index)


# 등장 빈도수 3회 미만 단어의 비중 확인
threshold = 3
total_cnt = len(tokenizer.word_index)
rare_cnt = 0
total_freq = 0
rare_freq = 0

# 단어-빈도수의 쌍을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어 등장 빈도수가 threshold(3) 미만이면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)
# 단어 집합(vocabulary)의 크기 : 43751
# 등장 빈도가 2번 이하인 희귀 단어의 수: 24337
# 단어 집합에서 희귀 단어의 비율: 55.62615711640877
# 전체 등장 빈도에서 희귀 단어 등장 빈도 비율: 1.8717147610743852

