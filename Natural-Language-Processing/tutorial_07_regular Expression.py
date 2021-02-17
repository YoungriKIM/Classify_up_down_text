# 정규표현식 정리 >  https://wikidocs.net/4308
# 정규표현식 맞는지 확인 > https://regexr.com/

# 정규 표현식 re(regular expression)

import re

text = '평화로운 수요일 오후 2시 17분~'
r = re.sub('[^가-힣]',' ',text)

print(r)



# ================================

import numpy as np
import pandas as pd

# 전처리에 쓸 pre_data 정의(1. 필요 없는 열 삭제/2. nan값 .으로 변환/3. 두 개의 열 한 문장으로 병합)
def pre_data(data):
    temp = data.copy()
    temp = temp.drop(['번호','뒷문맥','출전'],1)    # 필요없는 거 떨구고
    temp = temp.fillna('.')     # anan 값
    temp['검색어'] = temp['앞문맥'].str.cat(temp['검색어'].astype(str))
    temp['앞문맥'] = 0
    return temp

# up 데이터 한번에 모으기
df_test = []
for i in range(1, 51):
    file_path = "../NLP/sample_data/down_2/down_data ("+str(i)+").xls"
    temp = pd.read_excel(file_path)
    temp = pre_data(temp)
    df_test.append(temp)

down_data = pd.concat(df_test)
print(down_data.shape)    # (2427, 2)