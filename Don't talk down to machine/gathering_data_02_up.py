# 01 파일에서 up 전체 불러와서 적용

import numpy as np
import pandas as pd

# 전처리에 쓸 pre_data 정의(1. 필요 없는 열 삭제/2. nan값 .으로 변환/3. 두 개의 열 한 문장으로 병합)
def pre_data(data):
    temp = data.copy()
    temp = temp.drop(['번호','뒷문맥','출전'],1)
    temp = temp.fillna('.')
    temp['검색어'] = temp['앞문맥'].str.cat(temp['검색어'].astype(str))
    temp['앞문맥'] = 1
    return temp

# up 데이터 한번에 모으기
df_test = []
for i in range(1, 53):
    file_path = "../NLP/sample_data/up_2/up_data ("+str(i)+").xls"
    temp = pd.read_excel(file_path)
    temp = pre_data(temp)
    df_test.append(temp)

up_data = pd.concat(df_test)
print(up_data.shape)    #(1843, 2)



