# 01 파일에서 up 전체 불러와서 적용

import numpy as np
import pandas as pd
import re


# 전처리에 쓸 pre_data 정의
def pre_data(data):
    temp = data.copy()
    temp = temp.drop(['번호','뒷문맥','출전'],1)    # 필요없는거 버리고
    temp = temp.fillna('.')     # nan값 .으로
    temp['검색어'] = temp['앞문맥'].str.cat(temp['검색어'].astype(str))     # 한 문장으로 합병
    temp['앞문맥'] = 1      # 라벨링 1으로(존댓말)
    return temp

# up 데이터 한번에 모으기
df_test = []
for i in range(1, 53):
    file_path = "../NLP/sample_data/up_2/up_data ("+str(i)+").xls"
    temp = pd.read_excel(file_path)
    temp = pre_data(temp)
    df_test.append(temp)
up_data = pd.concat(df_test)

# 열 이름 변경
up_data.columns = ['label', 'data']

# 한글 아닌 문자 제외
up_data['data'] = up_data['data'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
print('down_data.shape: ', up_data.shape)    # (1843, 2)
print('down_data.tail:\n ', up_data.tail())



