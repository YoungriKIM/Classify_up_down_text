# 01 파일에서 down 전체 불러와서 적용

import numpy as np
import pandas as pd
import re


# 전처리에 쓸 pre_data 정의
def pre_data(data):
    temp = data.copy()
    temp = temp.drop(['번호','뒷문맥','출전'],1)    # 필요없는 거 떨구고
    temp = temp.fillna('.')     # nan 값 . 으로
    temp['검색어'] = temp['앞문맥'].str.cat(temp['검색어'].astype(str))     # 한 문장으로 합병
    temp['앞문맥'] = 0      # 라벨링 0으로(반말)
    return temp

# down 데이터 한번에 모으기
df_test = []
for i in range(1, 51):
    file_path = "../NLP/sample_data/down_2/down_data ("+str(i)+").xls"
    temp = pd.read_excel(file_path)
    temp = pre_data(temp)
    df_test.append(temp)
down_data = pd.concat(df_test)

# 열 이름 변경
down_data.columns = ['label', 'data']

# 한글 아닌 문자 제외
down_data['data'] = down_data['data'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
print('down_data.shape: ', down_data.shape)    # (2427, 2)
print('down_data.tail:\n ', down_data.tail())

# Null값 있는지 확인
# 중복값 있는지 확인(유일한 값 개수 찾기)
# 중복값 제거 후 샘플 수 확인