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

# 중복값 몇개인지 확인
print(len(up_data['label'].unique()))     # 1 > 라벨이 0으로 하나이니 ok
print('data열의 중복값: ', int(len(up_data['data']) - int(len(up_data['data'].unique()))))      # 2330 > 
# data열의 중복값:  93

# 중복값 제거
up_data.drop_duplicates(subset = ['data'], inplace = True)
print('중복값 제거 후의 데이터 수: ', len(up_data))
# 중복값 제거 후의 데이터 수:  1750


# 한글 아닌 문자 제외
up_data['data'] = up_data['data'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
print('down_data.shape: ', up_data.shape)    # (1843, 2)
print('down_data.tail:\n ', up_data.tail())

# ------------------------------------------------------------
# ------------------------------------------------------------

# 다음 파일에서 up + down 하기 위해 저장해보자

# ------ csv -------
# 저장하기
up_data.to_csv('../NLP/save/up_data_01.csv')
# 읽어와서 확인
csv_up_data = pd.read_csv('../NLP/save/up_data_01.csv', index_col=0)
print('load한 존대말 데이터: \n', csv_up_data[-5:])

