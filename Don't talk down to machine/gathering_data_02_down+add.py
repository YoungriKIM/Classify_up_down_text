# 데이터셋 부족하다고 판단, 더 모아서 csv로 저장 함
# 저장 완료
# csv_down_data = pd.read_csv('../NLP/save/down_data_02.csv', index_col=0)


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
for i in range(1, 65):
    file_path = "../NLP/sample_data/down_4_sum/down_data ("+str(i)+").xls"
    temp = pd.read_excel(file_path)
    temp = pre_data(temp)
    df_test.append(temp)
down_data = pd.concat(df_test)

# 열 이름 변경
down_data.columns = ['label', 'data']

# 중복값 몇개인지 확인
print(len(down_data['label'].unique()))     # 1 > 라벨이 0으로 하나이니 ok
print('data열의 중복값: ', int(len(down_data['data']) - int(len(down_data['data'].unique())))) 
# data열의 중복값: 232

# 중복값 제거
down_data.drop_duplicates(subset = ['data'], inplace = True)
print('중복값 제거 후의 데이터 수: ', len(down_data))
# 중복값 제거 후의 데이터 수:  2870

# 한글 아닌 문자 제외
down_data['data'] = down_data['data'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
print('down_data.shape: ', down_data.shape)    # (2870, 2)
print('down_data.tail:\n ', down_data.tail())


# ------------------------------------------------------------
# ------------------------------------------------------------

# 다음 파일에서 up + down 하기 위해 저장해보자
print(type(down_data))  # <class 'pandas.core.frame.DataFrame'>

# ------ csv -------
# 저장하기
down_data.to_csv('../NLP/save/down_data_02.csv')
# 읽어와서 확인
csv_down_data = pd.read_csv('../NLP/save/down_data_02.csv', index_col=0)
print('load한 반말 데이터: \n', csv_down_data[-5:])

### 용량: 110KB

