import numpy as np
import pandas as pd
import xlrd

# 불러오고
df = pd.read_excel("../NLP/sample_data/up/seyo_02.xls")
print(df.shape)

# 필요없는 열 삭제
df2 = df.drop(['번호','뒷문맥','출전'],1)
# print(df2.shape)
# print(df2.tail())

# nan값 .으로 변경
df2 = df2.fillna('.')
# print(df2.tail())

# 0열에 라벨, 1열에 문장
df2['검색어'] = df2['앞문맥'].str.cat(df2['검색어'].astype(str))
df2['앞문맥'] = 0
# print(df2.tail())

# 열이름 바꾸기
df2.columns = ['label', 'data']

# Null값 있는지 확인
print(df2.isnull().values.any())
# False : Null값 없음

# 중복값 있는지 확인(유일한 값 개수 찾기)
print(len(df2['label'].unique()))     # 1   > 0 하나뿐이니 ok
print(len(df2['data'].unique()))      # 44  > 50-44=6 중복값 6개

# 중복값 제거
df2.drop_duplicates(subset=['data'], inplace=True)
print('중복값 제거 후 총 샘플 수: ', len(df2))
print(df2.shape)
# 중복값 제거 후 총 샘플 수:  44