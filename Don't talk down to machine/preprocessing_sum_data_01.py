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