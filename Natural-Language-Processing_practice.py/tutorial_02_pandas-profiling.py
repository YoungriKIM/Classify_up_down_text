import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_profiling

# 실습파일: https://www.kaggle.com/uciml/sms-spam-collection-dataset
# 스팸 분류


data = pd.read_csv('../NLP/practice_data/spam.csv', encoding='latin1')

# encording = 'latin1' 의 의미 > 라틴어라서~
# 인터넷에서 csv 파일을 다운받아서, 무언가 글자가 깨어지면 인코딩 방식을 알아낸 후, 인코딩 옵션을 지정해서 불러오면 됩니다!
# 파이썬 관련 인코딩은 이 링크를 참고해 보세요!
# https://docs.python.org/3/library/codecs.html#standard-encodings

print(data.head())

#      v1                                                 v2 Unnamed: 2 Unnamed: 3 Unnamed: 4
# 0   ham  Go until jurong point, crazy.. Available only ...        NaN        NaN        NaN
# 1   ham                      Ok lar... Joking wif u oni...        NaN        NaN        NaN
# 2  spam  Free entry in 2 a wkly comp to win FA Cup fina...        NaN        NaN        NaN
# 3   ham  U dun say so early hor... U c already then say...        NaN        NaN        NaN
# 4   ham  Nah I don't think he goes to usf, he lives aro...        NaN        NaN        NaN
# ham: 정상메일 / spam: 스팸메일 
# v2: 메일의 내용
# 3~5열은 5개만 뽑아 봤는데 벌써 nan값이 존재


# pandas-profiling 이용하기
pr = data.profile_report()  # 프로파일링 결과 리포트를 pr에 저장
print(pr)   # > 바로보면 안보임; 

pr.to_file('../NLP/practice_data/pr_report.html')   # HTML로 저장
print('===== save complete =====')  # file:///C:/NLP/practice_data/pr_report.html#overview

# 데이터셋의 다양한 정보를 profiling 처럼 정리하여 overview를 보여줌
