import pandas as pd
import numpy as np

# 시리즈
sr = pd.Series([17000, 18000, 1000, 5000], index=['피자','치킨','콜라','맥주'])
print(sr)
# 피자    17000
# 치킨    18000
# 콜라     1000
# 맥주     5000
# dtype: int64

print(sr.values)
# [17000 18000  1000  5000]

print(sr.index)
# Index(['피자', '치킨', '콜라', '맥주'], dtype='object')

# 데이터프레임
values = [[1,2,3],[4,5,6],[7,8,9]]
index = ['one','two','three']
columns = ['A','B','C']
df = pd.DataFrame(values, index=index, columns=columns)

print(df)
print(df.index)
print(df.columns)
print(df.values)

# 데이터프레임의 생성
data = [
    ['1000', 'Steve', 90.72]
    ,['1001', 'James', 78.09]
    ,['1002', 'Youngri', 98.43]
    ,['1003', 'Jane', 64.16]
    ,['1004', 'Pilwoong', 81.30]
    ,['1005', 'Tony', 99.14]
]

# ...중략

# 넘파이 ndarray

a = np.zeros((2,3))
print(a)
a2 = np.ones((2,3))
print(a2)
a3 = np.full((2,7),8)
print(a3)
a4 = np.eye(3)
print(a4)
a5 = np.random.random((2,2))
print(a5)

a6 = np.arange(10)
print(a6)
a7 = np.arange(1, 10, 2)
print(a7)

a8 = np.array(np.arange(30)).reshape(5,6)
print(a8)

# matplotlib
import matplotlib.pyplot as plt
plt.title('students')
plt.plot([1,2,3,4], [2,4,8,6])
plt.plot([1.5,2.5,3.5,4.5],[3,5,8,10])
plt.xlabel('hour')
plt.ylabel('score')
plt.legend(['A students', 'B students'])
plt.show()

