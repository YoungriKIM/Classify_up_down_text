def preprocess_data(data):
    temp = data.copy()
    return temp.iloc[-48:,[1,2,3,4,5,6,7,8]]

df_test = []

for i in range(81):
    file_path = '../data/csv/dacon1/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp)
    df_test.append(temp)

all_test = pd.concat(df_test)
print(all_test.shape)   #(3888, 8)


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