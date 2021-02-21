# 각 문장의 길이가 서로 다를 수 있는데, 병렬 연산을 위해 길이를 맞춰주어야 한다.
# padding을 해보자

# ------------------------------------------------------------------------------------
# Numpy로 패딩하기

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

# sentences 미리 제공
sentences = [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]

# 토큰화
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

# 정수 인코딩
encoded = tokenizer.texts_to_sequences(sentences)
print(encoded)
# [[1, 5], [1, 8, 5], [1, 3, 5], [9, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6], \
# [1, 4, 6], [1, 4, 2], [7, 7, 3, 2, 10, 1, 11], [1, 12, 3, 13]]

# 동일한 길이로 맞춰주기 전 가장 긴 문장 길이 계산
max_len = max(len(item) for item in encoded)
print(max_len)
# 7

for item in encoded:        # 각 문장에 대해서
    while len(item) < max_len:   # max_len 보다 작으면
        item.append(0)           # 0 을 붙인다

padded_np = np.array(encoded)
print(padded_np)

# [[ 1  5  0  0  0  0  0]
#  [ 1  8  5  0  0  0  0]
#  [ 1  3  5  0  0  0  0]
#  [ 9  2  0  0  0  0  0]
#  [ 2  4  3  2  0  0  0]
#  [ 3  2  0  0  0  0  0]
#  [ 1  4  6  0  0  0  0]
#  [ 1  4  6  0  0  0  0]
#  [ 1  4  2  0  0  0  0]
#  [ 7  7  3  2 10  1 11]
#  [ 1 12  3 13  0  0  0]]

# 숫자 0 을 사용하고 있다면 zero padding 이라고 한다.

# ------------------------------------------------------------------------------------
# 케라스 전처리 도구로 패딩하기

from tensorflow.keras.preprocessing.sequence import pad_sequences

# 토큰화
encoded = tokenizer.texts_to_sequences(sentences)
# print(encoded)

# 패딩
padded = pad_sequences(encoded)
print(padded)
# [[ 0  0  0  0  0  1  5]
#  [ 0  0  0  0  1  8  5]
#  [ 0  0  0  0  1  3  5]
#  [ 0  0  0  0  0  9  2]
#  [ 0  0  0  2  4  3  2]
#  [ 0  0  0  0  0  3  2]
#  [ 0  0  0  0  1  4  6]
#  [ 0  0  0  0  1  4  6]
#  [ 0  0  0  0  1  4  2]
#  [ 7  7  3  2 10  1 11]
#  [ 0  0  0  1 12  3 13]]      # 결과가 다른 이유는 pad_sequence의 경우 문서 앞에 0으로 채워서

# 뒤에 0으로 붙이고 싶다면 인자로 padding = 'post' 추가
padded2 = pad_sequences(encoded, padding = 'post')
print(padded2)
# [[ 1  5  0  0  0  0  0]
#  [ 1  8  5  0  0  0  0]
#  [ 1  3  5  0  0  0  0]
#  [ 9  2  0  0  0  0  0]
#  [ 2  4  3  2  0  0  0]
#  [ 3  2  0  0  0  0  0]
#  [ 1  4  6  0  0  0  0]
#  [ 1  4  6  0  0  0  0]
#  [ 1  4  2  0  0  0  0]
#  [ 7  7  3  2 10  1 11]
#  [ 1 12  3 13  0  0  0]]

# 최대 길이보다 짧게 패딩하기
padded3 = pad_sequences(encoded, padding = 'post', maxlen = 5)
print(padded3)
# [[ 1  5  0  0  0]
#  [ 1  8  5  0  0]
#  [ 1  3  5  0  0]
#  [ 9  2  0  0  0]
#  [ 2  4  3  2  0]
#  [ 3  2  0  0  0]
#  [ 1  4  6  0  0]
#  [ 1  4  6  0  0]
#  [ 1  4  2  0  0]
#  [ 3  2 10  1 11]
#  [ 1 12  3 13  0]]        # 이 경우 길이가 5보다 긴 데이터는 손실된다.