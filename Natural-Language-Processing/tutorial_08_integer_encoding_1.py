# 정수 인코딩 integer encording
# 컴퓨터는 텍스트보다 숫자로 바꾸어야 더 잘 처리 할 수 있다

# 정수 인코딩
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."

# 문장 토큰화
text = sent_tokenize(text)
print(text)
# ['A barber is a person.', \
# 'a barber is good person.', \
# 'a barber is huge person.', \
# 'he Knew A Secret!', \
# 'The Secret He Kept is huge secret.', \
# 'Huge secret.', \
# 'His barber kept his word.', \
# 'a barber kept his word.',\
#  'His barber kept his secret.', \
# 'But keeping and keeping such a huge secret to himself was driving the barber crazy.', \
# 'the barber went up a huge mountain.']

# 정제와 단어 토큰화
vocab = {}
sentences = []
stop_words = set(stopwords.words('english'))

for i in text:
    sentence = word_tokenize(i) # 문장토큰화 한 걸로 단어토큰화 진행
    result = []     # 결과 리스트로 반환

    for word in sentence:
        word = word.lower() # 소문자화 하여 종류를 줄임
        if word not in stop_words:      # 불용어 리스트에 없는 것들만
            if len(word) > 2 :          # 그 중에서 길이가 3이상인 것들만
                result.append(word)     # result에 쌓을거야
                if word not in vocab:
                    vocab[word] = 0
                vocab[word] += 1        # vocab에 중복을 제거한 단어와 빈도수가 기록
    sentences.append(result)
print(sentences)

# [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], \
# ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], \
# ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'],\
#  ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'],\
#  ['barber', 'went', 'huge', 'mountain']]

# vocab 출력하여 중복된 빈도수 확인
print(vocab)
# {'barber': 8, 'person': 3, 'good': 1, 'huge': 5, 'knew': 1, 'secret': 6, 'kept': 4, \
# 'word': 2, 'keeping': 2, 'driving': 1, 'crazy': 1, 'went': 1, 'mountain': 1}

# 단어를 선책하여 빈도수 확인
print(vocab['barber'])
# 8

# 빈도수가 높은 순서대로 정렬
vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True)
print(vocab_sorted)
# [('barber', 8), ('secret', 6), ('huge', 5), ('kept', 4), ('person', 3), ('word', 2),\
#  ('keeping', 2), ('good', 1), ('knew', 1), ('driving', 1), ('crazy', 1), ('went', 1),\
#  ('mountain', 1)]

# 빈도수가 낮은 단어 제외하고 높을 수록 낮은 인덱스 부여
word_to_index = {}
i = 0
for (word, frequency) in vocab_sorted:
    if frequency > 1 :  # 빈도수가 적은 단어를 제외
        i = i+1
        word_to_index[word] = i
print(word_to_index)
# {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'word': 6, 'keeping': 7}

# 이제 word_to_index 로 sentences(46번 라인)의 각 단어를 정수로 바꿀 차례
# 그런데 ['barber', 'person']은 [1, 5]로 바꾸면 되지만
# ['barber', 'good', 'person'] 안의 good은 더 이상 없는 단어이기 때문에 
# Out-Of-Vocabulary(단어 집합에 없는 단어) > OOV 라는 단어를 추가하여
# 집합에 없는 단어들은 'OOV'의 인덱스로 인코딩 하겠음

# OOV 단어 추가하여 인덱싱
word_to_index['OOV'] = len(word_to_index) + 1   # 전부 8로 하겠다는 뜻

# word_to_index를 사용하여 sentences의 모든 단어를 맵핑되는 정수로 인코딩
encoded = []
for s in sentences:
    temp = []
    for w in s :
        try:
            temp.append(word_to_index[w])
        except KeyError:
            temp.append(word_to_index['OOV'])
    encoded.append(temp)
print(encoded)

# [[1, 5], [1, 8, 5], [1, 3, 5], [8, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6],
# [1, 4, 6], [1, 4, 2], [7, 7, 3, 2, 8, 1, 8], [1, 8, 3, 8]]

# ------------------------------------------------------------
# 파이썬 dictionary 자료형으로 정수 인코딩을 진행했는데
# 좀 더 쉽게 하기 위해 Counter, FreqDist, enumerate 또는 케라스 토크나이저를 사용해보자

# Counter 사용하기
from collections import Counter
print(sentences)  # 문장토큰화 후 단어토큰화 된 결과가 저장된 sentences
# [['barber', 'person'], ['barber', 'good', 'person'], ... ['barber', 'went', 'huge', 'mountain']]

# 단어을 한 집합으로 만들기 위하여 sentences에서 문장의 경계의 [,] 를 제거하고 하나의 리스트로 만들자
words = sum(sentences, [])
print(words)
# ['barber', 'person', 'barber', 'good',... 'barber', 'went', 'huge', 'mountain']

# 파이썬의 Counter()로 중복을 제거하고 단어의 빈도수를 기록
vocab = Counter(words)
print(vocab)
# Counter({'barber': 8, 'secret': 6, 'huge': 5, 'kept': 4, 'person': 3, 'word': 2, \
# 'keeping': 2, 'good': 1, 'knew': 1, 'driving': 1, 'crazy': 1, 'went': 1, 'mountain': 1})

# 특정단어 빈도수 확인
print(vocab['secret'])
# 6

# most_common()으로 상위 빈도수를 가진 단어 5개 리턴하기
vocab_size = 5
vocab = vocab.most_common(vocab_size)
print(vocab)
# [('barber', 8), ('secret', 6), ('huge', 5), ('kept', 4), ('person', 3)]

# 높은 빈도수를 가진 단어에 낮은 인덱스 부여
word_to_index = {}
i = 0
for (word, frequency) in vocab:
    i = i+1
    word_to_index[word] = i
print(word_to_index)
# {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}

# ------------------------------------------------------------
# NLTK의 FreqDist 사용하기
from nltk import FreqDist
import numpy as np

# np.hstack 으로 sentences내 단어를 하나의 리스트로 합
print(sentences)
# [['barber', 'person'], ... ['barber', 'went', 'huge', 'mountain']]
print(np.hstack(sentences))
# ['barber', 'person', 'barber', 'good',... 'barber', 'went', 'huge', 'mountain']

# FreqDist 이용하여 단어를 키(key), 빈도수가 값(values)로 변환
vocab = FreqDist(np.hstack(sentences))

# 특정단어 빈도수 검색 
print(vocab['barber'])
# 8

# most_common()으로 상위 빈도수를 가진 단어 5개 리턴하기
vocab_size = 5
vocab = vocab.most_common(vocab_size)
print(vocab)
# [('barber', 8), ('secret', 6), ('huge', 5), ('kept', 4), ('person', 3)]

# enumerate()를 사용해 높은 빈도수 단어에 낮은 정수 인덱스 부여
word_to_index = {word[0] : index + 1 for index, word in enumerate(vocab)}
print(word_to_index)
# {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
# 훨씬 빠르게 인덱싱 완료!



# 다음 파일에 이어서 작성