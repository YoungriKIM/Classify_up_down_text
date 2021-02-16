# 정수 인코딩 integer encording
# 컴퓨터는 텍스트보다 숫자로 바꾸어야 더 잘 처리 할 수 있다
# 이전 파일에 이어서 작성

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

# ------------------------------------------------------------
# keras의 텍스트 전처리를 이용해보자
# https://wikidocs.net/31766