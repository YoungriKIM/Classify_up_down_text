
# 불용어 제거
# 큰 의미가 없는 단어 토큰을 제거

# ------------------------------------------------------------------------------------
# nltk를 통한 불용어 제거
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example = "Family is not an important thing. It's everything."
stop_word = set(stopwords.words('english'))  # 영어에서 정의된 불용어 들을 가져오겠음

word_tokens = word_tokenize(example)

result = []
for w in word_tokens:
    if w not in stop_word:
        result.append(w)

print(word_tokens)  # 토큰화까지만 한 것
# ['Family', 'is', 'not', 'an', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
print(result)       # 불용어가 아닌 단어만 합친 것
# ['Family', 'important', 'thing', '.', 'It', "'s", 'everything', '.']

# ------------------------------------------------------------------------------------
# 한국어에서 불용어 제거 (직접 불용어를 정의하여 사용하자)

from nltk.tokenize import word_tokenize

example = "고기를 아무렇게나 구우려고 하면 안 돼. 아 그게 고기라고 다 같은 게 아니거든. 예컨대 삼겹살을 구울 때는 중요한 게 있지."
stop_words = "아무렇게나 아 그게 예컨대 구울 이제 가끔 어 오"  # 기준없이 랜덤하게 정함

stop_word = stop_words.split(' ')
word_tokens = word_tokenize(example)

result = []
for w in word_tokens:
    if w not in stop_words:
        result.append(w)
# 위의 4줄을 아래의 한 줄로 대체 가능
# result=[word for word in word_tokens if not word in stop_words]

print(word_tokens)
# ['고기를', '아무렇게나', '구우려고', '하면', '안', '돼', '.',\
#  '아', '그게', '고기라고', '다', '같은', '게', '아니거든', '.', \
# '예컨대', '삼겹살을', '구울', '때는', '중요한', '게', '있지', '.']
print(result)
# ['고기를', '구우려고', '하면', '안', '돼', '.', '고기라고', '다',\
#  '같은', '아니거든', '.', '삼겹살을', '때는', '중요한', '있지', '.']



# ------------------------------------------------------------------------------------
# 한국어에서 자주 쓰이는 불용어 리스트
# https://www.ranks.nl/stopwords/korean

# 직접 정의하는 것 보다. txt, csv로 정리해놓고 불러와서 쓰는 것이 효율적