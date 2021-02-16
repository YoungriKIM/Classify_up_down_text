# ------------------------------------------------------------------------------------
# 7. 품사 태깅(Part-of-sppech tagging)
# 단어의 의미를 제대로 파악하기 위해 해당 단어가 어떤 품사로 쓰였는지 알아야 한다.
# NLTK 와 KoNLPy 의 품사 태깅을 해보자

# KoNLPy의 형태소 분석기:
# Okt(Open Korea Text), 메캅(Mecab), 코모란(Komoran), 한나눔(Hannanum), 꼬꼬마(Kkma)
# 코모란이 빠르다고 하니 실전 들어가면 알아볼 것

text = "열심히 코딩한 나 자신, 점심에는 마라샹궈를 먹어봐요."

# Okt
# 1) morphs : 형태소 추출
# 2) pos : 품사 태깅(Part-of-speech tagging)
# 3) nouns : 명사 추출

# 형태소 추출
from konlpy.tag import Okt
okt = Okt()
print(okt.morphs(text))     # morphs: [언어]형태소
# ['열심히', '코딩', '한', '나', '자신', ',', \
# '점심', '에는', '마라샹궈', '를', '먹어', '봐요', '.']

# 품사 태깅
print(okt.pos(text))
# [('열심히', 'Adverb'), ('코딩', 'Noun'), ('한', 'Josa'),\
#  ('나', 'Noun'), ('자신', 'Noun'), (',', 'Punctuation'),\
#  ('점심', 'Noun'), ('에는', 'Josa'), ('마라샹궈', 'Noun'), ('를', 'Josa'), \
# ('먹어', 'Verb'), ('봐요', 'Verb'), ('.', 'Punctuation')]  

# 명사 추출
print(okt.nouns(text))
# ['코딩', '나', '자신', '점심', '마라샹궈']

# Kkma
# 1) morphs : 형태소 추출
# 2) pos : 품사 태깅(Part-of-speech tagging)
# 3) nouns : 명사 추출

# 형태소 추출
from konlpy.tag import Kkma
kkma = Kkma()
print(kkma.morphs(text))
# ['열심히', '코딩', '하', 'ㄴ', '나', '자신', ',', \
# '점심', '에', '는', '마라', '샹궈', '를', '먹', '어', '보', '아요', '.']

# 품사 태깅
print(kkma.pos(text))
# [('열심히', 'MAG'), ('코딩', 'NNG'), ('하', 'XSV'), ('ㄴ', 'ETD'), \
# ('나', 'NP'), ('자신', 'NNG'), (',', 'SP'), ('점심', 'NNG'), ('에', 'JKM'), ('는', 'JX'), \
# ('마라', 'NNG'), ('샹궈', 'UN'), ('를', 'JKO'), \
# ('먹', 'VV'), ('어', 'ECD'), ('보', 'VXV'), ('아요', 'EFN'), ('.', 'SF')]

print(kkma.nouns(text))
# ['코딩', '나', '자신', '점심', '마라', '마라샹궈', '샹궈']





# 한국어 형태소 분석기 성능 비교 : https://iostream.tistory.com/144
# http://www.engear.net/wp/%ED%95%9C%EA%B8%80-%ED%98%95%ED%83%9C%EC%86%8C-%EB%B6%84%EC%84%9D%EA%B8%B0-%EB%B9%84%EA%B5%90/

# 윈도우10 메캅 설치 :
# https://cleancode-ws.tistory.com/97