# 주어진 코퍼스(corpus)에서 토큰(token)이라 불리는 단위로 나누는 작업을 토큰화(tokenization)라고 부릅니다.
# 토큰의 단위가 상황에 따라 다르지만, 보통 의미있는 단위로 토큰을 정의합니다.

# ------------------------------------------------------------------------------------
# 1. 단어 토큰화 Word Tokenization : 토큰의 기준이 단어일 경우 (각 기능에 대한 것은 사용할 떄 찾아볼 것)
# 1) word_tokenize
from nltk.tokenize import word_tokenize
print(word_tokenize("Don't be a fooled by the dark sounding name, Ms. Hanna's Orphanage is as cheery as cheery goes for a party shop."))

# ['Do', "n't", 'be', 'a', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', \
# 'Ms.', 'Hanna', "'s", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', \
# 'for', 'a', 'party', 'shop', '.']

# 2) WordPunctTokenizer
from nltk.tokenize import WordPunctTokenizer
print(WordPunctTokenizer().tokenize("Don't be a fooled by the dark sounding name, Ms. Hanna's Orphanage is as cheery as cheery goes for a party shop."))
# ['Don', "'", 't', 'be', 'a', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ','\
# , 'Ms', '.', 'Hanna', "'", 's', 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', \
# 'goes', 'for', 'a', 'party', 'shop', '.']

# 3) text_to_word_sequence
from tensorflow.keras.preprocessing.text import text_to_word_sequence
print(text_to_word_sequence("Don't be a fooled by the dark sounding name, Ms. Hanna's Orphanage is as cheery as cheery goes for a party shop."))
# ["don't", 'be', 'a', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', 'ms', \
# "hanna's", 'orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', \
# 'a', 'party', 'shop']

# 4) TreebankWordTokenizer
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
text = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
print(tokenizer.tokenize(text))
# ['Starting', 'a', 'home-based', 'restaurant', 'may', 'be', 'an', 'ideal.'\
# , 'it', 'does', "n't", 'have', 'a', 'food', 'chain', 'or', 'restaurant', \
# 'of', 'their', 'own', '.']

# ------------------------------------------------------------------------------------
# 2. 문장 토큰화 sentence Tokenization : 토큰의 기준이 문장일 경우 (각 기능에 대한 것은 사용할 떄 찾아볼 것)
from nltk.tokenize import sent_tokenize
text="His barbar kept word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
print(sent_tokenize(text))
# ['His barbar kept word.',\
# 'But keeping such a huge secret to himself was driving him crazy.',\
# 'Finally, the barber went up a mountain and almost to the edge of a cliff.',\
# 'He dug a hole in the midst of some reeds.',\
# 'He looked about, to make sure no one was near.']

# ???? 문장의 단어 안에 마침표가 있을 때 ex: Ph.D
from nltk.tokenize import sent_tokenize
text = "I am actively looking for Ph.D. students. and you are a Ph.D student."
print(sent_tokenize(text))
# ['I am actively looking for Ph.D. students.', 'and you are a Ph.D student.']
# 단순 마침표를 구분하여 문장을 나누지 않는다.

# 한글 문장 토큰화
import kss
text = "오늘은 눈이 많이 와서 그런지 짬뽕이 생각나는 날입니다. 배가고픕니다. 점심시간은 왜 항상 멀게만 느껴질까요?"
print(kss.split_sentences(text))
# ['오늘은 눈이 많이 와서 그런지 짬뽕이 생각나는 날입니다.',\
# '배가고픕니다.',\
# '점심시간은 왜 항상 멀게만 느껴질까요?']

text2 = "절.대.로.안.돼"
print(kss.split_sentences(text2))
# ['절.대.로.안.돼']

# ------------------------------------------------------------------------------------
# 문장 토큰화에서의 예외 사항을 위한 이진 분류기(Binary Classifier)
# https://www.grammarly.com/blog/engineering/how-to-split-sentences

