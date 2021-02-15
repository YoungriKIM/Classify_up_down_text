# 주어진 코퍼스(corpus)에서 토큰(token)이라 불리는 단위로 나누는 작업을 토큰화(tokenization)라고 부릅니다.
# 토큰의 단위가 상황에 따라 다르지만, 보통 의미있는 단위로 토큰을 정의합니다.


#1. 단어 토큰화: 토큰의 기준이 단어일 경우 (각 기능에 대한 것은 사용할 떄 찾아볼 것)
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
