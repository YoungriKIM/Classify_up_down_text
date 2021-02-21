import numpy as np
from konlpy.tag import Kkma
kkma = Kkma()

f = open('../NLP/sample_data/stopword_02.txt','rt',encoding='utf-8')  # Open file with 'UTF-8' 인코딩
text = f.read()
stopword = text.split('\n') 

# morph = kkma.pos(text)
