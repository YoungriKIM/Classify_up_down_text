# 불용어에서 종미어구 삭제하기
# 이 부분은...다시...하....


import numpy as np
from konlpy.tag import Kkma
kkma = Kkma()

f = open('../NLP/sample_data/stopword.txt','rt',encoding='utf-8')  # Open file with 'UTF-8' 인코딩
text = f.read()
# lines = text.split('\n') 

morph = kkma.pos(text)
# print(morph)


drop_sample = ['EFN']

remain_stopword = []
flag = 1
for i in drop_sample:
    for x in morph:
        if i not in x:
            flag = 1
            continue
        else:
            flag = 0
            break
    if(flag==0):
        drop_sample.append(x)

print(drop_sample)

# morph = np.delete(morph, drop_sample)

# print(morph)
