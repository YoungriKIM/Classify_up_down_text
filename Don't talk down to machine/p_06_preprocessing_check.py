# 전처리 거치며 어떻게 되는지 보기 위한 파일~

# predict가 수정되면 같이 수정 할 것!


# ------------------------------------------------------------
# 필요한 세트 모음
# for 정규식
import re
# for 토큰화
from konlpy.tag import Okt, Kkma
okt = Okt()
kkma = Kkma()
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
vocab_size = 4998 + 2
tokenizer = Tokenizer(vocab_size, oov_token='OOV')
# for 불용어 제거
f = open('../NLP/sample_data/stopword_02.txt','rt',encoding='utf-8')  # Open file with 'UTF-8' 인코딩
text = f.read()
stopword = text.split('\n') 
# for 정수 인코딩
import pickle
tag_data = pickle.load(open('../NLP/save/project_01_tag_data.pickle', 'rb'))
tokenizer.fit_on_texts(tag_data)
# for 패딩
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_len = 15


# # ------------------------------------------------------------
# 정확도가 가장 높은 가중치 가져와서 모델로 적용 
from tensorflow.keras.models import load_model
file_path = '../NLP/modelcheckpoint/project_01.h5'
model = load_model(filepath=file_path)
print('===== load complete =====')

# ------------------------------------------------------------
# predict용 함수 정의
def do_predict(new_sentence):

    # 전처리
    onlykorean = re.compile("[^ㄱ-ㅎㅏ-ㅣ가-힣]").sub(' ',new_sentence)   # 한글 아닌 문자 제외
    print('한글만 남기기: ', onlykorean)
    token = kkma.morphs(onlykorean)     # 토큰화
    print('토큰화:', token)
    delstopword = [word for word in token if not word in stopword]  # 불용어 제거
    print('불용어 제거: ', delstopword)
    encoded = tokenizer.texts_to_sequences([token])   # 정수 인코딩
    print('정수 인코딩: ', encoded)
    padded = pad_sequences(encoded, maxlen = max_len, padding='pre')     # 패딩
    print('패딩: ', padded)

    # 예측
    score = float(model.predict(padded))
    if(score > 0.5):
        print(new_sentence, '는 {:.2f} % 확률로 존댓말입니다.'.format(score * 100))
    else:
        print(new_sentence, '는 {:.2f} % 확률로 반말입니다.'.format((1 - score) * 100))

# =============================================================
do_predict('이렇게 말했나요?')
do_predict('이렇게 말하지요')
do_predict('이렇게 말해보세요')
do_predict('이렇게 말한 거에요')
do_predict('이렇게 말하지 마시오')
do_predict('이렇게 말하죠')
do_predict('이렇게 말했죠')

# =============================================================
# 정리 중
# https://docs.google.com/spreadsheets/d/17OxKDrjJH6KJs_J-Ng0AsvaZXp6svhGdKIjEbLP0Xhw/edit#gid=0

'''

한글만 남기기:  이렇게 말했나요 
토큰화: ['이렇', '게', '말하', '었', '나요']
불용어 제거:  ['이렇', '게', '말하', '었', '나요']
정수 인코딩:  [[3446, 25, 1, 1, 1]]
패딩:  [[   0    0    0    0    0    0    0    0    0    0 3446   25    1    1     1]]
이렇게 말했나요? 는 88.07 % 확률로 반말입니다.

한글만 남기기:  이렇게 말하지요
토큰화: ['이렇', '게', '말하', '지요']
불용어 제거:  ['이렇', '게', '말하', '지요']
정수 인코딩:  [[3446, 25, 1, 1]]
패딩:  [[   0    0    0    0    0    0    0    0    0    0    0 3446   25    1 1]]
이렇게 말하지요 는 94.75 % 확률로 반말입니다.

한글만 남기기:  이렇게 말해보세요
토큰화: ['이렇', '게', '말하', '어', '보', '세요']
불용어 제거:  ['이렇', '게', '말하', '보', '세요']
정수 인코딩:  [[3446, 25, 1, 1, 385, 3216]]
패딩:  [[   0    0    0    0    0    0    0    0    0 3446   25    1    1  385  3216]]
이렇게 말해보세요 는 68.63 % 확률로 반말입니다.

한글만 남기기:  이렇게 말한 거에요
토큰화: ['이렇', '게', '말', '하', 'ㄴ', '거', '에', '요']
불용어 제거:  ['이렇', '게', '말', 'ㄴ', '거', '요']
정수 인코딩:  [[3446, 25, 26, 1, 1, 2, 1, 7]]
패딩:  [[   0    0    0    0    0    0    0 3446   25   26    1    1    2    1     7]]
이렇게 말한 거에요 는 100.00 % 확률로 존댓말입니다.

한글만 남기기:  이렇게 말하지 마시오
토큰화: ['이렇', '게', '말하', '지', '마시', '오']
불용어 제거:  ['이렇', '게', '말하', '지', '마시']
정수 인코딩:  [[3446, 25, 1, 155, 1, 1]]
패딩:  [[   0    0    0    0    0    0    0    0    0 3446   25    1  155    1    1]]
이렇게 말하지 마시오 는 94.38 % 확률로 반말입니다.

한글만 남기기:  이렇게 말하죠
토큰화: ['이렇', '게', '말하', '죠']
불용어 제거:  ['이렇', '게', '말하', '죠']
정수 인코딩:  [[3446, 25, 1, 453]]
패딩:  [[   0    0    0    0    0    0    0    0    0    0    0 3446   25    1   453]]
이렇게 말하죠 는 97.83 % 확률로 존댓말입니다.

한글만 남기기:  이렇게 말했죠
토큰화: ['이렇', '게', '말하', '었', '죠']
불용어 제거:  ['이렇', '게', '말하', '었', '죠']
정수 인코딩:  [[3446, 25, 1, 1, 453]]
패딩:  [[   0    0    0    0    0    0    0    0    0    0 3446   25    1    1  453]]
이렇게 말했죠 는 98.92 % 확률로 존댓말입니다.

'''