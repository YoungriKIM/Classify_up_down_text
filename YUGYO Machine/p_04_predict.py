# predict 진행하고 기록하기 위한 파일

# ------------------------------------------------------------
# 필요한 세트 모음
# for 정규식
import re
# for 토큰화
from konlpy.tag import Okt
okt = Okt()
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
    token = okt.morphs(onlykorean)     # 토큰화
    delstopword = [word for word in token if not word in stopword]  # 불용어 제거
    encoded = tokenizer.texts_to_sequences([token])   # 정수 인코딩
    padded = pad_sequences(encoded, maxlen = max_len, padding='pre')     # 패딩

    # 예측
    score = float(model.predict(padded))
    if(score > 0.5):
        print(new_sentence, '는 {:.2f} % 확률로 존댓말입니다.'.format(score * 100))
    else:
        print(new_sentence, '는 {:.2f} % 확률로 반말입니다.'.format((1 - score) * 100))

# =============================================================
do_predict('이렇게 말해')
do_predict('이렇게 말하곤 해')
do_predict('이렇게 말해볼까')
do_predict('이렇게 말하니까')
do_predict('이렇게 말하는거 어떠냐')
do_predict('이렇게 말하는구나')
do_predict('이렇게 말하지마')
do_predict('이렇게 말하잖아')
do_predict('이렇게 말하더라')
do_predict('이렇게 말하잖니')
do_predict('이렇게 말하네')
do_predict('이렇게 말했어')
do_predict('이렇게 말하자')
do_predict('이렇게 말하면 안돼')
print('-------------------------------')
do_predict('이렇게 말해요')
do_predict('이렇게 말했어요')
do_predict('이렇게 말했다니까요')
do_predict('이렇게 말했나요?')
do_predict('이렇게 말하지요')
do_predict('이렇게 말했잖아요')
do_predict('이렇게 말해보세요')
do_predict('이렇게 말한 거에요')
do_predict('이렇게 말하지 마시오')
do_predict('이렇게 말하죠')
do_predict('이렇게 말했죠')
do_predict('이렇게 말했습니다')
do_predict('이렇게 말합시다')

# =============================================================
# 정리 중
# https://docs.google.com/spreadsheets/d/17OxKDrjJH6KJs_J-Ng0AsvaZXp6svhGdKIjEbLP0Xhw/edit#gid=0