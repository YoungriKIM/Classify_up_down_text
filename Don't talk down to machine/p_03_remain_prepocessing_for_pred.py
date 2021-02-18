# predict 전용 파일 만들기 전 필요한 것만 모은 파일

# ------------------------------------------------------------
# 필요한 세트 모음
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


# ------------------------------------------------------------
# 정확도가 가장 높은 가중치 가져와서 모델로 적용 
from tensorflow.keras.models import load_model
loaded_model = load_model('../NLP/modelcheckpoint/project_01.h5')
print('===== load complete =====')


# ------------------------------------------------------------
# predict용 함수 정의
def do_predict(new_sentence):
    # 전처리
    onlykorean = new_sentence.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")   # 한글 아닌 문자 제외
    token = okt.morphs(new_sentence)     # 토큰화
    delstopword = [word for word in token if not word in token]  # 불용어 제거
    encoded = tokenizer.texts_to_sequences(delstopword)   # 정수 인코딩
    paded = pad_sequences(encoded, maxlen = max_len, padding='pre')     # 패딩
    # 예측
    score = float(loaded_model.predict(paded))
    if(score > 0.5):
        print('{:.2f} % 확률로 존댓말입니다.\n'.format(score * 100))
    else:
        print('{:.2f} % 확률로 반말입니다.\n'.format((1 - score) * 100))


