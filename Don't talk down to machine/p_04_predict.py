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
model = load_model('../NLP/modelcheckpoint/project_01.h5')
print('===== load complete =====')

# ------------------------------------------------------------
# predict용 함수 정의
def do_predict(new_sentence):

    # 전처리
    onlykorean = re.compile("[^ㄱ-ㅎㅏ-ㅣ가-힣]").sub(' ',new_sentence)   # 한글 아닌 문자 제외
    token = okt.morphs(onlykorean)     # 토큰화
    delstopword = [word for word in token if not word in stopword]  # 불용어 제거
    encoded = tokenizer.texts_to_sequences([delstopword])   # 정수 인코딩
    padded = pad_sequences(encoded, maxlen = max_len, padding='pre')     # 패딩

    # 예측
    score = float(model.predict(padded))
    if(score > 0.5):
        print(new_sentence, '는 {:.2f} % 확률로 존댓말입니다.\n'.format(score * 100))
    else:
        print(new_sentence, '는 {:.2f} % 확률로 반말입니다.\n'.format((1 - score) * 100))

# =============================================================
do_predict('이미 시작한 일 되돌릴 수는 없어요~~~!')
do_predict('만가서 반갑습니다 김영리에요.')
do_predict('저희 같이 코딩 합시다')
do_predict('존댓말에도 여러가지 종류가 있다는 걸 기억하십시오')
do_predict('내가 이렇게 반말해도 넌 모르겠지')
do_predict('이녀석 다 반말로 보이니ㅋㅋㅋㅋ')
do_predict('잘하자^^')
do_predict('슬슬 졸린데 이제 자러 갈까?')
do_predict('근데 오빠 왜 안 자..?')

# =============================================================
# 이미 시작한 일 되돌릴 수는 없어요~~~! 는 100.00 % 확률로 존댓말입니다.

# 만가서 반갑습니다 김영리에요. 는 99.98 % 확률로 존댓말입니다.

# 저희 같이 코딩 합시다 는 96.24 % 확률로 존댓말입니다.

# 존댓말에도 여러가지 종류가 있다는 걸 기억하십시오 는 71.37 % 확률로 반말입니다.

# 내가 이렇게 반말해도 넌 모르겠지 는 98.55 % 확률로 반말입니다.

# 이녀석 다 반말로 보이니ㅋㅋㅋㅋ 는 78.23 % 확률로 반말입니다.

# 잘하자^^ 는 100.00 % 확률로 반말입니다.

# 슬슬 졸린데 이제 자러 갈까? 는 85.24 % 확률로 존댓말입니다.

# 근데 오빠 왜 안 자..? 는 100.00 % 확률로 반말입니다.