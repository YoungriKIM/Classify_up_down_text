# okt로 품사 태깅하여 진행했으나 다른 기능들과 비교 필요하여 작성
# 이 파일은 확인만 하는용. 비교는 compare_morphs_tagging 파일로 !!!

# tagging 기능 다르게 사용하여 비교 : okt/ kkma/ hannanum/ komoran/Mecab
import re
from konlpy.tag import Okt, Kkma, Hannanum, Komoran
okt = Okt()
kkma = Kkma()
nanum = Hannanum()
komo = Komoran()

# ---------------------------------------------------------
# 유의깊게 볼 부분
# 존댓말 니다, 시다, ㄴ다, ㅆ다 / 반말 다
# 나요 / 지요 / 아요 / 세요 

def tagging(text):
    onlykorean = re.compile("[^ㄱ-ㅎㅏ-ㅣ가-힣]").sub(' ',text)   # 한글 아닌 문자 제외
    token = komo.morphs(onlykorean)
    print(text,'\n',token)

# ---------------------------------------------------------
tagging('겨울이 지나갔습니다.')
tagging('겨울은 지나갔다 합시다.')
tagging('아버지가방에들어가신다.')
tagging('겨울이 지나갔다.')
print('--------------------------------------')
tagging('배 고프지 않나요?.')
tagging('이럴때 배가 고프지요.')
tagging('이럴때 배가 고프죠.')
tagging('저는 배가 고파요.')
tagging('어서 점심을 시키세요.')






