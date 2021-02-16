# 토큰화 작업 후에 텍스트 데이터를 용도에 맞게 정체 및 정규화 해야 함

# + 정제 cleaning : 갖고 있는 corpus로부터 노이즈 데이터를 제거한다.
# + 정규화 normalization : 표현 방법이 다른 단어들을 통합시켜 같은 단어로 만들어준다.

# 1) 같은 의미의 단어 통합
# 2) 대, 소문자 통합
# 3) 불필요한 단어 제거(등장 빈도가 적은 단어/길이가 짧은 단어)
# 4) 정규 표현식(게재 시간, 날짜 등의 정규 표현식)

# ===========================
# [존반분류 프로젝트]의 경우 숫자, 영어 등을 없애도 될 것 같음
