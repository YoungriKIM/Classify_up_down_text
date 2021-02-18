# 정규표현식 정리 >  https://wikidocs.net/4308
# 정규표현식 맞는지 확인 > https://regexr.com/

# 정규 표현식 re(regular expression)

# 한글만 가져오기
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
