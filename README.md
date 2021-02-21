# Natural-Language-Processing Personal Project

### 주제: 존대말, 반말 구분 분류 인공지능 💬

### NLP / RNN

---

### 파일 소개
- 1) data: 최종 모델과 쌍으로 필요한 data 모음
>> up_data_02.csv : 1차로 전처리하고 모은 존댓말 csv
>> 
>> down_data_02.csv : 1차로 전처리하고 모든 반말 csv
>> 
>> stopword_02.txt : 종미어구와 겹치는 단어 삭제한 불용어 모음 txt
>> 
>> project_011_tag_data.pickle : predict에 초기에 불러올 데이터셋

2) YUGYO Machine : 데이터 불러오기 부터 예측까지 진행한 파일

3) Final_model : 최종 완성한 모델들
>> gathering_data_02_up+add.py : 1차로 존댓말 데이터 모으기
>> 
>> gathering_data_02_down+add.py : 1차로 반말 데이터 모으기
>> 
>> YUGYO_final_model.py : 최종 완성
>> 
>> predict.py : MC 후 예측만 돌리기

---
Notion: https://www.notion.so/yugyomachine/9e90ce15f2514f328cc1eda7beaefaa5?v=00939d42013b400ea069c58b652d8d7d
