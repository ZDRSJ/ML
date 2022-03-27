import pandas as pd
import pickle


def make_bus_dict(dong):
    '''버스 노선 사전에서 동(key)를 넣으면 노선수(value)를 리턴하는 함수'''
    # 버스 노선 파일 불러오기
    bus_line_data = pd.read_excel("bus_line_data.xlsx")

    # 각 컬럼별 값들 리스트로 만들기
    adr_dong = list(bus_line_data['adr_dong'])
    bus_num = list(bus_line_data['버스노선수'])

    # dict로 키-값 형태로 저장하기
    bus_dict = {}

    for i in range(0, len(adr_dong)):
        bus_dict[adr_dong[i]] = bus_num[i]

    return bus_dict[dong]

# test
# print(make_bus_dict())

# pickle 열기
with open('model.pkl','rb') as pickle_file:
    model = pickle.load(pickle_file)


# 예측
def predict():
    '''pickle model에 유저 input값을 변환하여 넣고 예측값을 변환하는 함수
    1. 유저의 input 값 : 동, 년도, 면적
        - 동 : 최초에 data를 0으로 모두 채워넣고, 해당 동만 1로 바꿔준다
    2. 함수에서 만드는 x값 : 동 -> 버스 dict에서 연결하여 넣기'''

    # request는 flask에서 썼던 함수라서 django에 맞는 함수를 사용해야함
    data_dong = request.form['dong']
    data_year = int(request.form['year'])
    data_area = int(request.form['area'])

    # 버스노선수 dict에서 뽑기
    data_bus = make_bus_dict(data_dong)

    temp = {
        'year' : data_year,
        '전용면적' : data_area,
        '버스노선수' : data_bus,
        '가좌동' : 0,
        '고양동' : 0,
        '관산동' : 0,
        '대화동' : 0,
        '덕은동' : 0,
        '덕이동' : 0,
        '도내동' : 0,
        '동산동' : 0,
        '마두동' : 0,
        '백석동' : 0,
        '사리현동' : 0,
        '삼송동' : 0,
        '성사동' : 0,
        '성석동' : 0,
        '식사동' : 0,
        '신원동' : 0,
        '원흥동' : 0,
        '일산동' : 0,
        '장항동' : 0,
        '주교동' : 0,
        '주엽동' : 0,
        '중산동' : 0,
        '지축동' : 0,
        '탄현동' : 0,
        '토당동' : 0,
        '풍동' : 0,
        '행신동' : 0,
        '향동동' : 0,
        '화정동' : 0
    }

    # 해당 동은 1로 바꿔주기
    temp[data_dong] = 1

    # dataframe(series) 형태로 변환
    data_f = pd.DataFrame([temp])

    # model에 넣어서 예측값 구하기
    result = model.predict(data_f)

    return int(result[0])



