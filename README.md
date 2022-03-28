## ML repository 안내

---
⚙️ Stack & Tool used
---
- python
- visual studio code

---
📂 Directory Structure
---

🗂 ML_code_file : ML 작업 코딩 폴더

ㄴ 🗳 ml_data.ipynb : 머신러닝 모델 작업 파일


ㄴ 🗳 ZDRSJ_model.pkl


🗂 data : 사용 데이터셋 폴더

ㄴ 🗳 ml_data.csv : 머신러닝 모델에 사용을 위한 EDA 데이터셋

    
 ㄴ 🗂 python
 
   

 ---

## 🤖 modeling(모델링)

1. 모델 선택
    1. 머신 러닝 방법과 시계열분석 모형 중 논문을 참고하여, 예측의 정확성이 더 우수한 머신러닝 방법 선택 
      1. [참고 논문](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002324086)
      2. 연구 결과 요약
          1. 첫째, 머신 러닝 방법의 예측력이 시계열분석 모형보다 우수한 것으로 나타났다.
          2. 둘째, 시장이 안정적인 상황에서는 머신 러닝 방법과 시계열분석 방법 모두 시장 추세를 적절히 예측하는 것으로 나타났다.
          3. 셋째, 구조적인 변화 또는 외부 충격으로 시장이 급변하는 경우 머신 러닝 방법은 시장 추세를 대체로 유사하게 예측하는 것으로 나타났으나, 시계열분석 방법은 시장 추세를 전혀 예측할 수 없는 것으로 나타났다. 향후 머신 러닝 방법을 활용함으로써 부동산 시장에 대한 예측의 정확성이 향상될 것으로 기대된다.
  2. 선택한 머신 러닝 모델 : **`LightGBM`**
      1. `LightGBM`과 `XGBoost` 두 모델의 성능을 RSME로 평가하여 성능을 비교한 결과 우수한 모델 채택
2. 평가 지표 선정
   1. RSME
       1. MSE에서 루트를 취하기 때문에, MSE의 단점을 어느 정도 해소
           1. MSE의 단점 : 예측값과 정답의 차이를 제곱하기 때문에, 이상치에 대해 민감 → 즉, 오차가 0과 1 사이인 경우에, MSE에서 그 오차는 본래보다 더 작게 반영되고, 오차가 1보다 클 때는 본래보다 더 크게 반영
       2. 이상치에 대한 민감도가 MSE와 MAE 사이에 있기 때문에, 이상치를 적절히 잘 다룬다고 간주되는 경향이 있음 → 즉, MSE보다 이상치에 상대적으로 둔감
3. 모델링 & 평가 결과
    1. 모델에 넣은 데이터 변수 : 동, 버스노선수, 단위당 가격, 연도
    2. 시계열 데이터이기 때문에, TimeSeriesSplit 기법을 사용
    3. `RMSE` : 22936.894, `R2 score` :  0.388로 저조하여 추후 보완 필요
    4. `feature_importances` 결과 : 면적, 연도, 버스 노선 수 순으로 높음
    <img width="601" alt="fi" src="https://user-images.githubusercontent.com/89832134/160360847-8aeffc37-b06e-4c19-87b3-7e1e9daa4189.png">

4. 보완해야 할 점
    1. 최초의 계획은 서포트 벡터 머신, 랜덤 포레스트, 그래디언트  부스팅 회귀 트리 등의 다양한 머신러닝 모델과 더불어 시계열분석 모형 모델까지 같이 모델링 하여 앙상블 기법 또는 비교하여 가장 좋은 성능을 나타내는 모델을 채택하여 진행하려고 하였으나 시간 부족으로 두 모델만 사용하여 모델링 진행
    2. 추후 프로젝트 보안 간 다양한 모델을 활용한 앙상블 기법 활용 예정
    3. 크롤링 등으로 얻은 더 많은 독립 변수가 있었으나, 서비스 관점에서 유저에게 입력받을 수 있는 값이 제한되어 모델링 전에 많은 변수를 drop 하였음. 추후 어떻게 보완해야 할지 고민 필요
    4. 공공데이터를 활용하여 독립 변수 추가 예정
