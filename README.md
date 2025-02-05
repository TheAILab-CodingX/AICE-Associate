# **KT AICE Associate 실습 코드**  

📌 **이 실습 코드는 더에이아이랩(The AI Lab)에서 출판한**  
📌 **AICE Associate 교재에서 사용하는 공식 실습 코드입니다.**  

---

## **📌 개요**
이 저장소는 **KT AICE Associate 자격시험**을 준비하는 학습자를 위한 **AI 모델링 실습 코드와 데이터를 포함**하고 있습니다.  
이 실습을 통해 **AI 모델링 프로세스(데이터 분석, 데이터 전처리, AI 모델링)를 단계별로 학습**할 수 있습니다.  

또한, **더에이아이랩(The AI Lab)에서는 AICE Associate 자격시험 준비를 위한 강의 VOD도 제공**하고 있습니다.  
**자격증 대비 강의가 필요하신 분들은 아래 링크를 통해 확인해 주세요.**  

🔗 **[AICE Associate 자격시험 대비 강의 VOD 구매하기](https://coding-x.com/class/15247/KT-AICE-ASSOCIATE-%EC%A2%85%ED%95%A9-(%EC%9D%B4%EB%A1%A0-+-%EC%8B%A4%EC%8A%B5))** 🚀 

---

## **📂 실습 파일 구성**
이 저장소에는 다음과 같은 주요 파일이 포함되어 있습니다.  

### **1️⃣ 실습 코드**
| 파일명 | 설명 |
|--------|------------------------------------------------|
| `AICE_asso_classification.ipynb` | 분류 문제 : 데이터분석 ~ AI 모델링 실습 |
| `AICE_asso_regression.ipynb` | 회귀 문제 : 데이터분석 ~ AI 모델링 실습 |

- **이 실습 코드는 KT AICE Associate 자격시험 대비를 위해 제작**되었습니다.  
- **데이터 분석 → 전처리 → 모델링 → 평가**의 AI 모델링 프로세스를 실습할 수 있습니다.  
- `AICE_asso_classification.ipynb`: **머신러닝/딥러닝 기반 분류 모델 구현 및 성능 평가**  
- `AICE_asso_regression.ipynb`: **머신러닝/딥러닝 기반 회귀 모델 구현 및 성능 평가 + 하이퍼파라미터 튜닝**  

---

### **2️⃣ 데이터 파일**
| 파일명 | 설명 |
|----------------------------|--------------------------------------|
| `Days_Information_Data.csv` | 일별 기상 및 환경 데이터 |
| `Metro_Traffic_Volume_Data.csv` | 지하철 교통량 데이터 |
| `hotel_bookings.csv` | 호텔 예약 관련 데이터 |

이 데이터들은 **머신러닝 및 딥러닝 모델을 학습시키기 위한 실습용 데이터셋**으로 활용됩니다.

---

## **🔧 실습 환경**
실습을 실행하기 위해서는 **Python과 주요 데이터 분석/머신러닝 라이브러리**가 필요합니다.  
아래 환경을 구축한 후 Jupyter Notebook을 실행하면 실습을 원활히 진행할 수 있습니다.

### **1️⃣ 필수 라이브러리 설치**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras xgboost lightgbm
```

### **2️⃣ Jupyter Notebook 실행**
```bash
jupyter notebook
```
실습 파일(`.ipynb`)을 실행하여 학습을 진행할 수 있습니다.

---

## **🛠 실습 개요**
각 실습 파일에서 진행하는 주요 과정을 간략히 설명합니다.

### **📌 Step 1. 데이터 분석 및 시각화**
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
df = pd.read_csv("hotel_bookings.csv")
# 데이터 구조 확인
print(df.info())
# 수치형 데이터 분포 확인 (시각화)
sns.histplot(df['lead_time'])  
plt.show()
```
✔ **데이터 구조 파악 (`info()`, `describe()`)**  
✔ **결측치 및 이상치 탐색**  
✔ **시각화를 통해 데이터 분포 분석**  

---

### **📌 Step 2. 데이터 전처리**
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# X, y 분리
X = df.drop(columns=['is_canceled'])  # 타겟 변수 제외
y = df['is_canceled']

# 학습데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
✔ **결측치 처리 및 범주형 데이터 변환(인코딩)**  
✔ **훈련 데이터와 테스트 데이터 분할 (`train_test_split`)**  
✔ **스케일링 적용 (`StandardScaler`)**  

---

### **📌 Step 3. 머신러닝 모델 학습 및 평가**
```python
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 모델 정의
model = XGBClassifier(n_estimators=100, max_depth=5)

# 모델 학습
model.fit(X_train_scaled, y_train)

# 예측
y_pred = model.predict(X_test_scaled)

# 모델 성능 평가
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
```
✔ **머신러닝 모델 학습 및 예측 (`fit()`, `predict()`)**  
✔ **모델 성능 평가 (`accuracy_score`, ...)**  

---

### **📌 Step 4. 딥러닝 모델 학습 및 평가**
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 모델 정의
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test))
```
✔ **Keras를 활용한 신경망 모델 구축 (`Sequential()`)**  
✔ **신경망 학습 (`fit()`) 및 평가 (`evaluate()`)**  

---

## **💡 KT AICE Associate 시험 대비 가이드**
이 실습 파일들은 **KT AICE Associate 자격시험**을 준비하는 데 도움이 됩니다.  
시험에서는 **AI 모델링 프로세스(데이터 분석, 전처리, 모델링) 전반을 이해하고 구현하는 능력**이 요구됩니다.

📌 **시험 대비 핵심 개념**  
✅ **데이터 처리 및 비시각화 분석** : pandas의 주요 함수 (`groupby`, `merge`, `fillna`, `drop` 등) 활용법 숙지  
✅ **데이터 시각화 분석** : matplotlib & seaborn으로 그래프 생성 흐름 (`figure` → `plot` → `show`) 익히기  
✅ **데이터 전처리** : scikit-learn의 일관성 이해 (`객체 생성` → `fit()` → `transform()`)  
✅ **머신러닝 & 딥러닝 모델의 전반적인 구현 프로세스 이해** (`모델 정의(및 컴파일)` → `모델 학습` → `모델 평가`)

시험은 실습 코드에 포함되지 않은 내용도 출제될 수 있으므로,  
**단순히 코드를 외우기보다 AI 구현 프로세스를 전반적으로 이해하는 것이 중요합니다!** 🚀  

---

### **📢 AICE BASIC vs AICE Associate 비교**
<table>
  <tr align="center">
    <th>구분</th>
    <th>AICE BASIC</th>
    <th>AICE Associate</th>
  </tr>
  <tr align="center">
    <td><strong>대상</strong></td>
    <td>비전공자, 기초 학습자</td>
    <td>Python으로 실질적인 AI 구현 프로세스를 학습하고자 하는 자</td>
  </tr>
  <tr align="center">
    <td><strong>코딩 여부</strong></td>
    <td>🟢 <strong>Python 없이</strong> 🟢데이터 분석 및 모델링 실습 가능</td>
    <td>🔵 <strong>Python 기반</strong> 🔵으로 데이터 분석 및 모델 구현</td>
  </tr>
  <tr align="center">
    <td><strong>학습 난이도</strong></td>
    <td><strong>초급</strong> (기초 AI 개념 및 실습)</td>
    <td><strong>중급</strong> (실제 AI 모델 구현)</td>
  </tr>
  <tr align="center">
    <td><strong>배지</strong></td>
    <td><img src="https://github.com/TheAILab-CodingX/AICE-Associate/blob/main/images/%E1%84%87%E1%85%A2%E1%84%8C%E1%85%B5_BASIC.png" width="120"></td>
    <td><img src="https://github.com/TheAILab-CodingX/AICE-Associate/blob/main/images/%E1%84%87%E1%85%A2%E1%84%8C%E1%85%B5_Asso.png" width="120"></td>
  </tr>
  <tr align="center">
    <td><strong>공인교육기관</strong></td>
    <td colspan="3"><strong><img src="https://github.com/TheAILab-CodingX/AICE-Associate/blob/main/images/logo_theailab.png" width="150"></strong></td>
  </tr>
</table>


🔗 **[AICE BASIC 자격시험 대비 강의 VOD 구매하기](https://coding-x.com/class/15224/KT-AICE-BASIC-%EC%A2%85%ED%95%A9-(%EC%9D%B4%EB%A1%A0+%EC%8B%A4%EC%8A%B5))** 🚀  
🔗 **[AICE Associate 자격시험 대비 강의 VOD 구매하기](https://coding-x.com/class/15247/KT-AICE-ASSOCIATE-%EC%A2%85%ED%95%A9-(%EC%9D%B4%EB%A1%A0-+-%EC%8B%A4%EC%8A%B5))** 🚀  

---

📌 **AICE BASIC과 AICE Associate 시험을 모두 대비하고 싶다면, 두 강의를 함께 수강하는 것을 추천합니다.**  
📌 **더에이아이랩(The AI Lab) 강의를 통해 AI 모델링 프로세스를 단계별로 실습하고, 자격증을 효과적으로 준비하세요!** 🚀

📌 **이 저장소가 도움이 되셨다면, GitHub ⭐ Star를 눌러주세요!** 😊 🚀
