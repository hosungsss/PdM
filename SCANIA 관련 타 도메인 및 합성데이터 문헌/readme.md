# Scania APS 기반 예지보전 핵심 문헌 2편 심층 비교 분석

---

## 1. 다중 특성 선택 및 심층 신경망(DNN) 기반 고장 진단 연구

### [문헌 1] Taghandiki & DallakehNejad (2023)
* **연구(저자 및 연도):** Kazem Taghandiki, Morteza DallakehNejad (2023)[cite: 10]
* **연구 대상 도메인:** Scania 트럭 공기압 시스템(Air Pressure System, APS) 결함 진단, 대규모 고차원 센서 스냅샷 분류[cite: 10]
* **연구내용:**
  * **데이터셋 구성:** UCI 머신러닝 저장소의 Scania APS 벤치마크(Train 60,000건, Test 16,000건, 170개 수치형 센서 변수)를 활용함[cite: 10].
  * **3중 교차 특성 선택(Feature Selection):** 고차원 노이즈를 줄이기 위해 Information Gain, Correlation, SVM 가중치 부여 알고리즘의 공통 교집합을 추출하여 170개 변수 중 상위 21개 핵심 센서(`ay_008`, `bj_000`, `ah_000` 등)를 선별함[cite: 10].
  * **전처리 및 결측 대체:** 선별된 21개 변수의 결측치를 RapidMiner의 `Replace All Missing` 연산자를 통해 속성별 평균값으로 일괄 대체하고, 레이블 속성을 이항(Binomial) 형태로 정규화함[cite: 10].
  * **심층 신경망(DNN) 모델링:** Tanh 활성화 함수를 기반으로 4개 레이어(입력 21 - 은닉 50 - 은닉 50 - 출력 2)로 구성된 완전연결 심층 신경망(DNN)을 구축하여 테스트 세트 분류 정확도 98.66%를 달성함[cite: 10].
  * **전통 머신러닝 비교:** SVM, Naive Bayes, Decision Tree, Random Forest, KNN 5개 모델과 비교하여 정확도(98.66%) 및 F-Measure(68.99%)에서 우위를 보임[cite: 10].
* **함의점:**
  * 170개에 달하는 복잡한 산업용 센서 데이터를 다중 필터링 기법으로 21개로 압축하여 모델 연산 부담을 줄이고 높은 일반 정확도를 확보할 수 있음을 입증함[cite: 10].
  * **한계 및 비용 취약성:** 산업 현장의 비대칭 오류 비용(미탐지 $\text{FN}=\$500$ vs 오탐지 $\text{FP}=\$10$)을 훈련 손실 함수에 반영하지 않고 일반 정확도에만 최적화함[cite: 10, 11]. 그 결과 테스트 세트에서 137건의 미탐지(FN)가 발생하여 총 수리 비용이 \$69,270(FN 비용 \$68,500)에 달하는 치명적인 경제적 손실을 초래함[cite: 10, 11].

---

## 2. 비용 민감 트랜스포머 및 하이브리드 리샘플링 기반 예지보전 연구

### [문헌 2] Beikmohammadi et al. (2026)
* **연구(저자 및 연도):** Ali Beikmohammadi, Mohammad Hosein Hamian, Neda Khoeyniha, Tony Lindgren, Olof Steinert, Sindri Magnússon (2026)[cite: 11]
* **연구 대상 도메인:** Scania 트럭 APS 데이터셋 및 SECOM 반도체 제조 공정, 극단적 불균형(Extreme Imbalance) 및 비대칭 비용 민감(Cost-Sensitive) 환경의 산업용 예지보전[cite: 11]
* **연구내용:**
  * **통계적 결측 정제:** 결측률 10%를 초과하는 28개 고결측 변수를 제거한 후, 잔여 결측치에 대해 베이지안 능형 회귀(Bayesian Ridge Regression)를 적용하여 100회 반복 추정 대체를 수행함[cite: 11].
  * **SVM-SMOTE + Repeated ENN 하이브리드 리샘플링:** SVM 결정 경계 인근의 소수 클래스를 50% 수준까지 오버샘플링(SVM-SMOTE)한 뒤, Repeated ENN을 통해 다수 클래스 노이즈 샘플을 반복 제거하여 클래스 불균형 비율을 59:1에서 1.94:1로 안정화함[cite: 11].
  * **표 형태 트랜스포머(Tabular Transformer):** 정적 다변량 센서 간의 상호작용과 의미론적 의존성을 포착하기 위해 4개 블록, 4개 Multi-Head Attention, Conv1D 피드포워드 네트워크, Global Average Pooling, MLP 분류기로 구성된 전용 트랜스포머 신경망을 설계함[cite: 11].
  * **비용 민감 Focal Loss 설계:** 컴퓨터 비전의 클래스 불균형 해결용으로 주로 쓰이던 Focal Loss를 비대칭 비용 대응 대리 손실함수로 재해석하여 적용함[cite: 11]. 하이퍼파라미터 $\alpha=0.95, \gamma=1.5$를 설정하여 고비용 미탐지(FN)에 손실 페널티 가중치를 집중시킴[cite: 11].
  * **성능 달성:** Scania APS 테스트 세트에서 FN 6.8건, FP 4건으로 오탐지와 미탐지를 동시에 억제하며 총비용 **\$3,440**를 기록, 기존 SOTA 모델 대비 압도적인 비용 절감을 달성함[cite: 11]. SECOM 반도체 데이터셋에서도 총비용 \$950을 기록하여 범용성을 검증함[cite: 11].
* **함의점:**
  * 자연어/시계열에 주로 쓰이던 트랜스포머의 Self-Attention 구조를 표 형태(Tabular) 산업 센서 도메인에 성공적으로 이식하여 센서 간 비선형 복합 관계를 효과적으로 모델링함[cite: 11].
  * Focal Loss를 손실 함수로 활용함으로써 단순히 샘플 수를 맞추는 리샘플링을 넘어, 비즈니스 비용 행렬에 부합하는 경계면 학습을 정밀하게 유도할 수 있음을 이론적·실증적으로 규명함[cite: 11].
  * **한계점:** 해당 연구는 시간 축(Time-step)이 배제된 정적 단면 스냅샷 데이터를 다루고 있어, 시간 경과에 따른 부품의 점진적 마모 궤적이나 우측검열(Right Censoring) 문제를 직접적으로 해결하지는 못함[cite: 11].

---

## 3. 두 논문 핵심 메트릭 및 접근법 비교 요약

| 비교 항목 | [문헌 1] Taghandiki & DallakehNejad (2023)[cite: 10] | [문헌 2] Beikmohammadi et al. (2026)[cite: 11] |
| :--- | :--- | :--- |
| **데이터셋** | Scania APS (170개 수치형 변수)[cite: 10] | Scania APS + SECOM 반도체 제조 데이터[cite: 11] |
| **결측치 처리 방식** | 단순 평균값 일괄 대체 (`Replace All Missing`)[cite: 10] | 10% 초과 변수 제거 + 베이지안 능형 회귀 100회 반복 대체[cite: 11] |
| **불균형 처리 전략** | 별도 리샘플링 미수행 (원시 데이터 사용)[cite: 10] | SVM-SMOTE(오버) + Repeated ENN(언더) 하이브리드 샘플링[cite: 11] |
| **특성 선택 / 추출** | IG, Correlation, SVM 3중 교집합 선별 (21개 변수)[cite: 10] | 고결측 28개 제거 (142개 변수 전체 활용)[cite: 11] |
| **모델 아키텍처** | 4-Layer 완전연결 심층 신경망 (DNN)[cite: 10] | 4-Block Multi-Head Attention 기반 Tabular Transformer[cite: 11] |
| **비용 민감성 고려** | **미고려** (일반 분류 정확도 목적함수 최적화)[cite: 10] | **Focal Loss ($\alpha=0.95, \gamma=1.5$)**를 비용 손실 대리자로 채택[cite: 11] |
| **테스트 분류 결과** | TP 238, TN 15,548, **FP 77, FN 137**[cite: 10] | TP 368.2, TN 15,621, **FP 4, FN 6.8**[cite: 11] |
| **테스트 비용 성과** | **\$69,270** (정확도는 98.66%이나 막대한 손실 발생)[cite: 10, 11] | **\$3,440** (기존 SOTA 능가 및 총비용 극소화 달성)[cite: 11] |

---

## 4. Scania Component X 시계열 MH-DCC-DNN 제작을 위한 시사점

1. **정확도 최적화의 함정과 비용 민감 손실의 필수성:**
   * Taghandiki et al.의 결과는 98.66%라는 높은 정확도를 달성하더라도 비대칭 비용($\text{FN}=500$)을 통제하지 못하면 \$69,270에 달하는 심각한 재정적 실패로 이어진다는 점을 단적으로 보여줍니다[cite: 10, 11].
   * 박사님의 연구 파이프라인에서 검열 마스크 기반 비대칭 손실 함수(Cost-Sensitive Hazard Loss)를 반드시 모델링 중심에 두어야 함을 강력하게 지지합니다.
2. **Tabular Transformer의 장점과 시계열 구조의 결합:**
   * Beikmohammadi et al.은 표 형태 데이터에서 트랜스포머의 효용성을 증명했으나 시계열 동적 궤적을 반영하지 못했습니다[cite: 11].
   * 박사님의 MH-DCC 모델은 이를 확장하여, **1D-CNN 동적 시계열 채널(Within 열화 잔차)**과 **정적 사양 임베딩 채널(Between 베이스라인)**을 분리하고 FiLM 레이어로 결합함으로써 정적 센서 상관관계와 시계열 동역학을 동시에 완벽히 포착할 수 있습니다.
3. **오탐지(FP) 억제를 위한 온톨로지 가드레일과의 연결:**
   * Beikmohammadi et al.은 Focal Loss와 리샘플링을 통해 FP를 4건 수준으로 통제했습니다[cite: 11].
   * Component X와 같은 복합 다변량 시계열 환경에서는 딥러닝이 미탐지를 피하려다 과잉 경보(FP)를 발생시키는 경향이 커지므로, LMM/GEE로 검증된 Grade-1 ECDF 95% 분위수 기반 심볼릭 온톨로지 규칙(Override)을 결합하여 추가적인 FP 비용 폭증을 원천 차단하는 박사님의 구조가 학술적으로 독보적인 차별점을 가집니다.
