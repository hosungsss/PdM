# Scania Component X 예지보전(PdM) 핵심 문헌 계통별 심층 분석 보고서

---

## 1. 표준 벤치마크 및 데이터셋 정의 연구 (Benchmark Foundation)

### [연구 1] Kharazian et al. (2025)
* **저자 및 연도:** Zahra Kharazian, Tony Lindgren, Sindri Magnússon, Olof Steinert, Oskar Andersson Reyna (2025)[cite: 5]
* **논문명:** *SCANIA Component X dataset: a real-world multivariate time series dataset for predictive maintenance* (*Nature Scientific Data*, 12:493)[cite: 5]
* **연구 대상 도메인:** 대형 트럭(Heavy-Duty Trucks)의 미공개 핵심 엔진 부품(Component X), 산업용 다변량 불규칙 시계열 예지보전 벤치마크[cite: 5]
* **연구 내용:**
  * **데이터 규모 및 구성:** 스웨덴 Scania AB 실운행 트럭 33,641대(Train 23,550대 / Validation 5,046대 / Test 5,045대)에서 수집된 152만 건의 시계열 Readout 데이터 공개[cite: 4, 5].
  * **피처 구조:** 차량 고유 ID 및 타임스텝을 포함하여 14개 센서 속성(6개 히스토그램 97개 bins + 8개 단일 누적 카운터 = 총 107개 시계열 변수)과 8개 범주형 정적 사양(`Spec_0`~`Spec_7`, 총 3,607개 고유 모델 조합)으로 구성[cite: 2, 5].
  * **우측검열 및 레이블링 체계:** 수리 시점($TTE$) 정보를 바탕으로 잔여 수명(RUL)에 따라 5단계 열화 등급(Class 0: $>48$, Class 1: $48\sim24$, Class 2: $24\sim12$, Class 3: $12\sim6$, Class 4: $6\sim0$ time step)을 정의[cite: 4, 5].
  * **비대칭 비용 행렬(Cost Matrix):** 현장 비즈니스 비용을 반영하여 미탐지(False Negative, 최대 500)에 극심한 페널티를 부여하고, 오탐지(False Positive, 7~10)에 낮은 비용을 책정하는 $5 \times 5$ 손실 평가 매트릭스 공식 정립[cite: 2, 5].
* **함의점:**
  * 기계 부품의 점진적 마모 과정을 반영하는 실차 다변량 시계열과 우측검열(Right Censoring) 데이터를 산업계 최초로 개방하여 재현 가능한 표준 연구 환경을 구축함[cite: 5].
  * 단순 정확도(Accuracy)가 아닌 경제적 총비용(Total Cost) 기반의 평가 패러다임을 확립하여, 산업용 AI 모델이 지향해야 할 현실적 최적화 목적함수를 제시함[cite: 5].

---

## 2. 기하학적 딥러닝 및 그래프 신경망 연구 (Geometric Deep Learning & GNN)

### [연구 2] Parton et al. (2024)
* **저자 및 연도:** Maurizio Parton, Andrea Fois, Michelangelo Vegliò, Carlo Metta, Marco Gregnanin (2024)[cite: 3]
* **논문명:** *Predicting the Failure of Component X in the Scania Dataset with Graph Neural Networks* (IDA 2024, *LNCS* Vol. 14642)[cite: 2, 3]
* **연구 대상 도메인:** 시계열 데이터의 위상수학적 그래프 변환, Graph Neural Networks (GNN) 기반 고장 조기 예측[cite: 3]
* **연구 내용:**
  * **시계열 시그니처(Path Signature) 기반 피처 선정:** 경로 이론(Path Theory)의 Lead-Lag 변환 및 절단 시그니처($\mathcal{S}_M$)를 적용하고, 기대 시그니처 정리(Expected Signature Theorem)에 기반한 거리 행렬 PCA를 수행하여 전체 시계열 중 가장 유의미한 변수로 `171_0`을 도출[cite: 3].
  * **가시성 그래프(Visibility Graph, VG) 구축:** 선별된 `171_0` 단변량 시계열을 가시 조건 수식에 따라 무방향 복잡계 그래프 네트워크로 변환(노드 = 타임스텝, 엣지 = 가시선 연결)[cite: 3].
  * **GIN(Graph Isomorphism Network) 모델링:** 114개 입력 채널(시계열 변수 + 타임스텝 + 사양)과 912개 은닉 채널, 15개 레이어로 구성된 GIN 네트워크 학습[cite: 3].
  * **이진화 압축 전략:** 미예측 고비용(FN 500)을 회피하기 위해 훈련 시 클래스 {1, 2, 3}을 모두 Class 4(고장)로 통합 취급하고, Class 0 임계값을 0.6으로 설정하여 검증 비용 40,109점 기록[cite: 2, 3].
* **함의점:**
  * 비유클리드(Non-Euclidean) 공간인 그래프 구조를 통해 센서 시계열의 시계열적 추세와 기하학적 구조를 동시에 보존하는 방법론적 참신성을 입증함[cite: 3].
  * 그러나 단일 센서(`171_0`)에 의존하여 그래프를 생성함으로써 다변량 센서 간의 교차 상관성을 충분히 활용하지 못했고, 고장 클래스를 Class 4로 단일화함에 따라 중간 단계의 세밀한 리드타임 제어 능력이 상실됨[cite: 3].

---

## 3. 맥락 인식 및 특징 공학·생존분석 연구 (Contextual Feature Engineering & Survival Analysis)

### [연구 3] Carpentier et al. (2024)
* **저자 및 연도:** Louis Carpentier, Arne De Temmerman, Mathias Verbeke (2024)[cite: 4]
* **논문명:** *Towards Contextual, Cost-Efficient Predictive Maintenance in Heavy-Duty Trucks* (IDA 2024, *LNCS* Vol. 14642)[cite: 2, 4]
* **연구 대상 도메인:** 대규모 자동 피처 추출(tsfresh), 차량 사양 기반 맥락적 클러스터링(Contextual Modeling), 가속수명 생존분석(AFT)[cite: 4]
* **연구 내용:**
  * **불균일 시계열 윈도우 및 피처 추출:** 누적 카운터의 1차 차분($\Delta$)을 적용한 후, 고정 크기/고정 시간/색인 기반의 3가지 슬라이딩 윈도우($w \in \{4, 8, 16, 32, 64\}$)를 설계[cite: 4]. `tsfresh`를 통해 342,300개 피처를 계산한 뒤 Kendall's $\tau_B$($p < 0.01$) 및 Pearson 상관계수($\rho < 0.5$) 필터링으로 704개 핵심 피처 선별[cite: 4].
  * **차량 사양 기반 맥락 모델링(Contextualization):** 사양 일치율과 정규화 센서 트렌드의 볼록 결합(Convex Combination) 거리를 정의하고, 계층적 군집화(Hierarchical Clustering)를 통해 사양 클러스터별 전용 서브 모델을 학습[cite: 4].
  * **다양한 패러다임 비교:** 분류(Classification), 회귀(Regression, 연속값 라운딩), 생존분석(Survival Analysis - XGBoost with Accelerated Failure Time)을 병렬 비교[cite: 4].
  * **비용 역설 발견:** 비대칭 비용 행렬 하에서 무조건 고장으로 판정하는 베이스라인(`Always predict 4`)이 차량당 평균 비용 약 9.9점의 매우 강력한 기준선으로 작동함을 증명 (생존분석 XGBoost가 약 9.8~10.0점 수준 기록)[cite: 4].
* **함의점:**
  * 차량 사양(Spec)을 독립적인 맥락(Context)으로 분리하여 조건부 학습을 시도한 선구적 연구임[cite: 4].
  * 극심한 비대칭 비용 하에서는 정밀한 모델이라도 조금만 보수적으로 예측하면 '무조건 고장 경보를 울리는 단순 규칙'보다 경제성이 떨어질 수 있다는 산업적 경보 남발 딜레마(Alarm Fatigue Paradox)를 실증적으로 규명함[cite: 4].

---

## 4. 다단계 전처리 및 전통적 머신러닝 최적화 연구 (Multi-stage Preprocessing & Tabular ML)

### [연구 4] Ferrisi et al. (2026)
* **저자 및 연도:** Stefania Ferrisi, Romina Conte, Rosita Guido, Giuseppina Ambrogio (2026)[cite: 2]
* **논문명:** *Enhancing Predictive Maintenance for Heavy Vehicles: A Multi-Stage Feature Selection and Preprocessing Framework on the Scania Dataset* (*Procedia Computer Science*, Vol. 277)[cite: 2]
* **연구 대상 도메인:** KDD(Knowledge Discovery in Database) 프레임워크 기반 다단계 데이터 전처리, 단일 시점 표(Tabular) 머신러닝 최적화[cite: 2]
* **연구 내용:**
  * **단일 시점 축약 및 결측 처리:** 시계열 전체를 학습하지 않고 차량별 마지막 관측치(Last Observation) 1개 행만 추출[cite: 2]. 국소 선형 보간 및 동일 차종 기반 선형 혼합 모델(LMM, statsmodels) 회귀 대체를 복합 적용[cite: 2].
  * **사양 통합 및 불균형 보정:** 8개 범주형 사양을 결합하여 3,607개 단일 'model' 변수로 재인코딩하고, 소수 고장 클래스를 증강하기 위해 SMOTE 오버샘플링을 적용[cite: 2].
  * **Random Forest 피처 선택:** Random Forest 중요도 평가를 통해 최종 53개 핵심 수치 변수를 추출[cite: 2].
  * **이진화 전환 및 Optuna 최적화:** 5-Class 다항 분류와 Healthy vs Faulty 이진 분류를 비교[cite: 2]. Optuna 프레임워크를 통해 DT, RF, XGBoost, SVM의 하이퍼파라미터를 최적화한 결과, Binary Random Forest가 검증 세트 최저 비용 36,595점 달성 (테스트 세트 비용 42,976점)[cite: 2].
* **함의점:**
  * 복잡한 시계열 딥러닝 없이 정밀한 통계적 결측치 대체(LMM)와 마지막 관측치 기반 이진 머신러닝만으로도 기존 다항 분류 벤치마크 비용을 능가할 수 있음을 입증함[cite: 2].
  * 그러나 시계열 동적 궤적이 완전히 유실되었고, 전체 정확도가 72.16%에 불과하며, 5개 등급 예측을 포기함으로써 고장 시점까지 남은 리드타임(Lead-time)을 전혀 파악할 수 없는 실무적 한계를 지님[cite: 2].

---

## 5. 비용 민감형 시계열 딥러닝 및 의사결정 최적화 연구 (Cost-Sensitive Deep Learning)

### [연구 5] Mraihi et al. (2026)
* **저자 및 연도:** Abdelhakim Mraihi, Valeriu Dimidov, Raoof Doorshi, Reza Khoshkangini (2026)[cite: 6]
* **논문명:** *Cost-Sensitive Deep Learning for Scania Component X: Minimising Operational Cost via Asymmetric Threshold Optimisation* (PHM Society European Conference 2026)[cite: 6]
* **연구 대상 도메인:** 시계열 딥러닝 아키텍처(CNN, Transformer, TCN), 가중 손실 함수 기반 비용 민감 학습, 비대칭 임계값 최적화[cite: 6]
* **연구 내용:**
  * **시계열 텐서화 및 피처 정제:** 차량별 최근 30개 타임스텝 윈도우를 추출(Zero-padding 적용)하고, 분산 기반 선택으로 64개 센서 변수를 추출하여 $(30 \times 64)$ 크기의 텐서 입력 파이프라인 구성[cite: 6].
  * **가중 이진 교차 엔트로피(WBCE) 학습:** 5등급을 Healthy(0) vs At-Risk(1~4)로 재정의하고, 극단적 비대칭성을 고려하여 $w_1=10, w_0=1$의 가중 손실 함수로 최적화 안정성 확보[cite: 6].
  * **사후 비대칭 임계값($\theta^*$) 최적화:** 이론적 베이즈 임계값($\theta_{\text{Bayes}} \approx 0.02$) 부근을 탐색하여 검증 세트의 공식 $5 \times 5$ 비용 행렬을 최소화하는 결정 경계($\theta^*$)를 도출하고, 이를 Fail-safe 규칙에 따라 Class 4로 매핑[cite: 6].
  * **시계열 아키텍처 3종 비교:**
    * **Transformer:** 테스트 세트 총비용 40,195점(차량당 8.09점, 최저 비용 달성), 유지보수율 68.0%, FNR 13.4%[cite: 6].
    * **CNN:** 총비용 44,798점(차량당 9.01점), 유지보수율 49.7%, FNR 35.9% (안정적이나 위험 통제 부족)[cite: 6].
    * **TCN:** 총비용 48,260점(차량당 9.71점), FN=0(완벽한 고장 방지)을 달성했으나 유지보수율 100%(전체 차량 점검 권고)로 과잉 정비(Over-maintenance) 붕괴[cite: 6].
* **함의점:**
  * Self-Attention 메커니즘을 갖춘 Transformer가 다변량 센서 상호작용과 장기 열화 시퀀스를 가장 균형 있게 학습하여 최적의 경제적 효율성(Cost per True Positive = 328)을 제공함을 증명함[cite: 6].
  * 모델 출력 확률을 직접 신뢰하기보다 비즈니스 비용 행렬에 정렬된 사후 임계값(Thresholding)을 튜닝하는 것이 총비용 제어의 핵심 열쇠임을 통계적으로 실증함[cite: 6].

---

## 6. 핵심 5대 연구 종합 비교 매트릭스 (Synthesis Matrix)

| 구분 | Kharazian et al. (2025)[cite: 5] | Parton et al. (2024)[cite: 3] | Carpentier et al. (2024)[cite: 4] | Ferrisi et al. (2026)[cite: 2] | Mraihi et al. (2026)[cite: 6] |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **주요 방법론** | 원전 벤치마크 및 비대칭 비용 매트릭스 정의[cite: 5] | 가시성 그래프(VG) + Graph Isomorphism Network (GIN)[cite: 3] | tsfresh 피처 엔지니어링 + AFT 생존분석 XGBoost[cite: 4] | 마지막 관측치 추출 + LMM 결측 대체 + Random Forest[cite: 2] | 최근 30TS 윈도우 + Transformer/TCN + 비대칭 임계값 최적화[cite: 6] |
| **시계열 처리 방식** | 107개 다변량 원시 시계열[cite: 5] | 단일 변수(`171_0`)를 복잡계 그래프 네트워크로 변환[cite: 3] | 다중 슬라이딩 윈도우 기반 통계량 피처 추출 (Tabular화)[cite: 4] | 마지막 타임스텝 1개만 단면 추출 (시계열 제거)[cite: 2] | 3D 시계열 텐서 ($(30 \times 64)$) 직접 입력[cite: 6] |
| **목적 분류 체계** | 5-Class 다항 열화 단계[cite: 5] | 이진 분류 병합 (Class 0 vs Class 4)[cite: 3] | 5-Class 다항 / 연속형 회귀 / 시간 추정 생존분석[cite: 4] | 이진 분류 (정상 vs 고장)[cite: 2] | 이진 위험도 학습 후 $5 \times 5$ Fail-Safe 매핑[cite: 6] |
| **달성 성능 (Cost 지표)** | 기준점 제시[cite: 5] | Validation Cost: **40,109**[cite: 2, 3] | Validation 평균 비용: **~9.8 / 건**[cite: 4] | Validation Cost: **36,595** (Test: 42,976)[cite: 2] | Test Cost: **40,195** (평균 8.09 / 건)[cite: 6] |
| **핵심 한계점** | 기준 데이터 제공에 국한[cite: 5] | 다변량 관계 유실, 고장 등급 세분화 실패[cite: 3] | `Always predict 4` 규칙 대비 뚜렷한 변별력 부족[cite: 4] | 시계열 맥락 상실, 잔여 수명(RUL) 리드타임 파악 불가[cite: 2] | 100% 점검(TCN) 등의 과잉 정비 딜레마, 설명력 부족[cite: 6] |

---

## 7. NS-PdM & MH-DCC DNN 작성을 위한 핵심 차별화 전략

1. **시계열 동적성과 정적 사양의 분기 융합 (Dual-Channel):**
   * Ferrisi et al.처럼 시계열을 버리거나(Tabular화)[cite: 2], Parton et al.처럼 단일 센서만 그래프화하는 한계를 넘어[cite: 3], 1D-CNN 동적 채널(Within 잔차)과 Spec 임베딩 정적 채널(Between 베이스라인)을 결합하고 FiLM으로 상호작용($X \times \text{Spec}$)을 학습시킴.
2. **이산 시간 위험률(Discrete-Time Hazard) 기반 구조적 단조성 보장:**
   * Mraihi et al.과 Ferrisi et al.은 최적화를 위해 문제를 이진 분류로 축약하여 5등급 순서 정보를 포기함[cite: 2, 6].
   * 박사님의 다중시간지평(Multi-Horizon) 헤드는 4개 이산 위험률($q_1 \sim q_4$)의 생존 곱($p_k = 1 - \prod (1 - q_j)$)을 통해 **5단계 리드타임 서열을 보존하면서도 누적 고장확률의 구조적 단조성($0 \le p_6 \le p_{12} \le p_{24} \le p_{48} \le 1$)을 수학적으로 강제**함.
3. **심볼릭 온톨로지 가드레일을 통한 오탐지(False Positive) 원천 차단:**
   * 기존 선행 연구 전반(Mraihi의 TCN 4,826건 FP[cite: 6], Ferrisi의 RF[cite: 2])에서 나타난 '미탐지 페널티(500)를 피하려다 수천 건의 가짜 알람을 울리는 경보 피로도 현상'을, GEE/LMM으로 검증된 정상(BLUE) 95% 분위수 기반 규칙 엔진(Override)으로 완벽하게 제어하여 최소 비용을 달성함.
