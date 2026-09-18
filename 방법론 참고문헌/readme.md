# 첨부 논문 기반 산업용 예지보전(PdM) 핵심 연구 종합 분석 보고서

---

## 1. 핵심 연구 종합 비교 매트릭스 (4대 연구 프레임워크)

| 논문 분류 | 연구 (저자 및 연도) | 연구 대상 도메인 | 핵심 연구내용 | 핵심 함의점 및 시사점 |
| :--- | :--- | :--- | :--- | :--- |
| **벤치마크 및 데이터셋 원전** | **Kharazian et al. (2025)**<br>(*Nature Scientific Data*)[cite: 12] | Scania 트럭 Component X<br>(다변량 불규칙 시계열)[cite: 12] | • 실운행 33,641대 트럭(152만 Readout) 데이터 정제[cite: 12]<br>• 14개 센서(107개 시계열 변수) 및 8개 차량 사양 구축[cite: 12]<br>• 잔여수명(RUL) 기반 5단계 열화 레이블(Class 0~4) 정의[cite: 12]<br>• 공식 $5 \times 5$ 비대칭 비용 행렬(FN 500 vs FP 10) 수립[cite: 12] | • 점진적 마모, 통신 리셋, 불규칙 주기, 90.4% 우측검열을 보존한 표준 벤치마크 확립[cite: 12]<br>• 정확도 중심이 아닌 '경제적 총비용(Total Cost) 최소화' 평가 패러다임 제시[cite: 12] |
| **생존분석 vs RUL 회귀 방법론** | **Rahat et al. (2023)**<br>(*PHM AP 2023*)[cite: 13] | 상용차 터보차저(Volvo) 및 항공기 터보팬(C-MAPSS)[cite: 13] | • Run-to-failure 시계열의 생존분석 정식화($(X_i, e_i, y_i)$) 프레임워크 제안[cite: 13]<br>• 선형, 구간 선형, 생존분석(RF/GBT) 6개 모델 비교[cite: 13]<br>• 검열률(1%~66%) 변화에 따른 오차 추세 실증[cite: 13]<br>• C-index 편향 극복을 위해 중앙 생존시간($S(t)=0.5$) 기반 MAE 평가[cite: 13] | • 검열 표본을 배제하는 선형 회귀의 예측 편향 실증[cite: 13]<br>• 우측검열이 지배적인 산업 환경에서는 생존 함수 기반 접근이 RUL 회귀보다 예측 강건성 우위 입증[cite: 13] |
| **불확실성 정량화 및 능동학습** | **Kharazian et al. (2024)**<br>(*PMLR* Vol. 230, CoPAL)[cite: 14, 15] | Scania Component X 및 NASA C-MAPSS (RUL 회귀)[cite: 14, 15] | • 컨포멀 예측 구간 크기를 불확실성 지표로 활용하는 CoPAL 알고리즘 개발[cite: 14, 15]<br>• 차량 단위 교환성(Vehicle-based Exchangeability) 기반 데이터 누출 차단[cite: 14, 15]<br>• Mondrian CPS 및 불확실성 쿼리(most_uncertain 등) 평가[cite: 14, 15]<br>• Component X RUL RMSE 13.68단위(17%) 조기 개선[cite: 14, 15] | • 전수 점검이 불가능한 플릿 환경에서 수학적으로 보장된 신뢰 구간 기반 정비 우선순위화[cite: 14, 15]<br>• 점 예측의 불확실성을 가시화하여 안전 필수 예지보전의 신뢰도와 설명력 제고[cite: 14, 15] |
| **비용민감 시계열 딥러닝** | **Mraihi et al. (2026)**<br>(*PHM Europe 2026*)[cite: 5] | Scania Component X<br>(다변량 시계열 딥러닝)[cite: 5] | • 5등급을 Healthy vs At-Risk 이진 분류로 재정식화[cite: 5]<br>• 가중 교차 엔트로피(WBCE: $w_1=10, w_0=1$) 적용[cite: 5]<br>• 최근 30TS 윈도우 기반 CNN, Transformer, TCN 비교[cite: 5]<br>• 사후 비대칭 임계값($\theta^*$) 튜닝 후 Fail-safe(Class 4) 사영[cite: 5] | • Transformer 모델이 최저 총비용(40,195) 및 최적 경제성(Cost/TP=328) 달성[cite: 5]<br>• TCN의 과잉 정비(FP 4,826건, 100% 점검) 한계 확인 및 비대칭 임계값 튜닝의 중요성 실증[cite: 5] |

---

## 2. 논문별 심층 분석 및 상세 기술

### [연구 1] Kharazian et al. (2025)
* **연구 (저자 및 연도):** Zahra Kharazian, Tony Lindgren, Sindri Magnússon, Olof Steinert, Oskar Andersson Reyna (2025)[cite: 12]
* **논문 정보:** *SCANIA Component X dataset: a real-world multivariate time series dataset for predictive maintenance* (*Nature Scientific Data*, 12:493)[cite: 12]
* **연구 대상 도메인:** 대형 트럭(Heavy-Duty Trucks)의 비공개 엔진 핵심 부품인 Component X, 산업용 다변량 불규칙 시계열 예지보전 표준 벤치마크[cite: 12]
* **연구내용:**
  * **데이터 수집 및 구조:** Scania 실운행 트럭 33,641대(Train 23,550대, Validation 5,046대, Test 5,045대)에서 수집된 152만 건 이상의 다변량 시계열 Readout을 정제 및 공개함[cite: 12].
  * **특징 구성:** 14개 센서 속성(6개 다차원 히스토그램 97개 bins + 8개 단일 누적 카운터 = 총 107개 시계열 변수)과 8개 범주형 차량 정적 사양(`Spec_0`~`Spec_7`, 총 3,607개 고유 모델 조합)으로 구성됨[cite: 12].
  * **5단계 열화 라벨링:** 수리 시점($TTE$)을 기준으로 잔여 수명(RUL)에 따라 5개 등급(Class 0: $>48$, Class 1: $48\sim24$, Class 2: $24\sim12$, Class 3: $12\sim6$, Class 4: $6\sim0$ time step)을 부여함[cite: 12].
  * **비대칭 비용 행렬:** 고장 미탐지(False Negative, 최대 500)와 불필요한 점검(False Positive, 7~10) 간의 불균형을 반영한 공식 $5 \times 5$ 손실 평가 매트릭스를 정립함[cite: 12].
* **함의점:**
  * 점진적 마모 과정, 센서 리셋 노이즈, 불규칙 주기, 우측검열(Train 90.4% 비고장) 등 산업 현장의 복잡성을 담은 표준 벤치마크를 정립함[cite: 12].
  * 정확도 중심의 기존 머신러닝 평가를 탈피하여 비즈니스 총비용(Total Cost) 기반의 최적화 연구 방향을 제시함[cite: 12].

---

### [연구 2] Rahat et al. (2023)
* **연구 (저자 및 연도):** Mahmoud Rahat, Zahra Kharazian, Peyman Sheikholharam Mashhadi, Thorsteinn Rögnvaldsson, Shamik Choudhury (2023)[cite: 13]
* **논문 정보:** *Bridging the Gap: A Comparative Analysis of Regressive Remaining Useful Life Prediction and Survival Analysis Methods for Predictive Maintenance* (*PHM AP 2023*)[cite: 13]
* **연구 대상 도메인:** 상용차 엔진 터보차저(Volvo Trucks, 461대) 및 항공기 터보팬 엔진(NASA C-MAPSS FD001, 100개 엔진), RUL 회귀와 생존분석의 통합 비교[cite: 13]
* **연구내용:**
  * **통합 정식화 프레임워크:** Run-to-failure 시계열 데이터를 생존분석 데이터 형태($(X_i, e_i, y_i)$)로 변환하는 범용 파이프라인을 구축함[cite: 13].
  * **3가지 열화 모델 비교:** 선형 열화(검열 배제), 구간 선형 열화(조기 RUL을 연구 종료 시점으로 고정), 생존분석 모델(생존 곡선 $S(t)$ 도출 후 중앙 생존시간 $S(t)=0.5$ 교차점으로 RUL 추정)을 설계함[cite: 13].
  * **학습 알고리즘 적용:** Random Forest(RF/RSF) 및 Gradient Boosting(GBT)을 결합하여 총 6개 모델을 훈련함[cite: 13].
  * **검열률 민감도 분석:** 연구 기간(EOS)을 조정하여 검열 표본 비율(C-MAPSS 1%~66%, Volvo 0%~54%) 변화에 따른 오차 수렴 추세를 실증함[cite: 13].
* **함의점:**
  * 검열 표본을 무시하는 단순 선형 회귀는 검열률이 증가할수록 예측 오차가 급증하며, 검열 정보를 보존하는 구간 선형 및 생존분석 모델이 일관되게 우수한 성능(낮은 MAE)을 제공함을 증명함[cite: 13].
  * 우측검열이 지배적인 산업 설비 도메인에서는 생존 곡선 기반의 모델링이 회귀 기반 예측보다 우수한 강건성을 보임을 규명함[cite: 13].

---

### [연구 3] Kharazian et al. (2024)
* **연구 (저자 및 연도):** Zahra Kharazian, Tony Lindgren, Sindri Magnússon, Henrik Boström (2024)[cite: 14, 15]
* **논문 정보:** *CoPAL: Conformal Prediction for Active Learning with Application to Remaining Useful Life Estimation in Predictive Maintenance* (*PMLR*, Vol. 230, pp. 195–217)[cite: 14, 15]
* **연구 대상 도메인:** 회귀 문제에서의 불확실성 정량화(Uncertainty Quantification) 및 능동학습(Active Learning), Scania Component X 및 NASA C-MAPSS RUL 예측[cite: 14, 15]
* **연구내용:**
  * **CoPAL 알고리즘 설계:** 컨포멀 예측(Conformal Prediction)을 통해 산출된 예측 구간(Prediction Interval)의 길이를 불확실성 지표로 활용하여 가장 유의미한 표본을 선별하는 최초의 회귀 능동학습 프레임워크를 제안함[cite: 14, 15].
  * **차량 단위 교환성 확보:** 시계열 데이터의 비교환성(Non-exchangeability)과 데이터 누출(Leakage)을 방지하기 위해 차량 단위(Vehicle-based)로 데이터를 분할하고, 시계열 관측치들의 예측 구간을 차량 단위로 평균 집계함[cite: 14, 15].
  * **컨포멀 예측기 및 샘플링 정책 비교:** Mondrian Regressor, 표준 Mondrian CPS, 정규화 Mondrian CPS를 적용하고, `most_uncertain`, `roulette`, `random` 등의 표본 추출 정책을 평가함[cite: 14, 15].
  * **성능 검증:** Component X에서 Random Forest + 정규화 Mondrian CPS 적용 시 초기 대비 RMSE가 13.68단위(17%) 감소하였고, C-MAPSS에서는 21.38단위(35%) 감소하여 무작위 추출 대비 빠른 성능 개선을 달성함[cite: 14, 15].
* **함의점:**
  * 전체 플릿을 전수 검사할 수 없는 산업 환경에서, 수학적으로 보장된 신뢰 구간을 기반으로 우선 점검 차량을 선별하는 비용 효율적 능동학습 체계를 입증함[cite: 14, 15].
  * 점 예측의 불확실성을 신뢰 구간으로 가시화하여 안전 필수 예지보전 시스템의 설명 가능성과 신뢰도를 제고함[cite: 14, 15].

---

### [연구 4] Mraihi et al. (2026)
* **연구 (저자 및 연도):** Abdelhakim Mraihi, Valeriu Dimidov, Raoof Doorshi, Reza Khoshkangini (2026)[cite: 5]
* **논문 정보:** *Cost-Sensitive Deep Learning for Scania Component X: Minimising Operational Cost via Asymmetric Threshold Optimisation* (*PHM Europe 2026*, Paper ID: 4963)[cite: 5]
* **연구 대상 도메인:** Scania Component X 다변량 시계열 데이터, 비용 민감 시계열 딥러닝(CNN, Transformer, TCN) 및 비대칭 결정 임계값 최적화[cite: 5]
* **연구내용:**
  * **이진 위험도 재정식화:** 5등급 다항 분류를 Healthy(0) vs At-Risk(1~4)로 축약하고, 가중 이진 교차 엔트로피(WBCE: $w_1=10, w_0=1$) 손실 함수를 적용함[cite: 5].
  * **시계열 텐서화:** 차량별 최근 30개 타임스텝 윈도우(Zero-padding 적용)와 분산 기반 64개 센서 변수를 결합하여 $(30 \times 64)$ 크기의 입력 시퀀스를 구성함[cite: 5].
  * **비대칭 사후 임계값($\theta^*$) 최적화:** 이론적 베이즈 임계값($\theta_{\text{Bayes}} \approx 0.02$) 부근을 탐색하여 검증 세트의 공식 $5 \times 5$ 비용을 최소화하는 최적 경계($\theta^*$)를 도출하고, 이를 Fail-safe 규칙에 따라 Class 4로 사영(Mapping)함[cite: 5].
  * **아키텍처별 실증 비교:**
    * Transformer: 테스트 세트 최저 총비용 40,195점(차량당 8.09점, FNR 13.4%, 유지보수율 68.0%)을 달성하여 최적의 경제성을 입증함[cite: 5].
    * TCN: 미탐지 0건(FN=0)을 달성했으나 전체 차량 점검(유지보수율 100%, FP 4,826건)으로 과잉 정비 붕괴를 초래함[cite: 5].
* **함의점:**
  * 비대칭 비용 환경에서는 손실 함수 가중치뿐만 아니라 비즈니스 비용 행렬에 정렬된 사후 임계값 튜닝이 최종 운용 비용 절감의 핵심임을 실증함[cite: 5].
  * 반면, 최적화를 위해 문제를 이진 분류로 축약함에 따라 5단계 열화 순서 및 정비 리드타임 정보가 유실되었으며, 미탐지 회피 과정에서 수천 건의 오탐지(FP)가 발생하는 경보 피로도(Alarm Fatigue)의 한계를 노출함[cite: 5].

---

## 3. MH-DCC-DNN & NS-PdM 접점 및 차별화 전략

1. **우측검열 자료 보존과 구조적 단조성 강제:**
   * Rahat et al. (2023)의 생존분석 우위 실증을 계승하여[cite: 13], 90.4%에 달하는 검열 표본을 버리지 않고 다중시간지평(Multi-Horizon) 이산 위험률($q_1 \sim q_4$) 학습에 온전히 활용함[cite: 12].
   * Mraihi et al. (2026)이 포기했던 **5단계 정비 리드타임 서열을 보존하면서, 누적 생존 곱($p_k = 1 - \prod (1 - q_j)$)을 통해 누적 고장확률의 구조적 단조성($0 \le p_6 \le p_{12} \le p_{24} \le p_{48} \le 1$)을 수학적으로 강제**함[cite: 5].
2. **비모수 ECDF 정규 점수(Normal Score) 변환 기반 신경망 최적화:**
   * Kharazian et al. (2024)의 컨포멀 신뢰구간 및 통계적 난이도 추정 논리를 수용하여[cite: 14, 15], 센서 데이터의 비정규성과 이상치 편향을 ECDF 기반 역정규 변환($z = \Phi^{-1}(u)$)으로 표준화하여 딥러닝 역전파의 안정성을 극대화함.
3. **심볼릭 온톨로지 가드레일을 통한 오탐지(FP) 원천 차단:**
   * Mraihi et al. (2026)의 TCN에서 나타난 4,826건의 과잉 점검(유지보수율 100%) 및 경보 피로도 문제를[cite: 5], 통계 게이트(LMM/GEE)로 검증된 정상(Grade 1) ECDF 95% 분위수 기반 지식 규칙(Override)으로 완벽히 차단하여 실질적 최소 비용을 달성함.
