# 산업용 예지보전(PdM) 핵심 문헌 벤치마크 및 방법론 개요

본 문서는 대형 상용차의 실제 운행 데이터에서 나타나는 극단적 클래스 불균형, 비정규 센서 노이즈, 우측검열(Right Censoring), 그리고 비대칭 비즈니스 정비 비용(미탐지 500 vs 오탐지 10) 문제를 해결하기 위해 선행 연구 7편의 핵심 기법과 한계점을 직관적으로 비교·정리한 기술 명세입니다[cite: 19, 20, 21].

---

## 1. 선행 연구 핵심 비교 매트릭스 (At a Glance)

| 연구 (저자 및 연도) | 대상 데이터 | 핵심 방법론 | 시계열/데이터 처리 | 주요 성과 | 핵심 한계점 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Kharazian et al. (2025)**[cite: 4, 12] | Scania Component X | 다변량 시계열 표준 벤치마크 및 $5 \times 5$ 비대칭 비용 행렬 정의[cite: 4, 12] | 107개 시계열 센서 + 8개 사양(Spec)[cite: 4, 12] | 산업 표준 데이터 및 비용 프로토콜 확립[cite: 4, 12] | 기준선 제공에 국한, 예측 모델 부재[cite: 4, 12] |
| **Parton et al. (2024)**[cite: 2, 19] | Scania Component X | 경로 시그니처 + 가시성 그래프(VG) + GIN[cite: 2, 19] | 단일 변수(`171_0`)만 위상수학 그래프 변환[cite: 2, 19] | Val Cost: **40,109**[cite: 1, 19] | 다변량 센서 상관성 유실, 고장 단계 이진화[cite: 2, 19] |
| **Carpentier et al. (2024)**[cite: 3, 19] | Scania Component X | `tsfresh`(704개) + 사양 군집화 + 생존 AFT[cite: 3, 19] | 다중 슬라이딩 윈도우 기반 통계 피처 추출[cite: 3, 19] | Val 평균 비용: **~9.8 / 건**[cite: 3, 19] | '무조건 고장 경보' 베이스라인 대비 변별력 한계[cite: 3, 19] |
| **Ferrisi et al. (2026)**[cite: 1, 19] | Scania Component X | LMM 결측 대체 + SMOTE + Random Forest[cite: 1, 19] | 차량별 마지막 1개 관측치만 단면 추출[cite: 1, 19] | Val Cost: **36,595** (Test: 42,976)[cite: 1, 19] | 시계열 궤적 상실, 5단계 리드타임 관리 불가[cite: 1, 19] |
| **Mraihi et al. (2026)**[cite: 5, 19] | Scania Component X | 최근 30TS + 시계열 Transformer + $\theta^*$ 튜닝[cite: 5, 19] | $(30 \times 64)$ 3D 시계열 텐서 직접 입력[cite: 5, 19] | Test Cost: **40,195** (평균 8.09 / 건)[cite: 5, 19] | 5단계 순서 유실, TCN 과잉 정비(FP 4,826건)[cite: 5, 19] |
| **Rahat et al. (2023)**[cite: 13, 20] | Volvo 터보차저 + C-MAPSS | RUL 회귀 vs 생존분석 비교 프레임워크[cite: 13, 20] | Run-to-failure $\rightarrow$ 생존 데이터($(X, e, y)$) 변환[cite: 13, 20] | 생존 곡선 $S(t)$ 외삽 및 중앙 생존시간 도출[cite: 13, 20] | 5등급 이산 리드타임 분류 미고려[cite: 13, 20] |
| **Kharazian et al. (2024)**[cite: 14, 20] | Scania Component X + C-MAPSS | 컨포멀 예측(CPS) 기반 능동학습(CoPAL)[cite: 14, 20] | 차량 단위 교환성 기반 불확실성 구간 추정[cite: 14, 20] | RUL RMSE 17%(실차) / 35%(합성) 조기 개선[cite: 14, 20] | 비고장 검열 표본 배제 (수명 종료 표본 위주)[cite: 14, 20] |
| **Taghandiki et al. (2023)**[cite: 10, 21] | Scania APS | 3중 피처선택(IG, Corr, SVM) + 4-Layer DNN[cite: 10, 21] | 170개 변수 중 21개 변수 평균값 일괄 대체[cite: 10, 21] | Accuracy: **98.66%**[cite: 10, 21] | 비용 미반영으로 FN 137건 $\rightarrow$ **\$69,270 손실**[cite: 10, 21] |
| **Beikmohammadi et al. (2026)**[cite: 11, 21] | Scania APS + SECOM | SVM-SMOTE + Repeated ENN + Tabular Transformer[cite: 11, 21] | 베이지안 능형회귀 대체 + 하이브리드 리샘플링[cite: 11, 21] | Test Cost: **\$3,440** (FN 6.8건, FP 4건)[cite: 11, 21] | 정적 스냅샷에 국한, 시계열 및 검열 미반영[cite: 11, 21] |

---

## 2. 계통별 핵심 발견 및 한계점 (Key Insights & Gaps)

**1. 정확도(Accuracy) 최적화의 함정**
* Taghandiki et al. (2023)은 98.66%의 높은 정확도를 달성했으나 비대칭 비용($\text{FN}=500$)을 고려하지 않아 137건의 미탐지가 발생했고, \$69,270라는 막대한 손실을 초래했습니다[cite: 10, 21].
* 반면 Beikmohammadi et al. (2026)은 Focal Loss($\alpha=0.95, \gamma=1.5$)를 비용 대리 손실함수로 적용하여 미탐지를 6.8건으로 억제하며 총비용을 \$3,440로 낮췄습니다[cite: 11, 21].

**2. 시계열 궤적 유실 vs 단일 센서 국한**
* Ferrisi et al. (2026)은 결측치를 선형 혼합 모형(LMM)으로 정교하게 대체했으나 시계열을 단 1개의 마지막 관측치로 축약하여 동적 열화 궤적을 잃었습니다[cite: 1, 19].
* Parton et al. (2024)은 시계열 시그니처와 가시성 그래프(VG)를 도입했으나 `171_0` 단일 센서만 변환하여 다변량 센서 간의 교차 상관성을 포착하지 못했습니다[cite: 2, 19].

**3. 비대칭 임계값과 과잉 정비(Alarm Fatigue) 딜레마**
* Mraihi et al. (2026)은 Transformer로 최저 비용(40,195)을 기록했으나, TCN의 경우 미탐지를 0건으로 막으려다 전체 차량을 점검(유지보수율 100%, FP 4,826건)하는 과잉 정비 붕괴를 보였습니다[cite: 5, 19].
* Carpentier et al. (2024) 역시 극심한 비대칭 환경에서는 '무조건 고장 경보를 울리는 단순 규칙(`Always predict 4`)'이 어설픈 모델보다 비용이 낮게 측정되는 베이스라인 역설을 규명했습니다[cite: 3, 19].

**4. 우측검열(Right Censoring)과 다중 리드타임 정보 보존**
* Rahat et al. (2023)은 검열 표본을 버리는 회귀 모형이 심각한 편향을 유발하며, 생존 곡선 $S(t)$ 기반 모델이 우수함을 실증했습니다[cite: 13, 20].
* 그러나 대다수 선행 연구(Parton, Ferrisi, Mraihi)는 최적화 편의를 위해 5개 열화 단계를 정상 vs 고장의 이진 분류로 축약하여 단계별 정비 리드타임 정보를 유실했습니다[cite: 1, 2, 5, 19].

* **심볼릭 온톨로지 가드레일 (False Positive 기각):** 딥러닝이 미탐지를 피하려다 발생시키는 수천 건의 가짜 경보를, 통계 게이트(LMM/GEE)로 검증된 정상(Grade 1) ECDF 95% 분위수 규칙을 통해 기각(Override)하여 실질적 최소 비용을 달성합니다[cite: 19, 20, 21].
---
