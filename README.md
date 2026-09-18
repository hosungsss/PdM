# Neuro-Symbolic Predictive Maintenance (NS-PdM) & MH-DCC-DNN Architecture

> **A New Paradigm for Neuro-Symbolic Predictive Maintenance: Spec-Conditioned Normal-Reference Quantile Learning with a Multi-Horizon Deep Calibrated Classifier**

본 저장소는 대형 상용차의 실제 운용 환경에서 발생하는 극단적인 클래스 불균형, 비정규 센서 노이즈, 우측검열(Right Censoring), 그리고 비대칭 비즈니스 정비 비용(미탐지 페널티 500 vs 오탐지 페널티 10)을 해결하기 위한 **뉴로-심볼릭 예지보전(NS-PdM) 파이프라인 및 다중시간지평 심층 보정 신경망(MH-DCC-DNN)** 연구 자산을 총괄합니다.

---

## 1. 프로젝트 개요 (Overview)

산업용 대형 트럭 플릿 환경에서 센서 기반 고장 예측 모델은 정상 데이터가 90% 이상을 차지하는 극단적 불균형과 심한 비정규 편포(Fat-tail, Skewness)로 인해 일반적인 가우시안 가정 기반 머신러닝이나 순수 딥러닝(Pure Deep Learning) 적용 시 심각한 한계에 직면합니다[cite: 16, 17, 18]. 특히 미탐지(False Negative) 비용이 오탐지(False Positive)보다 압도적으로 높은 비대칭 비용 구조 하에서는 기존 딥러닝 모델들이 미탐지를 피하기 위해 전체 차량에 무차별 정비 경보를 울리는 '과잉 정비(Over-maintenance, 100% 점검)' 딜레마에 빠지게 됩니다.

본 연구는 이러한 산업적 병목을 돌파하기 위해 다음의 4대 핵심 축을 통합한 재현 가능한 엔드투엔드(End-to-End) 아키텍처를 제공합니다:
1. **비모수적 분위수 학습 (Spec-Conditioned ECDF Normal-Reference):** 가우시안 정규분포 가정을 폐기하고, Train 세트 정상(Grade 1) 데이터 기반의 경험적 누적분포함수(ECDF) 분위수 사영을 통해 센서 피처를 표준정규 점수(Normal Score, $z = \Phi^{-1}(u)$)로 변환[cite: 16, 17].
2. **Within/Between 계층적 분해:** 급내상관도(ICC $\approx 0.42$) 검증을 통해 개체 간 고유 편차(Between)와 차량 내부의 순수 시간 열화 궤적(Within)을 분리하여 심슨 역설(Simpson's Reversal) 원천 차단[cite: 16, 17].
3. **다중시간지평 이산 위험률(Discrete-Time Hazard) 헤드:** 5단계 열화 등급을 단순 이진화하지 않고, 구간 생존 곱($p_k = 1 - \prod (1 - q_j)$)을 통해 누적 고장확률의 구조적 단조성($0 \le p_6 \le p_{12} \le p_{24} \le p_{48} \le 1$)을 수학적으로 강제[cite: 16, 17].
4. **심볼릭 온톨로지 가드레일 (Symbolic Guardrails):** GEE/LMM 통계 게이트를 통과한 정상 베이스라인 상위 95% 분위수 규칙을 적용하여, 신경망의 공격적 경보 중 허위 알람(False Positive)을 강제 기각(Override)함으로써 총 정비 비용 극소화[cite: 16, 17].

---

## 2. 데이터셋 아키텍처 및 벤치마크 규격 (Dataset Specifications)

본 연구 파이프라인은 Scania AB에서 공식 공개한 두 가지 산업용 표준 벤치마크 데이터를 포괄하여 검증을 수행합니다[cite: 16, 18].
Data Repository Structure
├── Scania Component X (Primary Benchmark: Multivariate Irregular Time Series)
│   ├── train_operational_readouts.csv  (1,122,452 rows, 23,550 unique vehicles)
│   ├── train_specifications.csv        (23,550 vehicles, 8 categorical specs: Spec_0~Spec_7)
│   ├── train_tte.csv                   (23,550 vehicles, length_of_study, in_study_repair)
│   ├── val_operational_readouts.csv    (196,227 rows, 5,046 vehicles, randomly censored)
│   ├── val_specifications.csv          (5,046 vehicles)
│   ├── val_labels.csv                  (5,046 vehicles, 5-class degradation labels)
│   ├── test_operational_readouts.csv   (198,140 rows, 5,045 vehicles)
│   ├── test_specifications.csv         (5,045 vehicles)
│   └── test_labels.csv                 (5,045 vehicles, gold-standard evaluation)
└── Scania APS (Comparative Domain: Static High-Dimensional Tabular Sensors)
├── aps_failure_training_set.csv    (60,000 instances, 170 numerical sensors)
└── aps_failure_test_set.csv        (16,000 instances, 170 numerical sensors)
