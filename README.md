# Permutation Importance 기반 EEG 전극 중요도 분석

## 개요

SOGNN(Self-Organized Graph Neural Network) 모델에서 **Permutation Importance(PI)**를 활용하여 62개 EEG 전극의 감정 분류 기여도를 정량화하고, **점진적 전극 제거 실험**을 통해 결과를 검증하는 실험이다.

## 연구 동기

감정 인식에 핵심적인 뇌 영역(전극)을 파악하는 것은 다음과 같은 의미를 가진다:
- 모델 해석 가능성 검증
- 신경과학 문헌(전두엽/측두엽의 감정 처리 역할)과의 정합성 확인
- 실용적 BCI 시스템을 위한 전극 수 최소화 근거 마련
- Feature selection 전략에 대한 실증적 근거 제공

---

## 방법론

### 1. Permutation Importance 계산

62개 전극 각각에 대해:
1. 모델의 기준 성능(Accuracy, AUC, F1) 측정
2. 해당 전극의 feature 값을 전체 샘플에 걸쳐 무작위 치환(permutation)
3. 모델 성능 재측정
4. **중요도 = 기준 성능 - 치환 후 성능**
5. 10회 반복 후 평균값 사용 (안정성 확보)

**해석:** 점수가 높을수록 해당 전극이 분류에 더 중요함을 의미

**참고 문헌:** Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.

---

### 2. 점진적 전극 제거 실험

PI 결과를 검증하기 위해 3가지 제거 전략을 비교한다:

#### 전략 A: 중요도 낮은 전극부터 제거 (예상: 완만한 성능 하락)
- PI 점수 오름차순으로 전극 제거
- 가설: 초기에는 성능이 유지되다가 점차 하락

#### 전략 B: 무작위 제거 (비교 기준선)
- 전극을 무작위 순서로 제거
- 가설: 선형적 또는 예측 불가능한 성능 하락
- 10개의 random seed로 평균 산출

#### 전략 C: 중요도 높은 전극부터 제거 (예상: 급격한 성능 하락)
- PI 점수 내림차순으로 전극 제거
- 가설: 성능이 급격히 하락
- PI가 핵심 전극을 올바르게 식별했는지 검증

---

### 3. 평가 지표

전극 제거 단계별(1, 2, 5, 10, 15, 20, 30, 40, 50, 60개 제거 시)로 다음을 측정:
- **Accuracy**: 전체 분류 정확도
- **AUC (macro)**: ROC 곡선 아래 면적
- **F1 Score (macro)**: 정밀도와 재현율의 조화 평균

---

## 예상 결과

### 가설

```
성능 vs. 제거된 전극 수

높음 ┤
     │ ████████╲          ← 낮은 중요도부터 제거 (완만한 하락)
     │ ─ ─ ─ ─ ╲╲         ← 무작위 제거 (중간 하락)
     │          ╲╲╲╲
     │            ╲╲╲
낮음 │              ████  ← 높은 중요도부터 제거 (급격한 하락)
     └─────────────────────
     0   10   20   30   62 제거된 전극 수
```

**검증 기준:**
- "높은 중요도부터 제거"가 가장 빠르게 하락 → PI가 핵심 전극을 정확히 식별
- "낮은 중요도부터 제거"가 가장 오래 유지 → 낮은 PI 전극이 실제로 불필요
- "무작위 제거"가 중간에 위치 → PI 순위가 유의미

---

## 신경과학적 검증

상위 전극을 기존 감정 처리 관련 뇌 영역과 비교:

| 뇌 영역 | 예상 전극 | 감정에서의 역할 |
|---------|-----------|----------------|
| **전두엽 (Frontal)** | Fp1, Fp2, F3, F4, F7, F8 | 감정 조절, 정서가(valence) 처리 |
| **측두엽 (Temporal)** | T7, T8, TP7, TP8 | 감정적 기억, 정서 인식 |
| **두정엽 (Parietal)** | P3, P4, Pz | 감정 자극에 대한 주의 |

**검증:** 상위 PI 전극이 전두엽/측두엽에 집중 → 신경과학적으로 타당

---

## 실행 단계

### 1단계: 모델 학습 및 저장
```bash
python main.py
```

### 2단계: Permutation Importance 계산
```bash
python compute_pi.py --model models/SOGNN_fold0_best.pth \
                     --data processed/ \
                     --output results/pi_scores.csv
```

### 3단계: 점진적 전극 제거 실험
```bash
python progressive_removal.py --pi_scores results/pi_scores.csv \
                              --strategies least,random,most \
                              --output results/removal_curves.csv
```

### 4단계: 시각화
```bash
python plot_results.py --data results/removal_curves.csv \
                       --output figures/
```

**출력 파일:**
- `figures/removal_curves.png` - 성능 vs. 제거 전극 수 그래프
- `figures/pi_topographic.png` - 뇌 지형도에 PI 점수 매핑
- `figures/top_electrodes_bar.png` - 상위 10개 전극 중요도 막대 그래프

---

## 결과 해석 가이드

### PI 유효성의 강한 근거:
- "높은 중요도부터 제거" 곡선이 무작위보다 유의하게 빠르게 하락 (p < 0.05)
- 상위 10개 전극에 전두엽/측두엽 영역 포함
- PI 점수와 실제 제거 영향 간 Spearman 상관계수 > 0.7

### 추가 해석:
- **"낮은 중요도" 곡선의 plateau** → 최소 필수 전극 세트 식별 가능
- **모든 곡선의 초기 급락** → 소수 핵심 전극에 중요도 집중
- **후반부 plateau** → 나머지 전극은 중복 정보 제공

---

## 통계적 검증

### 유의성 검정
- **Bootstrap CI**: PI 점수의 95% 신뢰구간 (1000회 리샘플링)
- **Permutation Test**: 귀무가설 = 해당 전극이 성능에 무관 (p < 0.05)
- **다중비교 보정**: Bonferroni 보정 (p < 0.05/62)

### 강건성 확인
- 15개 LOSO fold 전체에서 PI 교차 검증
- 중요도 점수의 평균 및 표준편차 보고
- Fold 간 일관되게 상위 10위에 포함되는 전극 식별

---

## 파일 구조

```
SOGNN-main/
├── compute_pi.py              # Permutation Importance 계산
├── progressive_removal.py     # 점진적 전극 제거 실험
├── plot_results.py            # 결과 시각화
├── results/
│   ├── pi_scores.csv          # 62개 전극별 PI 점수
│   ├── removal_curves.csv     # 제거 단계별 성능 기록
│   └── top_electrodes.txt     # 중요도 순위 전극 목록
└── figures/
    ├── removal_curves.png     # 메인 결과 그래프
    ├── pi_topographic.png     # 뇌 지형도 시각화
    └── correlation_heatmap.png # PI vs. 제거 영향 상관관계
```

---

## 접근법의 장점

1. **모델 무관 (Model-Agnostic)**: 어떤 신경망 구조에도 적용 가능
2. **해석 용이**: 전극과 성능 간의 직접적 관계 파악
3. **교차 검증**: 3가지 독립적 전략으로 결과 검증
4. **신경과학적 근거**: fMRI/병변 연구 결과와 비교 가능
5. **실용성**: 실제 BCI 시스템의 최소 전극 세트 도출

---

## 한계점

- **Permutation의 분포 외 문제**: 치환된 데이터가 현실적인 노이즈를 반영하지 않을 수 있음
- **독립성 가정**: 전극 간 상호작용(시너지/중복)을 고려하지 않음
- **재학습 미수행**: 전극 제거 시 사전 학습된 모델 사용 (재최적화 없음)

**완화 방안**: 3가지 독립적 전략 + 통계적 검정을 통한 결과 검증

---

## 참고 문헌

1. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
2. Altmann, A., et al. (2010). Permutation importance: a corrected feature importance measure. *Bioinformatics*, 26(10), 1340-1347.
3. [arXiv:2311.17204] Optimal EEG electrode set for emotion recognition (2023)
4. Davidson, R. J. (2004). What does the prefrontal cortex "do" in affect? *Perspectives on Psychological Science*, 1(3), 219-234.

---

**Last Updated**: 2026-02-10
