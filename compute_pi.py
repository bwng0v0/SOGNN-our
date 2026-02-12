"""
Permutation Importance (PI) for SOGNN EEG electrode importance analysis.

==========================================================================
개요 (Overview)
==========================================================================
62개 EEG 전극 각각의 감정 인식 기여도를 정량화합니다.
특정 전극의 feature를 샘플 간에 무작위 셔플(permute)하여
해당 전극의 정보를 파괴한 뒤, 성능 하락폭(= PI)을 측정합니다.

  PI = baseline_score - permuted_score
  PI > 0 : 해당 전극이 성능에 기여 (클수록 중요)
  PI ~ 0 : 해당 전극이 거의 기여하지 않음
  PI < 0 : 해당 전극이 오히려 노이즈 역할 (드묾)

==========================================================================
셔플 방식 (Permutation Strategy)
==========================================================================
- 셔플 단위: 전극(electrode) 단위
  → 전극 i의 1325차원 feature (5밴드 x 265) 전체를 통째로 교환
- 셔플 범위: 같은 LOSO fold의 테스트셋 내부 (= 단일 피험자의 72개 트라이얼)
  → 같은 전극 번호끼리만 교환 (전극1 ↔ 전극1, 전극13과 섞이지 않음)
  → 세션/트라이얼/감정 라벨 구분 없이 무작위 교환
- 반복: 셔플의 랜덤성을 보정하기 위해 n_reps회 반복 후 평균

==========================================================================
실행 예시 (Usage)
==========================================================================
    python compute_pi.py                                       # 전체 실행 (15 folds x 62 electrodes x 10 reps)
    python compute_pi.py --folds 0 --n_reps 2                  # 빠른 테스트 (1 fold, 2 reps)
    python compute_pi.py --folds 0 1 2 --n_reps 5              # 선택 fold
    python compute_pi.py --resume ./result/pi_raw_scores.csv   # 중간 결과부터 재개

==========================================================================
출력 파일 (Outputs)
==========================================================================
    ./result/pi_fold{i}_raw.csv   : fold별 개별 결과 (fold x 62 electrodes x n_reps 행)
    ./result/pi_raw_scores.csv    : 전체 누적 raw 결과
    ./result/pi_summary.csv       : 전극별 집계 요약 (62행, 평균/표준편차/순위)
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

import Net
from Net import SOGNN
from datapipe import get_dataset

# ── 상수 정의 ────────────────────────────────────────────────
SUBJECTS = 15          # LOSO 피험자 수 (= fold 수)
CLASSES = 4            # 감정 클래스 수 (neutral, sad, fear, happy)
N_ELECTRODES = 62      # EEG 전극 수
BATCH_SIZE = 16        # 평가 시 배치 크기 (학습과 동일하게 설정)


# ═════════════════════════════════════════════════════════════
# 1. 모델 로드
# ═════════════════════════════════════════════════════════════
def load_fold_model(fold, device):
    """LOSO fold별 저장된 best 모델을 로드하여 eval 모드로 반환.

    Args:
        fold: LOSO fold 번호 (0~14). 해당 피험자를 테스트로 남긴 모델.
        device: 'cuda' 또는 'cpu'

    Returns:
        model: eval 모드의 SOGNN 모델
    """
    path = f'./models/SOGNN_fold{fold}_best.pth'
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model = SOGNN().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Dropout 비활성화, BatchNorm 고정 → 결정론적 평가
    return model


# ═════════════════════════════════════════════════════════════
# 2. 테스트 데이터 수집
# ═════════════════════════════════════════════════════════════
def collect_test_data(test_dataset):
    """PyG InMemoryDataset에서 전체 샘플을 하나의 텐서로 합침.

    테스트셋의 모든 Data 객체를 순회하면서 x, y를 쌓아
    전극별 셔플이 가능한 형태로 만듭니다.

    Args:
        test_dataset: EmotionDataset (테스트용)

    Returns:
        x_all: (N, 62, 1325) 텐서 — N개 샘플, 62개 전극, 1325차원 feature
               1325 = 5 frequency bands x 265 time features (DE features flatten)
        y_all: (N, 4) 텐서 — one-hot 인코딩된 감정 라벨
    """
    x_list = []
    y_list = []
    for i in range(len(test_dataset)):
        data = test_dataset[i]
        x_list.append(data.x.unsqueeze(0))   # (62, 1325) → (1, 62, 1325)
        y_list.append(data.y.unsqueeze(0))    # (4,) → (1, 4)

    x_all = torch.cat(x_list, dim=0)  # (N, 62, 1325)
    y_all = torch.cat(y_list, dim=0)  # (N, 4)
    return x_all, y_all


# ═════════════════════════════════════════════════════════════
# 3. DataLoader 생성
# ═════════════════════════════════════════════════════════════
def make_dataloader(x_all, y_all):
    """텐서를 PyG Data 객체 리스트로 변환하여 DataLoader 생성.

    SOGNN의 forward()는 PyG의 to_dense_batch()를 사용하므로,
    반드시 PyG DataLoader를 통해 batch 텐서가 생성되어야 합니다.

    Note: edge_index는 설정하지 않음 — SOGC 모듈이 자체적으로
          adjacency matrix를 학습하여 구성하므로 불필요.

    Args:
        x_all: (N, 62, 1325) 텐서
        y_all: (N, 4) 텐서

    Returns:
        DataLoader (batch_size=16)
    """
    data_list = []
    for i in range(x_all.shape[0]):
        # Data.x = (62, 1325): 62개 노드(전극), 1325차원 feature
        # Data.y = (4,): one-hot 라벨
        data_list.append(Data(x=x_all[i], y=y_all[i]))
    return DataLoader(data_list, batch_size=BATCH_SIZE)


# ═════════════════════════════════════════════════════════════
# 4. 모델 평가
# ═════════════════════════════════════════════════════════════
def evaluate_from_loader(model, loader, device):
    """DataLoader로부터 모델을 평가하여 acc, auc, f1을 반환.

    main.py의 evaluate() 함수와 동일한 방식으로 평가합니다:
    - model의 두 번째 출력(softmax 확률)을 사용
    - macro 평균으로 AUC, F1 계산

    Args:
        model: eval 모드의 SOGNN
        loader: PyG DataLoader
        device: 'cuda' 또는 'cpu'

    Returns:
        dict with keys: 'acc', 'auc', 'f1'
    """
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for data in loader:
            # 라벨을 device로 보내기 전에 CPU에서 추출 (numpy 변환 용이)
            label = data.y.view(-1, CLASSES)  # (batch, 4)

            data = data.to(device)
            # model 출력: (raw_logits, softmax_predictions)
            # PI에서는 softmax 확률(pred)만 사용
            _, pred = model(data.x, data.edge_index, data.batch)

            pred = pred.detach().cpu().numpy()
            pred = np.squeeze(pred)
            # 마지막 배치가 1개 샘플일 때 squeeze로 1차원이 되는 것 방지
            if pred.ndim == 1:
                pred = pred.reshape(1, -1)

            predictions.append(pred)
            labels.append(label.numpy())

    predictions = np.vstack(predictions)  # (N, 4) — softmax 확률
    labels = np.vstack(labels)            # (N, 4) — one-hot 라벨

    # AUC: one-hot 라벨 vs softmax 확률 (macro 평균)
    try:
        auc = roc_auc_score(labels, predictions, average='macro')
    except ValueError:
        # 특정 클래스가 테스트셋에 없으면 AUC 계산 불가
        auc = float('nan')

    # F1, Accuracy: argmax로 클래스 인덱스 변환 후 비교
    f1 = f1_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1), average='macro')
    acc = accuracy_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1))

    return {'acc': acc, 'auc': auc, 'f1': f1}


# ═════════════════════════════════════════════════════════════
# 5. 전극 셔플 (PI의 핵심 연산)
# ═════════════════════════════════════════════════════════════
def permute_electrode(x_all, electrode_idx, rng):
    """특정 전극의 feature를 샘플 간에 무작위 셔플.

    예시 (electrode_idx=0, N=4인 경우):
        원본:  샘플A의 전극0, 샘플B의 전극0, 샘플C의 전극0, 샘플D의 전극0
        셔플:  샘플C의 전극0, 샘플A의 전극0, 샘플D의 전극0, 샘플B의 전극0

    이렇게 하면 해당 전극의 feature가 라벨과의 상관관계를 잃게 되어,
    모델이 그 전극에 의존하던 만큼 성능이 하락합니다.

    Note:
    - 같은 전극 번호끼리만 교환 (전극0 ↔ 전극0)
    - 다른 전극 번호와는 절대 섞이지 않음
    - 나머지 61개 전극은 원본 유지

    Args:
        x_all: (N, 62, 1325) 전체 테스트 데이터
        electrode_idx: 셔플할 전극 인덱스 (0~61)
        rng: numpy random generator (재현성을 위한 시드 관리)

    Returns:
        x_perm: (N, 62, 1325) — electrode_idx 전극만 셔플된 복사본
    """
    x_perm = x_all.clone()       # 원본 보존을 위해 복사
    n = x_perm.shape[0]          # 샘플 수 (= 테스트 피험자의 트라이얼 수, 보통 72)
    perm_idx = rng.permutation(n)  # [0, 1, ..., n-1]을 무작위 셔플한 인덱스

    # 해당 전극의 1325차원 feature 벡터를 셔플된 순서로 재배치
    # x_all[perm_idx, electrode_idx, :] : 셔플된 순서로 해당 전극의 feature를 가져옴
    x_perm[:, electrode_idx, :] = x_all[perm_idx, electrode_idx, :]
    return x_perm


# ═════════════════════════════════════════════════════════════
# 6. 단일 Fold PI 계산
# ═════════════════════════════════════════════════════════════
def compute_pi_single_fold(fold, device, n_reps, base_seed):
    """하나의 LOSO fold에 대해 62개 전극 전체의 PI를 계산.

    절차:
        1) fold별 best 모델 로드
        2) 테스트 데이터 수집 (해당 피험자 1명의 72개 트라이얼)
        3) 원본 데이터로 baseline 성능 측정
        4) 각 전극(0~61)에 대해:
            - n_reps회 반복하여 셔플 → 평가 → PI 기록
            - 매 반복마다 다른 시드로 셔플 (랜덤성 보정)

    Args:
        fold: LOSO fold 번호 (0~14)
        device: 연산 장치
        n_reps: 전극당 셔플 반복 횟수
        base_seed: 기본 시드 (재현성 보장)

    Returns:
        list of dicts — 각 dict는 한 번의 셔플 실험 결과:
            fold, electrode, rep,
            baseline_acc/auc/f1,
            perm_acc/auc/f1,
            pi_acc/auc/f1 (= baseline - perm)
    """
    print(f'\n{"="*60}')
    print(f'Fold {fold}: loading model and data...')

    # SOGC 내부에서 amask 텐서를 생성할 때 사용하는 device 동기화
    Net.device = device
    model = load_fold_model(fold, device)

    # 테스트 데이터: 해당 fold에서 제외된 피험자 1명의 전체 데이터
    _, test_dataset = get_dataset(SUBJECTS, fold)
    x_all, y_all = collect_test_data(test_dataset)

    # ── Baseline 평가: 셔플 없이 원본 데이터로 성능 측정 ──
    baseline_loader = make_dataloader(x_all, y_all)
    baseline = evaluate_from_loader(model, baseline_loader, device)
    print(f'Fold {fold} baseline: acc={baseline["acc"]:.4f}, '
          f'auc={baseline["auc"]:.4f}, f1={baseline["f1"]:.4f}')

    # ── 전극별 셔플 → 평가 → PI 계산 ──
    results = []
    for ei in range(N_ELECTRODES):
        t0 = time.time()

        for rep in range(n_reps):
            # 시드 구성: base + fold*10000 + electrode*100 + rep
            # → 모든 (fold, electrode, rep) 조합이 고유한 시드를 가짐
            seed = base_seed + fold * 10000 + ei * 100 + rep
            rng = np.random.default_rng(seed)

            # 전극 ei의 feature를 샘플 간 셔플
            x_perm = permute_electrode(x_all, ei, rng)
            perm_loader = make_dataloader(x_perm, y_all)
            perm = evaluate_from_loader(model, perm_loader, device)

            # PI = baseline - permuted (양수이면 해당 전극이 성능에 기여)
            results.append({
                'fold': fold,
                'electrode': ei,
                'rep': rep,
                'baseline_acc': baseline['acc'],
                'baseline_auc': baseline['auc'],
                'baseline_f1': baseline['f1'],
                'perm_acc': perm['acc'],
                'perm_auc': perm['auc'],
                'perm_f1': perm['f1'],
                'pi_acc': baseline['acc'] - perm['acc'],
                'pi_auc': baseline['auc'] - perm['auc'],
                'pi_f1': baseline['f1'] - perm['f1'],
            })

        elapsed = time.time() - t0
        if (ei + 1) % 10 == 0 or ei == 0:
            print(f'  Electrode {ei:2d}/{N_ELECTRODES} done '
                  f'({n_reps} reps, {elapsed:.1f}s)')

    return results


# ═════════════════════════════════════════════════════════════
# 7. 전체 Fold PI 계산 (+ 중간 저장 / 이어하기)
# ═════════════════════════════════════════════════════════════
def compute_pi_all_folds(folds, device, n_reps, base_seed, resume_df=None):
    """지정된 모든 fold에 대해 PI를 계산하고 중간 결과를 저장.

    각 fold 완료 시마다:
    - fold별 개별 CSV 저장 (./result/pi_fold{i}_raw.csv)
    - 누적 전체 CSV 저장 (./result/pi_raw_scores.csv)
    → 중단 후 --resume 옵션으로 이어하기 가능

    Args:
        folds: 계산할 fold 번호 리스트 (예: [0,1,...,14])
        device: 연산 장치
        n_reps: 전극당 셔플 반복 횟수
        base_seed: 기본 시드
        resume_df: 이전 실행의 raw_scores DataFrame (이어하기용, None이면 처음부터)

    Returns:
        raw_df: 전체 결과 DataFrame
    """
    os.makedirs('./result', exist_ok=True)

    all_results = []

    # ── 이어하기: 이전에 완료된 fold는 건너뜀 ──
    if resume_df is not None:
        all_results = resume_df.to_dict('records')
        completed_folds = set(resume_df['fold'].unique())
        folds = [f for f in folds if f not in completed_folds]
        print(f'Resuming: {len(completed_folds)} folds already done, '
              f'{len(folds)} remaining')

    # ── fold별 순차 실행 ──
    for fold in folds:
        fold_results = compute_pi_single_fold(fold, device, n_reps, base_seed)
        all_results.extend(fold_results)

        # fold별 개별 저장 (디버깅/분석 용도)
        fold_df = pd.DataFrame(fold_results)
        fold_df.to_csv(f'./result/pi_fold{fold}_raw.csv', index=False)

        # 누적 전체 저장 (중단 대비)
        raw_df = pd.DataFrame(all_results)
        raw_df.to_csv('./result/pi_raw_scores.csv', index=False)
        print(f'Fold {fold} saved. Total rows so far: {len(all_results)}')

    raw_df = pd.DataFrame(all_results)
    return raw_df


# ═════════════════════════════════════════════════════════════
# 8. 결과 집계
# ═════════════════════════════════════════════════════════════
def summarize_results(raw_df):
    """전체 raw PI 점수를 전극별로 집계하여 요약 테이블 생성.

    집계 방식:
    - 각 전극에 대해 모든 (fold x rep) 결과를 모아 평균/표준편차 계산
    - n_observations = folds x n_reps (예: 15 x 5 = 75)
    - 평균 PI 기준으로 순위 매김 (1 = 가장 중요)

    Args:
        raw_df: compute_pi_all_folds()의 반환값

    Returns:
        summary: 62행 DataFrame — 전극별 PI 평균, 표준편차, 순위
    """
    summary = raw_df.groupby('electrode').agg(
        pi_acc_mean=('pi_acc', 'mean'),   # accuracy 하락폭 평균
        pi_acc_std=('pi_acc', 'std'),     # accuracy 하락폭 표준편차
        pi_auc_mean=('pi_auc', 'mean'),   # AUC 하락폭 평균
        pi_auc_std=('pi_auc', 'std'),     # AUC 하락폭 표준편차
        pi_f1_mean=('pi_f1', 'mean'),     # F1 하락폭 평균
        pi_f1_std=('pi_f1', 'std'),       # F1 하락폭 표준편차
        n_observations=('pi_acc', 'count'),  # 관측 수 (= folds x reps)
    ).reset_index()

    # 각 지표별 순위 (PI가 클수록 순위가 높음 = 더 중요한 전극)
    summary['rank_by_acc'] = summary['pi_acc_mean'].rank(ascending=False).astype(int)
    summary['rank_by_auc'] = summary['pi_auc_mean'].rank(ascending=False).astype(int)
    summary['rank_by_f1'] = summary['pi_f1_mean'].rank(ascending=False).astype(int)

    summary = summary.sort_values('rank_by_acc')
    return summary


# ═════════════════════════════════════════════════════════════
# 9. 메인 실행
# ═════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='Compute Permutation Importance for SOGNN electrodes')
    parser.add_argument('--folds', type=int, nargs='+', default=None,
                        help='Fold indices to evaluate (default: all 0-14)')
    parser.add_argument('--n_reps', type=int, default=10,
                        help='Number of permutation repetitions per electrode (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed (default: 42)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to existing pi_raw_scores.csv to resume from')
    args = parser.parse_args()

    folds = args.folds if args.folds is not None else list(range(SUBJECTS))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 실행 설정 출력 ──
    print(f'Device: {device}')
    print(f'Folds: {folds}')
    print(f'Repetitions per electrode: {args.n_reps}')
    total = len(folds) * N_ELECTRODES * args.n_reps
    print(f'Total evaluations: {len(folds)} folds x {N_ELECTRODES} electrodes '
          f'x {args.n_reps} reps = {total}')

    # ── 이어하기 로드 ──
    resume_df = None
    if args.resume and os.path.exists(args.resume):
        resume_df = pd.read_csv(args.resume)
        print(f'Loaded {len(resume_df)} rows from {args.resume}')

    # ── PI 계산 실행 ──
    t_start = time.time()
    raw_df = compute_pi_all_folds(folds, device, args.n_reps, args.seed, resume_df)
    elapsed = time.time() - t_start

    # ── 결과 집계 및 저장 ──
    summary = summarize_results(raw_df)
    summary.to_csv('./result/pi_summary.csv', index=False)

    # ── 최종 보고 ──
    print(f'\n{"="*60}')
    print(f'DONE in {elapsed/60:.1f} minutes')
    print(f'Raw scores: ./result/pi_raw_scores.csv ({len(raw_df)} rows)')
    print(f'Summary:    ./result/pi_summary.csv ({len(summary)} rows)')
    print(f'\nTop 10 electrodes by PI (accuracy drop):')
    print(summary.head(10)[['electrode', 'pi_acc_mean', 'pi_acc_std',
                            'rank_by_acc']].to_string(index=False))


if __name__ == '__main__':
    main()
