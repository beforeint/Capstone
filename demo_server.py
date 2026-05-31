#!/usr/bin/env python3
"""
KG-StockMixer 서버 실험 데모 (교육용)
======================================
실제 NASDAQ 50종목 데이터 + Wikidata KG + GAT 기반 학습

실행 방법:
  1) 직접 실행  : python demo_server.py
  2) SLURM 제출 : sbatch run_demo.sh
  3) KG 없는 비교: python demo_server.py --no_kg
  4) 로그 확인  : tail -f logs/demo_*.out
"""
#%%
import argparse, os, pickle, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#%%
"""경로 설정 — demo_server.py 기준 상대경로이므로 어디서 실행해도 동작"""

BASE_DIR  = Path(__file__).parent   # KG_Demo/
DATA_DIR  = BASE_DIR / 'data'       # KG_Demo/data/
LOG_DIR   = BASE_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)

#%%
"""argparse — 학생들이 실험 조건을 바꿔가며 비교할 수 있도록"""

def get_args(debug=False):
    parser = argparse.ArgumentParser(description='KG-StockMixer Demo')
    parser.add_argument('--epochs', type=int,   default=30,
                        help='학습 epoch 수')
    parser.add_argument('--lr',     type=float, default=0.001,
                        help='Adam optimizer 학습률')
    parser.add_argument('--alpha',  type=float, default=0.1,
                        help='Ranking loss 가중치 (0이면 MSE only)')
    parser.add_argument('--seed',   type=int,   default=0,
                        help='랜덤 시드 (재현성)')
    parser.add_argument('--no_kg',  action='store_true',
                        help='KG 없이 baseline 실험 → KG 유무 효과 비교 가능')
    return parser.parse_args(args=[] if debug else None)

#%%
"""재현성 고정"""

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

#%%
"""데이터 로드"""

def load_data():
    """
    data/ 폴더의 실제 NASDAQ 주가 데이터를 로드합니다.

    파일 구성:
      eod_data.pkl  : (S, T, 5) — 종목별 일별 OHLCV 정규화 특성
      mask_data.pkl : (S, T)    — 거래 가능 여부 (1=유효, 0=상장폐지 등)
      gt_data.pkl   : (S, T)    — 종목별 일별 실제 수익률 (ground truth)
      price_data.pkl: (S, T)    — 종목별 일별 정규화 종가

    학습/검증/테스트 분할 (날짜 기준):
      train : index 0   ~ 756  (2013-01-02 ~ 2015-11-19)
      valid : index 756 ~ 1008 (2015-11-19 ~ 2016-11-18)
      test  : index 1008~      (2016-11-18 ~ 2017-10-27)
    """
    print('\n[ 1단계 ] 데이터 로드', flush=True)
    def load_pkl(path):
        with open(path, 'rb') as f:
            return pickle.load(f).astype(np.float32)

    eod   = load_pkl(DATA_DIR / 'eod_data.pkl')
    mask  = load_pkl(DATA_DIR / 'mask_data.pkl')
    gt    = load_pkl(DATA_DIR / 'gt_data.pkl')
    price = load_pkl(DATA_DIR / 'price_data.pkl')
    tickers = (DATA_DIR / 'tickers.txt').read_text().strip().split('\n')

    S, T, feat = eod.shape
    print(f'  종목 수   : {S}개')
    print(f'  거래일 수 : {T}일')
    print(f'  특성 수   : {feat}개 (OHLCV)')
    print(f'  종목 예시 : {tickers[:5]} ...')
    return eod, mask, gt, price, tickers


def load_kg(S: int, no_kg: bool):
    """
    Wikidata 지식 그래프를 GAT 마스크로 변환합니다.

    KG 원본 형식:
      wiki_relation.npy : (S, S, 43) — 43종류 관계별 이진 행렬

    GAT 마스크 변환 (연결 여부만 사용):
      connected[i,j] = True  → mask[i,j] =   0.0  (attention 허용)
      connected[i,j] = False → mask[i,j] = -1e9   (attention 차단, softmax ≈ 0)

    no_kg=True 이면 None 반환 → 모델이 KG branch를 건너뜀
    """
    print('\n[ 2단계 ] KG 로드', flush=True)
    if no_kg:
        print('  KG 사용 안 함 → baseline 모드 (KG 효과 비교 실험)')
        return None

    rel  = np.load(DATA_DIR / 'wiki_relation.npy')     # (S, S, 43)
    conn = (rel.sum(axis=2) > 0)                        # (S, S) bool
    print(f'  KG 엣지: {int(conn.sum())}개  연결 종목: {int((conn.sum(1)>0).sum())}/{S}개')
    print(f'  변환: any-relation-exists → 0.0(허용) / -1e9(차단)')
    return np.where(conn, 0.0, -1e9).astype(np.float32)


#%%
"""모델 정의"""

class KGGATLayer(nn.Module):
    """
    단일-헤드 Graph Attention Network (GAT) 레이어.

    핵심 수식:
      Step 1 — 선형 변환       z_i = W · h_i
      Step 2 — Attention score  e_ij = LeakyReLU(a_src·z_i + a_dst·z_j)
      Step 3 — Masked softmax   α_ij = softmax_j(e_ij + mask_ij)
                                mask: 연결=0, 비연결=-1e9 → 비연결 차단
      Step 4 — 이웃 집계        h_i' = ELU(Σ_j α_ij · z_j)

    KG 마스크가 있으면 KG에 정의된 이웃 종목의 정보만 집계합니다.
    마스크 없이 학습하면 전체 종목에서 attention → 노이즈 증가.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.W     = nn.Linear(dim, dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(dim))
        self.a_dst = nn.Parameter(torch.empty(dim))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.normal_(self.a_src, std=0.01)
        nn.init.normal_(self.a_dst, std=0.01)

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H    : (S, D) 종목별 표현 벡터
            mask : (S, S) KG 마스크
        Returns:
            (S, D) KG 이웃 정보가 반영된 표현
        """
        Z     = self.W(H)                                  # (S, D)
        e_src = (Z * self.a_src).sum(-1, keepdim=True)    # (S, 1) 쿼리
        e_dst = (Z * self.a_dst).sum(-1, keepdim=True)    # (S, 1) 키
        e     = F.leaky_relu(e_src + e_dst.T, 0.2)        # (S, S) attention
        alpha = F.softmax(e + mask, dim=1)                 # (S, S) masked
        return F.elu(alpha @ Z)                            # (S, D) 집계


class KGStockMixer(nn.Module):
    """
    교육용 KG-StockMixer.

    전체 구조:
      ┌─ Time Mixing ─────────────────────────────────┐
      │  x (S,T,F) → Conv1d → Flatten → time_fc      │
      │  → h_time (S,H)  ← 시계열 패턴 표현          │
      └───────────────────────────────────────────────┘
                          ↓
      ┌─ KG Mixing ───────────────────────────────────┐
      │  h_time + KG mask → KGGATLayer               │
      │  → h_kg (S,H)  ← 이웃 종목 정보 통합         │
      └───────────────────────────────────────────────┘
                          ↓
      concat(h_time, h_kg) → out_fc → tanh·0.05
      → pred (S,1)  ← 수익률 예측값 ([-0.05, +0.05] 범위)

    no_kg=True이면 KG branch 생략, h_time만으로 예측합니다.
    """
    def __init__(self, S: int, lookback: int, feat: int,
                 hidden: int = 64, kg_mask=None):
        super().__init__()
        self.has_kg = kg_mask is not None

        # Time Mixing: Conv1d로 시계열 압축 후 FC
        self.conv    = nn.Conv1d(feat, feat, kernel_size=2, stride=2)
        self.time_fc = nn.Linear(feat * (lookback // 2), hidden)

        # KG Mixing + Fusion
        if self.has_kg:
            self.gat = KGGATLayer(hidden)
            self.out = nn.Linear(hidden * 2, 1)   # concat(time, kg) → 1
        else:
            self.out = nn.Linear(hidden, 1)        # time only → 1

        if kg_mask is not None:
            self.register_buffer('kg_mask', torch.FloatTensor(kg_mask))

    def forward(self, x: torch.Tensor, kg_mask=None) -> torch.Tensor:
        """
        Args:
            x      : (S, T, F) 종목별 시계열 특성
            kg_mask: 동적 KG 주입 시 사용 (None이면 self.kg_mask 사용)
        Returns:
            pred: (S, 1) 예측 수익률
        """
        S = x.shape[0]

        # Step 1: Time Mixing
        h    = F.relu(self.conv(x.permute(0, 2, 1))).reshape(S, -1)
        h_t  = F.relu(self.time_fc(h))              # (S, hidden)

        # Step 2: KG Mixing (있을 때만)
        if self.has_kg:
            mask = kg_mask if kg_mask is not None else self.kg_mask
            h_g  = self.gat(h_t, mask)              # (S, hidden)
            feat_vec = torch.cat([h_t, h_g], dim=-1)   # (S, hidden*2)
        else:
            feat_vec = h_t                           # (S, hidden)

        # Step 3: 수익률 예측 (tanh로 범위 제한)
        return torch.tanh(self.out(feat_vec)) * 0.05   # (S, 1)


#%%
"""손실 함수"""

def get_loss(pred: torch.Tensor, gt: torch.Tensor,
             mask: torch.Tensor, alpha: float):
    """
    MSE Loss + Pairwise Ranking Loss.

    MSE Loss:
      예측 수익률과 실제 수익률의 평균 제곱 오차.
      mask=0인 종목(거래 불가)은 제외.

    Pairwise Ranking Loss:
      '종목 A 수익률이 종목 B보다 높으면 A를 더 높게 예측해야 한다'는 조건.
      예측 순위와 실제 순위가 다를 때 패널티 부여.
      alpha 값이 클수록 순위 일치를 더 중시.

    전체: loss = MSE + alpha × RankLoss
    """
    S = pred.shape[0]
    e = torch.ones(S, 1, device=pred.device)

    mse  = F.mse_loss(pred * mask, gt * mask)

    # 모든 종목 쌍 (i,j)에 대해 순위 일치 여부 확인
    pred_diff = pred @ e.T - e @ pred.T    # pred_i - pred_j
    gt_diff   = e @ gt.T  - gt  @ e.T     # gt_j   - gt_i  (부호 반대)
    mask_pair = mask @ mask.T              # 둘 다 유효한 쌍만
    rank = torch.mean(F.relu(pred_diff * gt_diff * mask_pair))

    return mse + alpha * rank


#%%
"""배치 추출 / 평가"""

def get_batch(eod, gt, mask, lookback: int, offset: int):
    """
    offset 시점의 배치 데이터 추출.

    입력 x  : [offset : offset+lookback] 구간 (과거 lookback일)
    정답 gt : [offset+lookback] 시점 수익률 (다음 날 예측 대상)
    """
    L = lookback
    x = torch.FloatTensor(eod[:,  offset:offset+L,  :])
    g = torch.FloatTensor(gt[:,   offset+L:offset+L+1])
    m = torch.FloatTensor(mask[:, offset+L:offset+L+1])
    return x, g, m


@torch.no_grad()
def evaluate(model, eod, gt, mask, lookback, start, end, device):
    """
    IC (Information Coefficient) 계산.

    IC = 일별 Pearson 상관계수(예측 수익률, 실제 수익률)의 평균.
    IC > 0 : 예측 방향이 실제와 일치
    IC > 0.02: 금융 예측에서 실용적으로 유의미한 수준
    """
    model.eval()
    L = lookback; S = eod.shape[0]; n = end - start
    P = np.zeros((S, n)); G = np.zeros((S, n)); M = np.zeros((S, n))

    for col, off in enumerate(range(start - L, end - L)):
        x, g, m = get_batch(eod, gt, mask, L, off)
        P[:, col] = model(x.to(device))[:, 0].cpu().numpy()
        G[:, col] = g[:, 0].numpy()
        M[:, col] = m[:, 0].numpy()

    ics = []
    for t in range(n):
        valid = M[:, t] > 0
        if valid.sum() < 2: continue
        p, gt_t = P[valid, t], G[valid, t]
        if p.std() > 1e-8:
            ics.append(np.corrcoef(p, gt_t)[0, 1])
    return float(np.nanmean(ics)) if ics else 0.0


#%%
"""메인"""

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    LOOKBACK    = 16     # 입력 시퀀스 길이 (16 거래일 = 약 3주)
    VALID_INDEX = 756    # 학습/검증 경계 (2015-11-19)
    TEST_INDEX  = 1008   # 검증/테스트 경계 (2016-11-18)

    print('='*55)
    print(f'  KG-StockMixer 서버 실험')
    print(f'  device : {device}')
    print(f'  KG     : {"OFF (baseline)" if args.no_kg else "Wikidata"}')
    print(f'  epochs : {args.epochs}   lr : {args.lr}   seed : {args.seed}')
    print('='*55)

    # 데이터 / KG 로드
    eod, mask, gt, price, tickers = load_data()
    S, T, n_feat = eod.shape     # n_feat 사용 (F는 torch.nn.functional 예약)
    kg_mask = load_kg(S, args.no_kg)

    # 모델 초기화
    print('\n[ 3단계 ] 모델 초기화')
    model     = KGStockMixer(S, LOOKBACK, n_feat, hidden=64,
                             kg_mask=kg_mask).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f'  파라미터 수: {n_params:,}개')
    print(f'  구조: Conv1d → time_fc → {"KGGATLayer → " if not args.no_kg else ""}out_fc')

    # 학습 루프
    print(f'\n[ 4단계 ] 학습')
    print(f'  {"Epoch":>5}  {"Loss":>10}  {"Valid IC":>9}  {"Test IC":>9}')
    print('  ' + '-'*40)
    best_val = best_test = -np.inf
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        offsets = np.random.permutation(VALID_INDEX - LOOKBACK - 1)
        total_loss = 0
        for off in offsets:
            x, g, m = get_batch(eod, gt, mask, LOOKBACK, off)
            x, g, m = x.to(device), g.to(device), m.to(device)
            optimizer.zero_grad()
            loss = get_loss(model(x), g, m, args.alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        val  = evaluate(model, eod, gt, mask, LOOKBACK,
                        VALID_INDEX, TEST_INDEX, device)
        test = evaluate(model, eod, gt, mask, LOOKBACK,
                        TEST_INDEX, T, device)

        if val > best_val:
            best_val, best_test = val, test

        if ep % 5 == 0 or ep == 1:
            mark = ' ★' if val == best_val else ''
            print(f'  {ep:>5}  {total_loss/len(offsets):>10.5f}'
                  f'  {val:>9.4f}  {test:>9.4f}{mark}', flush=True)

    elapsed = time.time() - t0
    print(f'\n{"="*55}')
    print(f'  Best Valid IC : {best_val:.4f}')
    print(f'  Best Test  IC : {best_test:.4f}')
    print(f'  학습 시간     : {elapsed:.1f}초')
    print('='*55)
    print('\n  실습 포인트:')
    print('  → python demo_server.py --no_kg   : KG 없는 baseline 비교')
    print('  → python demo_server.py --alpha 0 : ranking loss 제거')
    print('  → sbatch run_demo.sh              : GPU 서버에서 실행')
