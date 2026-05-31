#!/usr/bin/env python3
"""
KG-StockMixer 교육용 미니 데모
================================
10개 NASDAQ 종목 / 직접 정의한 KG / GAT 기반 / 단계별 학습 과정 출력

실행: python demo.py
VSCode: 각 #%% 블록을 Shift+Enter로 셀 단위 실행 가능
"""
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)

#%%
"""하이퍼파라미터"""

config = {
    'stocks':    ['AAPL','GOOG','MSFT','AMZN','NVDA',
                  'META','TSLA','INTC','AMD','QCOM'],
    'T':         300,    # 총 거래일
    'lookback':  16,     # 모델 입력 길이
    'fea_num':   5,      # 특성 수 (OHLCV 대응)
    'train_end': 220,
    'valid_end': 260,
    'epochs':    40,
    'lr':        0.001,
    'alpha':     0.1,    # rank loss 가중치
    'hidden':    32,
    'gat_heads': 2,
}
S   = len(config['stocks'])
idx = {t: i for i, t in enumerate(config['stocks'])}


#%%
"""KG 정의 — 직접 수정해보세요"""

RELATIONS = ['Supplier', 'Customer', 'Competitor', 'Partner']
R = len(RELATIONS)
REL_IDX = {r: i for i, r in enumerate(RELATIONS)}

# (from, to, 관계): 뉴스 기사·공급망 데이터 기반
KG_EDGES = [
    # Supplier: from이 to에 부품/서비스 공급
    ('INTC', 'AAPL',  'Supplier'),   # Intel CPU → Apple Mac
    ('NVDA', 'AAPL',  'Supplier'),   # NVIDIA GPU → Apple Mac Pro
    ('QCOM', 'AAPL',  'Supplier'),   # Qualcomm modem → iPhone
    ('INTC', 'MSFT',  'Supplier'),   # Intel → Microsoft 서버
    ('AMD',  'MSFT',  'Supplier'),   # AMD → Microsoft Xbox
    ('NVDA', 'AMZN',  'Supplier'),   # NVIDIA → AWS 데이터센터
    ('NVDA', 'GOOG',  'Supplier'),   # NVIDIA → Google Cloud
    # Customer: from이 to로부터 구매
    ('AAPL', 'INTC',  'Customer'),
    ('AAPL', 'QCOM',  'Customer'),
    ('MSFT', 'INTC',  'Customer'),
    ('AMZN', 'NVDA',  'Customer'),
    # Competitor: 직접 시장 경쟁
    ('AAPL', 'GOOG',  'Competitor'), # 스마트폰 OS
    ('AAPL', 'MSFT',  'Competitor'), # PC/생산성
    ('GOOG', 'META',  'Competitor'), # 온라인 광고
    ('GOOG', 'MSFT',  'Competitor'), # 클라우드/검색
    ('AMD',  'INTC',  'Competitor'), # CPU
    ('AMD',  'NVDA',  'Competitor'), # GPU
    ('NVDA', 'AMD',   'Competitor'),
    ('AMZN', 'MSFT',  'Competitor'), # 클라우드
    # Partner: 전략적 협력
    ('MSFT', 'META',  'Partner'),    # Azure-Meta 클라우드
    ('MSFT', 'AMZN',  'Partner'),
]


#%%
"""KG 행렬 구성 및 시각화"""

def build_kg(stocks, edges):
    """엣지 목록 → (S, S, R) 인접 텐서"""
    n = len(stocks)
    mat = np.zeros((n, n, R), dtype=np.float32)
    for src, dst, rel in edges:
        if src in idx and dst in idx:
            mat[idx[src], idx[dst], REL_IDX[rel]] = 1.0
    return mat

def kg_to_gat_mask(mat):
    """
    KG → GAT 마스크
      연결 있음 → 0.0    (attention 허용)
      연결 없음 → -1e9   (softmax 후 ≈ 0, 사실상 차단)
    """
    connected = (mat.sum(axis=2) > 0)
    return np.where(connected, 0.0, -1e9).astype(np.float32)

def print_kg(mat, stocks):
    print("\n" + "="*58)
    print("  KG 구조")
    print("="*58)
    print(f"  shape: {mat.shape}  ({len(stocks)} × {len(stocks)} × {R} relations)")
    print(f"  Relations: {RELATIONS}")
    print(f"  총 엣지: {int((mat>0).sum())}개  |  연결 종목: {int((mat.sum(axis=(1,2))>0).sum())}/{len(stocks)}개\n")

    for r_idx, rname in enumerate(RELATIONS):
        edges_r = [(stocks[i], stocks[j])
                   for i in range(len(stocks))
                   for j in range(len(stocks))
                   if mat[i,j,r_idx] > 0]
        if edges_r:
            pairs = '  '.join(f"{s}→{d}" for s,d in edges_r)
            print(f"  [{rname}]  {pairs}")

    print("\n  GAT mask (● = 0.0 허용 / · = -1e9 차단)  행=from, 열=to")
    mask = kg_to_gat_mask(mat)
    header = "        " + " ".join(f"{s:4s}" for s in stocks)
    print(header)
    for i, s in enumerate(stocks):
        row = " ".join("  ● " if mask[i,j]==0 else "  · " for j in range(len(stocks)))
        print(f"  {s:5s}  {row}")
    print("="*58 + "\n")


#%%
"""합성 데이터 생성"""

def generate_data(config):
    """
    KG 관계 있는 종목끼리 수익률 상관관계가 높은 합성 데이터.
    섹터 공통 팩터 + 개별 노이즈로 구성.
    """
    T   = config['T']
    rng = np.random.default_rng(42)

    sectors = {
        'device': ['AAPL','TSLA'],
        'cloud':  ['GOOG','MSFT','AMZN','META'],
        'chip':   ['NVDA','INTC','AMD','QCOM'],
    }
    factor = np.zeros((T, S))
    for members in sectors.values():
        # 섹터 공통 팩터 약화 (0.008→0.003): 너무 쉬운 패턴 방지
        f = rng.normal(0, 0.003, T).cumsum()
        for m in members:
            if m in idx:
                factor[:, idx[m]] = f

    # 개별 노이즈 강화 (0.012→0.018): 현실적 IC 범위
    returns = factor + rng.normal(0, 0.018, (T, S))
    prices  = 100 * np.exp(np.cumsum(returns, axis=0))

    F = config['fea_num']
    eod = np.zeros((S, T, F), dtype=np.float32)
    for t in range(T):
        c = prices[t]
        eod[:,t,0] = c / prices[max(0,t-1)] - 1
        eod[:,t,1] = c / c.mean()
        eod[:,t,2] = rng.uniform(0.99,1.01,S) * c / c.mean()
        eod[:,t,3] = rng.uniform(0.99,1.01,S) * c / c.mean()
        eod[:,t,4] = rng.lognormal(0, 0.3, S)

    return {
        'eod':   eod,
        'gt':    returns.T.astype(np.float32),
        'mask':  np.ones((S, T), dtype=np.float32),
        'price': (prices.T / 100).astype(np.float32),
    }


#%%
"""모델 — GAT 기반 KGGATMixer"""

class KGGATMixerMini(nn.Module):
    """
    단순화된 단일-헤드 GAT KG Mixer.

    Step 1: 선형 변환      z_i = W @ h_i
    Step 2: Attention score e_ij = LeakyReLU(a_src·z_i + a_dst·z_j)
    Step 3: Masked softmax  α_ij = softmax_j(e_ij + mask_ij)
                            mask: 연결=0, 비연결=-1e9
    Step 4: 이웃 집계       z_i' = Σ_j α_ij · z_j
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W     = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(out_dim))
        self.a_dst = nn.Parameter(torch.empty(out_dim))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.normal_(self.a_src, std=0.01)
        nn.init.normal_(self.a_dst, std=0.01)

    def forward(self, H, mask):
        """
        H:    (S, D)  종목 표현
        mask: (S, S)  GAT 마스크 (연결=0, 비연결=-1e9)
        반환: (S, out_dim)
        """
        Z = self.W(H)                              # (S, out_dim)

        # Additive attention: e_ij = LeakyReLU(a_src·z_i + a_dst·z_j)
        e_src = (Z * self.a_src).sum(-1, keepdim=True)   # (S, 1)
        e_dst = (Z * self.a_dst).sum(-1, keepdim=True)   # (S, 1)
        e     = F.leaky_relu(e_src + e_dst.T, 0.2)       # (S, S)

        alpha = F.softmax(e + mask, dim=1)         # (S, S) masked softmax
        return F.elu(alpha @ Z)                    # (S, out_dim)


class StockMixerMini(nn.Module):
    """
    교육용 KG-StockMixer

    흐름:
      x (S,T,F) → Conv1d → Flatten → time_fc → h_time (S,H)
                                               ↓
                                          KGGATMixer → h_kg (S,H)
                                               ↓
                             concat(h_time, h_kg) → out_fc → pred (S,1)
    """
    def __init__(self, config, gat_mask=None):
        super().__init__()
        T = config['lookback']
        F = config['fea_num']
        H = config['hidden']

        self.conv    = nn.Conv1d(F, F, kernel_size=2, stride=2)
        self.time_fc = nn.Linear(F * (T // 2), H)
        self.gat     = KGGATMixerMini(H, H)
        self.out_fc  = nn.Linear(H * 2, 1)

        if gat_mask is not None:
            self.register_buffer('gat_mask', torch.FloatTensor(gat_mask))
        else:
            self.register_buffer('gat_mask', torch.zeros(len(config['stocks']),
                                                          len(config['stocks'])))

    def forward(self, x, gat_mask=None):
        """
        x:        (S, T, F)
        gat_mask: 동적 KG 주입 시 사용, 없으면 self.gat_mask
        """
        mask = gat_mask if gat_mask is not None else self.gat_mask
        S    = x.shape[0]

        # Time Mixing
        h = x.permute(0, 2, 1)
        h = F.relu(self.conv(h)).reshape(S, -1)
        h_time = F.relu(self.time_fc(h))    # (S, H)

        # KG Mixing (GAT)
        h_kg = self.gat(h_time, mask)       # (S, H)

        # Fusion: tanh로 [-1,1] 바운딩 후 0.05 스케일 → 수익률 직접 예측
        return torch.tanh(self.out_fc(torch.cat([h_time, h_kg], dim=-1))) * 0.05


#%%
"""손실함수 / 평가"""

def get_loss(pred, gt, mask, alpha):
    """
    MSE + Pairwise Rank Loss (return 공간에서 직접 계산)
    pred: 모델이 예측한 수익률 (S, 1)
    gt  : 실제 수익률          (S, 1)
    """
    S = pred.shape[0]
    e = torch.ones(S, 1, device=pred.device)
    mse  = F.mse_loss(pred * mask, gt * mask)
    diff = pred @ e.T - e @ pred.T
    rank = torch.mean(F.relu(diff * (e @ gt.T - gt @ e.T) * (mask @ mask.T)))
    return mse + alpha * rank, mse, rank

def get_batch(data, config, offset):
    L  = config['lookback']
    x  = torch.FloatTensor(data['eod'][:,  offset:offset+L, :])
    gt = torch.FloatTensor(data['gt'][:,   offset+L:offset+L+1])
    m  = torch.FloatTensor(data['mask'][:, offset+L:offset+L+1])
    return x, gt, m

def train_epoch(model, optimizer, data, config):
    model.train()
    L = config['lookback']
    offsets = np.random.permutation(config['train_end'] - L - 1)
    total = 0
    for off in offsets:
        x, gt, m = get_batch(data, config, off)
        optimizer.zero_grad()
        loss, _, _ = get_loss(model(x), gt, m, config['alpha'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
    return total / len(offsets)

@torch.no_grad()
def evaluate(model, data, config, start, end):
    model.eval()
    L = config['lookback']
    S = len(config['stocks'])
    n = end - start
    pred_arr = np.zeros((S, n))
    gt_arr   = np.zeros((S, n))
    mask_arr = np.zeros((S, n))
    for col, off in enumerate(range(start - L, end - L)):
        x, gt, m = get_batch(data, config, off)
        pred_arr[:, col] = model(x)[:, 0].numpy()
        gt_arr[:,   col] = gt[:, 0].numpy()
        mask_arr[:, col] = m[:, 0].numpy()
    ics = []
    for t in range(n):
        valid = mask_arr[:, t] > 0
        if valid.sum() < 2: continue
        pr, gt_t = pred_arr[valid, t], gt_arr[valid, t]
        if pr.std() > 1e-8:
            ics.append(np.corrcoef(pr, gt_t)[0, 1])
    return float(np.mean(ics)) if ics else 0.0


#%%
"""메인"""

if __name__ == '__main__':

    # 1. KG 구성 및 시각화
    print("\n[ 1단계 ] KG 구성 및 시각화")
    kg_mat   = build_kg(config['stocks'], KG_EDGES)
    gat_mask = kg_to_gat_mask(kg_mat)
    print_kg(kg_mat, config['stocks'])

    # 2. 데이터
    print("[ 2단계 ] 데이터 생성")
    data = generate_data(config)
    print(f"  eod : {data['eod'].shape}  (종목 × 거래일 × 특성)")
    print(f"  학습: 0~{config['train_end']}일  "
          f"검증: {config['train_end']}~{config['valid_end']}일  "
          f"테스트: {config['valid_end']}~{config['T']}일\n")

    # 3. 모델
    print("[ 3단계 ] 모델 초기화")
    model     = StockMixerMini(config, gat_mask=gat_mask)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"  파라미터: {n_params:,}개")
    print("  구조: Conv1d → time_fc → KGGATMixer(GAT) → out_fc\n")

    # 4. 학습
    print("[ 4단계 ] 학습")
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Valid IC':>9}  {'Test IC':>8}")
    print("  " + "-"*40)
    best_val, best_test = -np.inf, 0.0

    for ep in range(1, config['epochs'] + 1):
        tr_loss = train_epoch(model, optimizer, data, config)
        val_ic  = evaluate(model, data, config, config['train_end'], config['valid_end'])
        test_ic = evaluate(model, data, config, config['valid_end'], config['T'])

        if val_ic > best_val:
            best_val, best_test = val_ic, test_ic

        if ep % 5 == 0 or ep == 1:
            mark = " ★" if val_ic == best_val else ""
            print(f"  {ep:>5}  {tr_loss:>10.4f}  {val_ic:>9.4f}  {test_ic:>8.4f}{mark}")

    # 5. 결과
    print("\n" + "="*50)
    print(f"  Best Valid IC : {best_val:.4f}")
    print(f"  Best Test  IC : {best_test:.4f}")
    print("="*50)
    print("\n  실습 포인트:")
    print("  1. KG_EDGES 엣지 추가/삭제 → IC 변화 관찰")
    print("  2. gat_mask 대신 torch.zeros 사용 → KG 없는 baseline 비교")
    print("  3. config['alpha'] 조정 → rank loss 효과 확인")
    print("  4. config['gat_heads'] 조정 → multi-head 효과 확인\n")
