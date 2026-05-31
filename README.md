# KG-StockMixer 교육용 데모

NASDAQ 주가 예측에 지식 그래프(KG)와 GAT를 결합한 모델의 교육용 구현입니다.

---

## 파일 구조

```
KG_Demo/
├── demo.py          # 로컬 단독 실행 (합성 데이터, 외부 의존 없음)
├── demo_server.py   # 서버 실험 (실제 NASDAQ 데이터 + Wikidata KG)
├── run_demo.sh      # SLURM 제출 스크립트
└── data/
    ├── eod_data.pkl      # (50, 1245, 5) OHLCV 정규화 특성
    ├── mask_data.pkl     # (50, 1245)    거래 가능 여부
    ├── gt_data.pkl       # (50, 1245)    실제 수익률
    ├── price_data.pkl    # (50, 1245)    정규화 종가
    ├── tickers.txt       # 50개 종목명
    └── wiki_relation.npy # (50, 50, 43) Wikidata KG
```

---

## 빠른 시작

### 1) 로컬 실행 (외부 파일 불필요)

```bash
python demo.py
```

합성 데이터로 10개 종목에 대해 GAT 기반 학습을 실행합니다.  
KG 인접 행렬과 학습 과정이 단계별로 출력됩니다.

### 2) 서버 직접 실행

```bash
conda activate stockmixer
python demo_server.py                # KG 사용 (Wikidata)
python demo_server.py --no_kg        # KG 없는 baseline
python demo_server.py --epochs 50    # epoch 수 조정
```

### 3) SLURM 제출 (GPU)

```bash
sbatch run_demo.sh                   # GPU 서버에서 실행
sbatch run_demo.sh --no_kg           # baseline 비교
tail -f logs/demo_*.out              # 로그 실시간 확인
squeue -u $USER                      # job 상태 확인
```

---

## 서버 경로 설정

`demo_server.py`는 파일 위치 기준 상대경로를 사용합니다.

```python
BASE_DIR = Path(__file__).parent   # demo_server.py가 있는 폴더
DATA_DIR = BASE_DIR / 'data'       # data/ 폴더
```

따라서 **어느 위치에서 실행해도 동작**합니다.

```bash
# 어디서 실행해도 OK
python /home/user/KG_Demo/demo_server.py
cd /home/user && python KG_Demo/demo_server.py
```

`run_demo.sh`도 마찬가지입니다:

```bash
cd "$(dirname "$0")"   # 스크립트 위치로 자동 이동
python demo_server.py "$@"
```

---

## 데이터 구조

| 변수 | shape | 설명 |
|------|-------|------|
| `eod_data` | (S, T, 5) | 종목별 일별 OHLCV 정규화 특성 |
| `mask_data` | (S, T) | 거래 가능 여부 (1=유효, 0=상장폐지) |
| `gt_data` | (S, T) | 종목별 일별 실제 수익률 |
| `price_data` | (S, T) | 종목별 정규화 종가 |

**학습/검증/테스트 분할:**

```
train : index    0 ~ 756   (2013-01-02 ~ 2015-11-19)
valid : index  756 ~ 1008  (2015-11-19 ~ 2016-11-18)
test  : index 1008 ~       (2016-11-18 ~ 2017-10-27)
```

---

## 모델 구조

```
x (S, T, F)
    │
    ▼  Time Mixing
  Conv1d(F→F, kernel=2, stride=2)   # 시계열 압축: T → T//2
    │
  Flatten → Linear(F×T//2, H)       # 종목별 시계열 표현
    │
  h_time (S, H)
    │
    ├── (KG 있을 때) ──────────────────────────────────┐
    │                KGGATLayer(H, H)                   │
    │                → h_kg (S, H)                      │
    │                                                    │
    └── concat(h_time, h_kg)   or   h_time (KG 없을 때)
                │
              Linear → tanh × 0.05
                │
            pred (S, 1)   ← 예측 수익률 [-0.05, +0.05]
```

### KGGATLayer (Graph Attention Network)

```
Step 1  z_i = W · h_i                                  선형 변환

Step 2  e_ij = LeakyReLU(a_src·z_i + a_dst·z_j)       Attention score
               ↑ i: 쿼리(수신)  j: 키(송신)

Step 3  α_ij = softmax_j(e_ij + mask_ij)              Masked softmax
               mask_ij = 0    (KG 연결)
               mask_ij = -1e9 (KG 비연결 → softmax ≈ 0)

Step 4  h_i' = ELU(Σ_j α_ij · z_j)                   이웃 집계
```

KG 마스크 덕분에 **KG에서 관계가 정의된 종목끼리만** 정보를 주고받습니다.

---

## 손실 함수

```
Loss = MSE + α × RankLoss

MSE:       예측 수익률과 실제 수익률의 평균 제곱 오차
RankLoss:  종목 쌍(i, j)에서 실제 수익 순위와 예측 순위가 다를 때 패널티
α:         --alpha 인자로 조정 (기본 0.1)
```

---

## 평가 지표 — IC (Information Coefficient)

```
IC = 일별 Pearson 상관계수(예측 수익률, 실제 수익률)의 평균

IC > 0    : 예측 방향이 실제와 일치하는 경향
IC > 0.02 : 실제 금융 예측에서 유의미한 수준
```

---

## 실습 포인트

```bash
# 1. KG 유무 비교
python demo_server.py
python demo_server.py --no_kg

# 2. Ranking loss 효과
python demo_server.py --alpha 0      # MSE only
python demo_server.py --alpha 0.5    # rank 강조

# 3. demo.py에서 KG_EDGES 직접 수정
#    엣지 추가/삭제 후 IC 변화 관찰
```

`demo.py`의 `KG_EDGES` 리스트를 수정하면 KG 구조를 바꿔가며 실험할 수 있습니다.

---

## 환경 설정

```bash
# conda 환경 (PyTorch + numpy)
conda create -n stockmixer python=3.8
conda activate stockmixer
pip install torch numpy
```
