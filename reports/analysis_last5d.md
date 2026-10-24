# 실거래 성과 분석 2026-09-29 ~ 2026-10-06 (KST)

- OKX bills 레그 358건 → 윈도우 내 청산 완료 라운드트립 **14**건, 미청산 포지션 0개
- trades.jsonl 라운드트립 14건 (진입기록 없는 청산 3건 스킵). bills 대비 부족분 = 거래소 SL/강제청산/수동 청산 추정

## 1. 핵심 지표

- **trades**: 14
- **wins**: 7
- **losses**: 7
- **win_rate**: 50.0
- **total_net_pnl**: 40.47
- **total_fee**: -21.85
- **profit_factor**: 2.86
- **avg_win**: 8.89
- **avg_loss**: -3.11
- **payoff**: 2.86
- **expectancy**: 2.89
- **avg_roe_pct**: 0.3
- **mdd_usdt**: -7.75
- **mdd_pct_of_start**: None
- **avg_hold_h**: 0.4
- **median_hold_h**: 0.1
- **avg_win_hold_h**: 0.3
- **avg_loss_hold_h**: 0.5
- **avg_dca_entries**: 13.71
- **liquidations**: 0
- **taker_ratio**: 0.98

## 방향별

| side   |   n |   win_rate |   net_pnl |   avg_pnl |   pf |   avg_hold_h |
|:-------|----:|-----------:|----------:|----------:|-----:|-------------:|
| long   |   8 |       25   |     -6.98 |     -0.87 | 0.4  |          0.5 |
| short  |   6 |       83.3 |     47.45 |      7.91 | 5.66 |          0.4 |

## 섹터별

| sector   |   n |   win_rate |   net_pnl |   avg_pnl |   pf |   avg_hold_h |
|:---------|----:|-----------:|----------:|----------:|-----:|-------------:|
| major    |   6 |       33.3 |     -1.45 |     -0.24 | 0.62 |          0.4 |
| alt      |   8 |       62.5 |     41.92 |      5.24 | 3.34 |          0.5 |

## BTC 레짐(진입시점)별

| regime     |   n |   win_rate |   net_pnl |   avg_pnl |   pf |   avg_hold_h |
|:-----------|----:|-----------:|----------:|----------:|-----:|-------------:|
| trend_up   |   5 |       20   |     -1.14 |     -0.23 | 0.7  |          0.4 |
| trend_down |   9 |       66.7 |     41.6  |      4.62 | 3.32 |          0.4 |

## 진입 시간대 KST별

|   entry_hour_kst |   n |   win_rate |   net_pnl |   avg_pnl |          pf |   avg_hold_h |
|-----------------:|----:|-----------:|----------:|----------:|------------:|-------------:|
|               10 |   1 |      100   |      0.17 |      0.17 | 1.72168e+08 |          0.1 |
|                9 |  13 |       46.2 |     40.29 |      3.1  | 2.85        |          0.5 |

## 종목별

| symbol            |   n |   win_rate |   net_pnl |   avg_pnl |          pf |   avg_hold_h |
|:------------------|----:|-----------:|----------:|----------:|------------:|-------------:|
| UNI/USDT:USDT     |   1 |          0 |    -10.17 |    -10.17 | 0           |          0.1 |
| USELESS/USDT:USDT |   1 |          0 |     -3.93 |     -3.93 | 0           |          0.7 |
| BZ/USDT:USDT      |   1 |          0 |     -3.82 |     -3.82 | 0           |          0.7 |
| BTC/USDT:USDT     |   2 |         50 |     -1.54 |     -0.77 | 0.1         |          0.1 |
| ETH/USDT:USDT     |   1 |          0 |     -0.95 |     -0.95 | 0           |          0.1 |
| SOL/USDT:USDT     |   1 |          0 |     -0.72 |     -0.72 | 0           |          0.1 |
| XRP/USDT:USDT     |   1 |          0 |     -0.44 |     -0.44 | 0           |          1.8 |
| OKB/USDT:USDT     |   1 |        100 |      1.91 |      1.91 | 1.90832e+09 |          0.3 |
| AVAX/USDT:USDT    |   1 |        100 |      2.19 |      2.19 | 2.19114e+09 |          0.1 |
| SUI/USDT:USDT     |   1 |        100 |      2.68 |      2.68 | 2.6775e+09  |          0.1 |
| INJ/USDT:USDT     |   1 |        100 |      7.14 |      7.14 | 7.14052e+09 |          0.2 |
| BCH/USDT:USDT     |   1 |        100 |      7.68 |      7.68 | 7.68387e+09 |          0.8 |
| SEI/USDT:USDT     |   1 |        100 |     40.43 |     40.43 | 4.04313e+10 |          1   |

## 일자별(청산일)

| day   |   n |   win_rate |   net_pnl |   avg_pnl |   pf |   avg_hold_h |
|:------|----:|-----------:|----------:|----------:|-----:|-------------:|
| 10-02 |   5 |       20   |     -1.14 |     -0.23 | 0.7  |          0.4 |
| 09-29 |   9 |       66.7 |     41.6  |      4.62 | 3.32 |          0.4 |

## 최대 손실 10건

| symbol            | side   | sector   | entry_kst   | exit_kst    |   hold_h |   n_entries |   n_exits |   lev |   margin |   net_pnl |    roe | regime     |   btc_adx |   btc_move_pct | liquidation   |
|:------------------|:-------|:---------|:------------|:------------|---------:|------------:|----------:|------:|---------:|----------:|-------:|:-----------|----------:|---------------:|:--------------|
| UNI/USDT:USDT     | short  | alt      | 09-29 09:17 | 09-29 09:21 |    0.055 |           7 |        15 |     5 |  631.232 |   -10.173 | -0.016 | trend_down |      20.1 |           0    | False         |
| USELESS/USDT:USDT | long   | alt      | 09-29 09:18 | 09-29 10:01 |    0.703 |          33 |        17 |     5 |  150.496 |    -3.928 | -0.026 | trend_down |      20.1 |          -0.15 | False         |
| BZ/USDT:USDT      | long   | alt      | 09-29 09:18 | 09-29 10:01 |    0.711 |           5 |        10 |     5 |  266.778 |    -3.822 | -0.014 | trend_down |      20.1 |          -0.15 | False         |
| BTC/USDT:USDT     | long   | major    | 10-02 09:02 | 10-02 09:08 |    0.1   |           1 |         1 |    10 |  127.156 |    -1.709 | -0.013 | trend_up   |      21.7 |           0    | False         |
| ETH/USDT:USDT     | long   | major    | 10-02 09:11 | 10-02 09:15 |    0.061 |           2 |         2 |    10 |  105.184 |    -0.949 | -0.009 | trend_up   |      21.7 |           0    | False         |
| SOL/USDT:USDT     | long   | major    | 10-02 09:01 | 10-02 09:04 |    0.055 |           9 |         8 |    10 |   52.648 |    -0.719 | -0.014 | trend_up   |      21.7 |           0    | False         |
| XRP/USDT:USDT     | long   | major    | 10-02 09:11 | 10-02 11:00 |    1.812 |           3 |        11 |    10 |   52.61  |    -0.437 | -0.008 | trend_up   |      21.7 |           0    | False         |
| BTC/USDT:USDT     | short  | major    | 09-29 10:03 | 09-29 10:08 |    0.078 |          14 |         2 |     5 |  199.587 |     0.172 |  0.001 | trend_down |      21   |           0    | False         |
| OKB/USDT:USDT     | long   | alt      | 09-29 09:19 | 09-29 09:35 |    0.263 |          24 |        23 |     5 |  648.481 |     1.908 |  0.003 | trend_down |      20.1 |           0    | False         |
| AVAX/USDT:USDT    | short  | major    | 09-29 09:18 | 09-29 09:21 |    0.057 |          15 |         2 |     5 |  133.319 |     2.191 |  0.016 | trend_down |      20.1 |           0    | False         |

## 최대 수익 10건

| symbol         | side   | sector   | entry_kst   | exit_kst    |   hold_h |   n_entries |   n_exits |   lev |   margin |   net_pnl |    roe | regime     |   btc_adx |   btc_move_pct | liquidation   |
|:---------------|:-------|:---------|:------------|:------------|---------:|------------:|----------:|------:|---------:|----------:|-------:|:-----------|----------:|---------------:|:--------------|
| SEI/USDT:USDT  | short  | alt      | 09-29 09:18 | 09-29 10:16 |    0.964 |          31 |        31 |     5 |  530.17  |    40.431 |  0.076 | trend_down |      20.1 |          -0.15 | False         |
| BCH/USDT:USDT  | short  | alt      | 09-29 09:18 | 09-29 10:06 |    0.808 |          12 |         8 |     5 |  533.061 |     7.684 |  0.014 | trend_down |      20.1 |          -0.15 | False         |
| INJ/USDT:USDT  | short  | alt      | 09-29 09:19 | 09-29 09:28 |    0.151 |          34 |        30 |     5 |  428.485 |     7.141 |  0.017 | trend_down |      20.1 |           0    | False         |
| SUI/USDT:USDT  | long   | alt      | 10-02 09:11 | 10-02 09:18 |    0.117 |           2 |         6 |     5 |  210.048 |     2.677 |  0.013 | trend_up   |      21.7 |           0    | False         |
| AVAX/USDT:USDT | short  | major    | 09-29 09:18 | 09-29 09:21 |    0.057 |          15 |         2 |     5 |  133.319 |     2.191 |  0.016 | trend_down |      20.1 |           0    | False         |
| OKB/USDT:USDT  | long   | alt      | 09-29 09:19 | 09-29 09:35 |    0.263 |          24 |        23 |     5 |  648.481 |     1.908 |  0.003 | trend_down |      20.1 |           0    | False         |
| BTC/USDT:USDT  | short  | major    | 09-29 10:03 | 09-29 10:08 |    0.078 |          14 |         2 |     5 |  199.587 |     0.172 |  0.001 | trend_down |      21   |           0    | False         |
| XRP/USDT:USDT  | long   | major    | 10-02 09:11 | 10-02 11:00 |    1.812 |           3 |        11 |    10 |   52.61  |    -0.437 | -0.008 | trend_up   |      21.7 |           0    | False         |
| SOL/USDT:USDT  | long   | major    | 10-02 09:01 | 10-02 09:04 |    0.055 |           9 |         8 |    10 |   52.648 |    -0.719 | -0.014 | trend_up   |      21.7 |           0    | False         |
| ETH/USDT:USDT  | long   | major    | 10-02 09:11 | 10-02 09:15 |    0.061 |           2 |         2 |    10 |  105.184 |    -0.949 | -0.009 | trend_up   |      21.7 |           0    | False         |

- 손실 집중도: 최악 5건이 총손실의 **94.7%**

## 재진입 상위 종목 (왕복 의심)

| symbol         |   roundtrips |
|:---------------|-------------:|
| BTC/USDT:USDT  |            2 |
| AVAX/USDT:USDT |            1 |
| BCH/USDT:USDT  |            1 |
| BZ/USDT:USDT   |            1 |
| ETH/USDT:USDT  |            1 |
| INJ/USDT:USDT  |            1 |
| OKB/USDT:USDT  |            1 |
| SEI/USDT:USDT  |            1 |
| SOL/USDT:USDT  |            1 |
| SUI/USDT:USDT  |            1 |

## 참고: 기존 daily_report

- daily_report_20261002.txt: OKX  | N=331 | WR=43.2% | Avg=+0.273% | Sum=+90.41%
- daily_report_20261003.txt: OKX  | N=339 | WR=43.7% | Avg=+0.294% | Sum=+99.53%
- daily_report_20261004.txt: OKX  | N=334 | WR=43.4% | Avg=+0.289% | Sum=+96.39%
- daily_report_20261005.txt: OKX  | N=334 | WR=43.4% | Avg=+0.289% | Sum=+96.39%
- daily_report_20261006.txt: OKX  | N=334 | WR=43.4% | Avg=+0.289% | Sum=+96.39%
