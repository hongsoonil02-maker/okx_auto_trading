# -*- coding: utf-8 -*-
"""
backtest_v4pro_scenarios.py — Argon V4-Pro (20x 4-Major Wave Rider) 충실 재현 백테스트
시나리오 S0~S5 비교. 실전 코드(okx_jev_demo_trader.py)의 진입/청산 로직을 1H/15m 캔들로 재현.

근사치(문서화):
  - 인트라바 경로는 1H OHLC로 근사. 동일 봉 SL/TP 동시 터치 시 SL 우선(보수적).
  - Stage 플로어/트레일은 봉 고저로 피크 갱신 후 플로어 가격 터치 시 플로어가로 청산.
  - MA20 구조청산은 확정봉 종가로 청산. 진입은 신호봉 다음 봉 시가.
  - Jev AI 게이트는 패스스루(실측 승인률 82%), 펀딩 가드 미적용(실측 미발동).
  - 수수료: 테이커 0.05% x 2 (메이커 실패→시장가 폴백 실측 반영). 펀딩피 제외.
"""
import os, time, math, json
import requests
import numpy as np
import pandas as pd

DATA_DIR = "/home/hongsoonil02/quant_system/data"
os.makedirs(DATA_DIR, exist_ok=True)
SYMS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "XRP-USDT-SWAP"]
CORE = ("BTC-USDT-SWAP", "ETH-USDT-SWAP")

# ── 데이터 / 체결 수익률 ─────────────────────────────────
def linear_return(entry_px, exit_px, is_long):
    """USDT-margined linear swap return; live trader와 동일한 가격 PnL식."""
    if is_long:
        return exit_px / entry_px - 1.0
    return (entry_px - exit_px) / entry_px


def align_15m_to_confirmed_1h(d15raw, ref_ts):
    """1H 확정 시점에 실전에서 보는 마지막 15m 확정봉(t+45m)을 맞춘다."""
    out = {}
    target_ts = np.asarray(ref_ts) + 45 * 60 * 1000
    for s, f in d15raw.items():
        idx = np.searchsorted(f["ts"].values, target_ts, side="right") - 1
        idx = np.clip(idx, 0, len(f) - 1)
        out[s] = f.iloc[idx].reset_index(drop=True)
    return out


def trim_1h_to_15m_coverage(data_1h, data_15m_raw):
    """1H 확정 시점에 실제 존재하는 15m 봉만 사용하도록 양끝을 자른다."""
    start = max(f["ts"].iloc[0] for f in data_15m_raw.values()) - 45 * 60 * 1000
    end = min(f["ts"].iloc[-1] for f in data_15m_raw.values()) - 45 * 60 * 1000
    return {
        s: f[(f["ts"] >= start) & (f["ts"] <= end)].reset_index(drop=True)
        for s, f in data_1h.items()
    }


def fetch(inst, bar, target):
    f = os.path.join(DATA_DIR, f"{inst}_{bar}_v4.pkl")
    if os.path.exists(f):
        df = pd.read_pickle(f)
        if len(df) >= target * 0.9:
            return df
    out, after = [], ""
    print(f"fetch {inst} {bar} x{target}")
    while len(out) < target:
        p = {"instId": inst, "bar": bar, "limit": "100"}
        if after: p["after"] = after
        try:
            d = requests.get("https://www.okx.com/api/v5/market/history-candles", params=p, timeout=8).json().get("data", [])
        except Exception as e:
            print("err", e); break
        if not d: break
        out.extend(d); after = d[-1][0]; time.sleep(0.12)
    if not out: raise RuntimeError(f"no data {inst} {bar}")
    out = sorted(out, key=lambda x: int(x[0]))
    df = pd.DataFrame(out, columns=["ts","open","high","low","close","vol","vc","vcq","confirm"])
    df = df[df["confirm"] == "1"].reset_index(drop=True)
    df["ts"] = df["ts"].astype(np.int64)
    for c in ("open","high","low","close","vol"): df[c] = df[c].astype(float)
    df.to_pickle(f)
    print(f"  cached {len(df)}: {pd.to_datetime(df['ts'].iloc[0],unit='ms')} ~ {pd.to_datetime(df['ts'].iloc[-1],unit='ms')}")
    return df

def wilder_adx(h, l, c, n=14):
    up, dn = h.diff(), -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    a = lambda s: s.ewm(alpha=1/n, adjust=False).mean()
    tr_, pdm_, mdm_ = a(tr), a(pd.Series(pdm, index=h.index)), a(pd.Series(mdm, index=h.index))
    pdi, mdi = 100*pdm_/tr_, 100*mdm_/tr_
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()

def prep_1h(df):
    df = df.copy()
    c = df["close"]
    df["ma5"], df["ma10"], df["ma20"] = c.rolling(5).mean(), c.rolling(10).mean(), c.rolling(20).mean()
    df["ma20_slope3"] = (df["ma20"]/df["ma20"].shift(3)-1)*100
    df["ma5_slope1"] = (df["ma5"]/df["ma5"].shift(1)-1)*100
    df["spread"] = (df["ma5"]-df["ma20"]).abs()/df["ma20"]*100
    df["dist"] = (c/df["ma20"]-1)*100
    df["ret3"] = (c/c.shift(3)-1)*100
    df["adx"] = wilder_adx(df["high"], df["low"], c)
    above = c > df["ma20"]
    # 파동 시작 봉 인덱스(원웨이브원샷·파동나이): 현재 사이드 유지가 깨진 지점+1
    grp = (above != above.shift()).cumsum()
    df["wave_start"] = df.groupby(grp).cumcount().pipe(lambda x: df.index - x)
    df["wave_age"] = df.groupby(grp).cumcount()
    df["ema20d"] = c.ewm(span=20 * 24, adjust=False).mean()
    df["ema20d_slope3d"] = (df["ema20d"] / df["ema20d"].shift(3 * 24) - 1) * 100
    return df

def prep_15m(df):
    df = df.copy()
    c = df["close"]
    df["ema9"], df["ema20"] = c.ewm(span=9, adjust=False).mean(), c.ewm(span=20, adjust=False).mean()
    tr = pd.concat([df["high"]-df["low"],(df["high"]-c.shift()).abs(),(df["low"]-c.shift()).abs()],axis=1).max(axis=1)
    df["atr_ratio"] = tr.rolling(14).mean()/c
    df["vol_ratio"] = df["vol"].rolling(3).mean()/df["vol"].rolling(20).mean()
    df["adx"] = wilder_adx(df["high"], df["low"], c)
    return df

# ── 시나리오 ────────────────────────────────────────────
BASE = dict(
    leverage=20.0, breadth=4,
    base_sl_core=0.016, base_sl_alt=0.020, max_dyn_sl=0.032, dyn_mult=2.2,
    tp=0.10,
    s1_arm=0.0150, s1_floor=0.0025,
    s2_arm=0.0220, s2_floor=0.0110,
    s3_arm=0.0320, s3_drop=0.0100, s3_min=0.0100,
    ma20_exit=True, ma20_buf=0.0020,
    slope3=0.18, imp_slope3=0.10, imp_ma5=0.30, imp_ret3=1.50,
    min_spread=0.30, min_adx1h=18.0, min_adx15=18.0, min_volr=0.75,
    min_dist=0.15, max_dist=3.10, max_dist_imp=4.80,
    btc_wave_max=10, sym_wave_max=6,
    max_pos=2, margin_pct=0.20, margin_cap=5000.0,
    post_loss_cd=1, normal_cd=1,   # 60분 = 1H봉 1개
    streak_lock=2, lockout_bars=4,
    deleverage=False,              # S4: 연패 시 20→10→5
    fee=0.0005,
    risk_sizing=False, risk_per_trade=0.005, max_total_risk=0.012,
    risk_fee=0.001, risk_slip=0.001,
    daily_stop=False, daily_stop_pct=0.02,
    daily_filter=False, daily_slope_min=0.20,
)
SCEN = {
    "S0_현행_20x":        {},
    "S1_레버10x":         {"leverage": 10.0},
    "S2_SL완화_TP단축":   {"base_sl_core":0.025, "base_sl_alt":0.025, "max_dyn_sl":0.040, "dyn_mult":3.0, "tp":0.05},
    "S3_게이트3/4":       {"breadth": 3},
    "S4_연패디레버리징":   {"deleverage": True},
    "S5_트레일수확":      {"tp": 9.99, "s2_floor":0.015, "s3_arm":0.025, "s3_drop":0.007},
    "S8_현재_10x_3of4":  {"leverage":10.0, "breadth":3,
                           "base_sl_core":0.025, "base_sl_alt":0.025,
                           "max_dyn_sl":0.040, "dyn_mult":3.0, "tp":0.05},
    "R1_S8_리스크0.5%":  {"leverage":10.0, "breadth":3,
                           "base_sl_core":0.025, "base_sl_alt":0.025,
                           "max_dyn_sl":0.040, "dyn_mult":3.0, "tp":0.05,
                           "risk_sizing":True},
    "R2_S8_리스크+일손실2%": {"leverage":10.0, "breadth":3,
                           "base_sl_core":0.025, "base_sl_alt":0.025,
                           "max_dyn_sl":0.040, "dyn_mult":3.0, "tp":0.05,
                           "risk_sizing":True, "daily_stop":True},
}
LIQ_ADVERSE = {20.0: 0.040, 10.0: 0.090, 5.0: 0.190}   # 격리 청산까지 역행폭 근사

def run(data1h, data15, cfg0, start_eq=5000.0):
    cfg = dict(BASE); cfg.update(cfg0)
    syms = list(data1h.keys())
    n = len(data1h["BTC-USDT-SWAP"])
    eq = start_eq
    pos = {}            # sym -> dict
    cooldown_until = {} # sym -> bar idx
    lockout = {}        # (sym,side) -> bar idx
    streak = {}         # (sym,side) -> int
    traded_waves = set()
    trades = []
    eq_curve = [start_eq]
    liq_count = 0
    current_day = None
    daily_start_eq = start_eq
    daily_realized = 0.0

    def equity_mtm(bar_idx):
        """확정손익 + 보유 포지션의 종가 기준 미실현·예상 수수료."""
        value = eq
        for s, p in pos.items():
            close_px = data1h[s]["close"].iloc[bar_idx]
            unrealized = linear_return(p["entry"], close_px, p["side"] == "LONG")
            value += p["margin"] * p["lev"] * (unrealized - 2 * cfg["fee"])
        return value

    def lev_for(sym, side):
        if not cfg["deleverage"]: return cfg["leverage"]
        s = streak.get((sym, side), 0)
        return {0: 20.0, 1: 10.0}.get(s, 5.0)

    for t in range(60, n - 1):
        b_ts = data1h["BTC-USDT-SWAP"]["ts"].iloc[t]
        bar_day = pd.to_datetime(b_ts, unit="ms").date()
        if bar_day != current_day:
            current_day = bar_day
            daily_start_eq = eq
            daily_realized = 0.0
        # ── 포지션 관리 (봉 t 인트라바: 시가→고저→종가) ──
        for sym in list(pos.keys()):
            p = pos[sym]; d = data1h[sym]
            o, hi, lo, cl = (d[c].iloc[t] for c in ("open","high","low","close"))
            is_long = p["side"] == "LONG"
            fav = hi if is_long else lo            # 유리 극값
            adv = lo if is_long else hi            # 불리 극값
            ent = p["entry"]
            # 피크 갱신
            if is_long: p["peak"] = max(p["peak"], fav)
            else:       p["peak"] = min(p["peak"], fav)
            peak_pl = linear_return(ent, p["peak"], is_long)
            # 청산(갭): 불리 극값이 청산선 돌파
            liq_line = LIQ_ADVERSE.get(p["lev"], 0.040)
            adverse = -linear_return(ent, adv, is_long)
            p["mae_pct"] = max(p.get("mae_pct", 0.0), adverse * 100.0)
            exit_px, reason = None, None
            if adverse >= liq_line:
                exit_px, reason = ent*(1-(liq_line if is_long else -liq_line)), "LIQUIDATED"
            # SL (보수적: 먼저)
            elif adverse >= p["sl"]:
                exit_px, reason = ent*(1-p["sl"] if is_long else 1+p["sl"]), "STOP_LOSS"
            else:
                cur_pl_lo = linear_return(ent, adv, is_long)   # 봉 내 최악 수익률
                # TP
                fav_pl = linear_return(ent, fav, is_long)
                if fav_pl >= p["tp"]:
                    exit_px, reason = ent*(1+p["tp"] if is_long else 1-p["tp"]), "WAVE_TP"
                # Stage2 플로어
                elif peak_pl >= cfg["s2_arm"] and cur_pl_lo <= cfg["s2_floor"]:
                    exit_px = ent*(1+cfg["s2_floor"] if is_long else 1-cfg["s2_floor"]); reason = "STAGE2_LOCK"
                # Stage1 본절 플로어
                elif peak_pl >= cfg["s1_arm"] and cur_pl_lo <= cfg["s1_floor"]:
                    exit_px = ent*(1+cfg["s1_floor"] if is_long else 1-cfg["s1_floor"]); reason = "STAGE1_BE"
                # Stage3 트레일
                elif peak_pl >= cfg["s3_arm"]:
                    trail_px = p["peak"]*(1-cfg["s3_drop"]) if is_long else p["peak"]*(1+cfg["s3_drop"])
                    trail_pl = linear_return(ent, trail_px, is_long)
                    hit = (lo <= trail_px) if is_long else (hi >= trail_px)
                    if hit and trail_pl >= cfg["s3_min"]:
                        exit_px, reason = trail_px, "STAGE3_TRAIL"
                # MA20 구조청산 (종가 확정)
                if exit_px is None and cfg["ma20_exit"]:
                    ma5, ma20 = d["ma5"].iloc[t], d["ma20"].iloc[t]
                    if is_long and cl < ma20*(1-cfg["ma20_buf"]) and ma5 < ma20:
                        exit_px, reason = cl, "MA20_EXIT"
                    elif (not is_long) and cl > ma20*(1+cfg["ma20_buf"]) and ma5 > ma20:
                        exit_px, reason = cl, "MA20_EXIT"
            if exit_px is not None:
                pl = linear_return(ent, exit_px, is_long)
                if reason == "LIQUIDATED":
                    pnl = -p["margin"]; liq_count += 1
                else:
                    pnl = p["margin"]*p["lev"]*(pl - 2*cfg["fee"])
                eq += pnl
                daily_realized += pnl
                is_loss = pnl < 0
                k = (sym, p["side"])
                streak[k] = streak.get(k, 0) + 1 if is_loss else 0
                if is_loss and streak[k] >= cfg["streak_lock"]:
                    lockout[k] = t + cfg["lockout_bars"]; streak[k] = 0
                cooldown_until[sym] = t + (cfg["post_loss_cd"] if is_loss else cfg["normal_cd"])
                trades.append(dict(sym=sym, side=p["side"], lev=p["lev"], pnl=pnl,
                                   ret=pl*100, reason=reason, bars=t-p["t0"],
                                   mfe=linear_return(ent, p["peak"], is_long)*100,
                                   mae=-p.get("mae_pct", 0.0),
                                   etime=str(pd.to_datetime(p["entry_ts"],unit="ms"))[:16],
                                   xtime=str(pd.to_datetime(d["ts"].iloc[t],unit="ms"))[:16]))
                del pos[sym]

        # ── 진입 평가 (봉 t 확정 기준 → t+1 시가 진입) ──
        if cfg["daily_stop"] and daily_start_eq > 0 and daily_realized <= -daily_start_eq * cfg["daily_stop_pct"]:
            eq_curve.append(equity_mtm(t)); continue
        if len(pos) >= cfg["max_pos"]:
            eq_curve.append(equity_mtm(t)); continue
        w = {s: data1h[s].iloc[t] for s in syms}
        bull = sum(1 for s in syms if w[s]["close"] > w[s]["ma20"] and w[s]["ma5"] > w[s]["ma10"])
        bear = sum(1 for s in syms if w[s]["close"] < w[s]["ma20"] and w[s]["ma5"] < w[s]["ma10"])
        bt = w["BTC-USDT-SWAP"]
        b15 = data15["BTC-USDT-SWAP"].iloc[t]
        bslope, bm5s, bret3 = bt["ma20_slope3"], bt["ma5_slope1"], bt["ret3"]
        bull_vel = bslope >= cfg["slope3"] or (bslope >= cfg["imp_slope3"] and bm5s >= cfg["imp_ma5"] and bret3 >= cfg["imp_ret3"])
        bear_vel = bslope <= -cfg["slope3"] or (bslope <= -cfg["imp_slope3"] and bm5s <= -cfg["imp_ma5"] and bret3 <= -cfg["imp_ret3"])
        sync_bull = (bt["close"] > bt["ma20"] and bt["ma5"] > bt["ma10"] and b15["close"] > b15["ema20"]
                     and bull >= cfg["breadth"] and bull_vel)
        sync_bear = (bt["close"] < bt["ma20"] and bt["ma5"] < bt["ma10"] and b15["close"] < b15["ema20"]
                     and bear >= cfg["breadth"] and bear_vel)
        if not (sync_bull or sync_bear):
            eq_curve.append(equity_mtm(t)); continue
        side = "LONG" if sync_bull else "SHORT"
        if (side == "LONG" and not cfg.get("allow_long", True)) or (side == "SHORT" and not cfg.get("allow_short", True)):
            eq_curve.append(equity_mtm(t)); continue
        if cfg.get("daily_filter", False):
            btc_row = data1h["BTC-USDT-SWAP"].iloc[t]
            d_ema = btc_row["ema20d"]
            d_slope = btc_row["ema20d_slope3d"]
            if pd.isna(d_ema) or pd.isna(d_slope):
                eq_curve.append(equity_mtm(t)); continue
            if side == "LONG" and not (btc_row["close"] > d_ema and d_slope >= cfg["daily_slope_min"]):
                eq_curve.append(equity_mtm(t)); continue
            if side == "SHORT" and not (btc_row["close"] < d_ema and d_slope <= -cfg["daily_slope_min"]):
                eq_curve.append(equity_mtm(t)); continue
        impulse = (sync_bull and bslope < cfg["slope3"]) or (sync_bear and bslope > -cfg["slope3"])
        dist_cap = cfg["max_dist_imp"] if impulse else cfg["max_dist"]
        cands = []
        for s in syms:
            if s in pos or t < cooldown_until.get(s, 0) or t < lockout.get((s, side), 0):
                continue
            r = w[s]
            if (s, int(bt["wave_start"])) in traded_waves or (s, int(r["wave_start"])) in traded_waves:
                continue
            if bt["wave_age"] > cfg["btc_wave_max"] and r["wave_age"] > cfg["sym_wave_max"]:
                continue
            if r["spread"] < cfg["min_spread"] or r["adx"] < cfg["min_adx1h"]:
                continue
            s15 = data15[s].iloc[t]
            dma, m5s = r["dist"], r["ma5_slope1"]
            if sync_bull:
                ok1 = (r["close"] > r["ma20"] and r["close"] > r["ma5"] and r["ma5"] > r["ma10"] >= r["ma20"]*0.999
                       and m5s > 0.05 and cfg["min_dist"] <= dma <= dist_cap)
                ok15 = (s15["close"] > s15["ema9"] > s15["ema20"]) and s15["adx"] >= cfg["min_adx15"]
            else:
                ok1 = (r["close"] < r["ma20"] and r["close"] < r["ma5"] and r["ma5"] < r["ma10"] <= r["ma20"]*1.001
                       and m5s < -0.05 and -dist_cap <= dma <= -cfg["min_dist"])
                ok15 = (s15["close"] < s15["ema9"] < s15["ema20"]) and s15["adx"] >= cfg["min_adx15"]
            if not (ok1 and ok15) or s15["vol_ratio"] < cfg["min_volr"]:
                continue
            score = r["ret3"] if sync_bull else -r["ret3"]
            cands.append((s, score, s15["atr_ratio"]))
        cands.sort(key=lambda x: x[1], reverse=True)
        for s, score, atr15 in cands:
            if len(pos) >= cfg["max_pos"]: break
            base_sl = cfg["base_sl_core"] if s in CORE else cfg["base_sl_alt"]
            sl = max(base_sl, min(cfg["max_dyn_sl"], cfg["dyn_mult"]*atr15))
            entry_px = data1h[s]["open"].iloc[t+1]
            lev = lev_for(s, side)
            risk_buffer = sl + cfg["risk_fee"] + cfg["risk_slip"]
            if cfg["risk_sizing"]:
                open_risk = sum(p["risk"] for p in pos.values())
                allowed_risk = min(
                    eq * cfg["risk_per_trade"],
                    max(0.0, eq * cfg["max_total_risk"] - open_risk),
                )
                notional = allowed_risk / risk_buffer
                margin = notional / lev
                trade_risk = notional * risk_buffer
            else:
                margin = min(eq*cfg["margin_pct"], cfg["margin_cap"])
                trade_risk = margin * lev * risk_buffer
            if margin < 50: continue
            pos[s] = dict(side=side, entry=entry_px, peak=entry_px, sl=sl, tp=cfg["tp"],
                          lev=lev, margin=margin, risk=trade_risk, t0=t+1,
                          entry_ts=data1h[s]["ts"].iloc[t+1])
            traded_waves.add((s, int(bt["wave_start"])))
            traded_waves.add((s, int(w[s]["wave_start"])))
        eq_curve.append(equity_mtm(t))

    final_eq = equity_mtm(n - 1)
    tr = pd.DataFrame(trades)
    if len(tr) == 0:
        return dict(cfg=cfg0, trades=0, final_eq=final_eq, net=final_eq-start_eq,
                    ret_pct=(final_eq/start_eq-1)*100)
    wins = tr[tr.pnl > 0]; losses = tr[tr.pnl <= 0]
    eqs = pd.Series(eq_curve)
    mdd = ((eqs - eqs.cummax())/eqs.cummax()).min()*100
    return dict(
        cfg=cfg0, trades=len(tr), winrate=len(wins)/len(tr)*100,
        net=final_eq-start_eq, closed_net=tr.pnl.sum(), final_eq=final_eq,
        ret_pct=(final_eq/start_eq-1)*100,
        pf=(wins.pnl.sum()/abs(losses.pnl.sum())) if len(losses) and losses.pnl.sum() != 0 else float("inf"),
        avg_w=wins.pnl.mean() if len(wins) else 0, avg_l=losses.pnl.mean() if len(losses) else 0,
        expect=tr.pnl.mean(), mdd=mdd, liq=liq_count,
        tp_hits=len(tr[tr.reason=="WAVE_TP"]),
        reasons=tr.reason.value_counts().to_dict(),
        by_sym=tr.groupby("sym").pnl.sum().round(1).to_dict(),
        tr=tr,
    )

def main():
    days = int(os.getenv("BT_DAYS", "120"))
    t1h, t15 = days*24+80, days*96+80
    d1 = {s: prep_1h(fetch(s, "1H", t1h)) for s in SYMS}
    d15raw = {s: prep_15m(fetch(s, "15m", t15)) for s in SYMS}
    d1 = trim_1h_to_15m_coverage(d1, d15raw)
    # 15m → 1H 정렬 (실전과 동일하게 1H 확정 시점의 마지막 15m 확정봉)
    ref_ts = d1["BTC-USDT-SWAP"]["ts"].values
    d15 = align_15m_to_confirmed_1h(d15raw, ref_ts)
    span0 = pd.to_datetime(ref_ts[60], unit="ms"); span1 = pd.to_datetime(ref_ts[-1], unit="ms")
    span_days = (span1 - span0).total_seconds() / 86400.0
    print(f"\n백테스트 구간: {span0} ~ {span1} ({span_days:.1f}일, 1H봉 {len(ref_ts)-61}개)\n")
    rows = []
    for name, ov in SCEN.items():
        r = run(d1, d15, ov)
        if r.get("trades", 0) == 0:
            print(f"{name}: 거래 없음"); continue
        rows.append(r)
        print(f"[{name}] {r['trades']}건 승률 {r['winrate']:.1f}% PF {r['pf']:.2f} net {r['net']:+.0f}U "
              f"({r['ret_pct']:+.1f}%) MDD {r['mdd']:.1f}% 기대 {r['expect']:+.1f}U TP도달 {r['tp_hits']}건 청산 {r['liq']}건")
        print(f"   사유: {r['reasons']}")
        print(f"   심볼: {r['by_sym']}")
        tr = r["tr"]
        mid_ts = span0 + (span1 - span0) / 2
        exit_ts = pd.to_datetime(tr["xtime"])
        for label, subset in (
            ("전반", tr[exit_ts <= mid_ts]),
            ("후반", tr[exit_ts > mid_ts]),
            ("XRP제외", tr[tr["sym"] != "XRP-USDT-SWAP"]),
        ):
            wins = subset[subset.pnl > 0]["pnl"].sum()
            losses = abs(subset[subset.pnl <= 0]["pnl"].sum())
            pf = wins / losses if losses > 0 else float("inf")
            wr = (subset.pnl > 0).mean() * 100 if len(subset) else 0.0
            print(f"   {label}: {len(subset)}건 승률 {wr:.1f}% PF {pf:.2f} net {subset.pnl.sum():+.0f}U")
    pd.to_pickle(rows, os.path.join(DATA_DIR, "v4pro_scenario_results.pkl"))

if __name__ == "__main__":
    main()
