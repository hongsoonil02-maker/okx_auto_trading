# -*- coding: utf-8 -*-
"""
risk_manager.py - V1.1 리스크 관리 모듈

1. Max Drawdown (MDD) 관리
   - Peak Equity 기반 MDD 추적
   - 개별 봇 MDD 15% 초과 시 PANIC_MODE 진입
   - PANIC_MODE 진입 시 12시간 정지 (PAUSE)

2. 포트폴리오 MDD
   - 포트폴리오 전체 equity 15% 초과 시 전체 정지

3. 상관관계 리스크 (Correlation)
   - 동일 방향 포지션 수 제한 (최대 3개)
   - 코인 변동성 고려하여 리스크 파라미터 조정
"""

import os
import json
import time
import logging
from datetime import datetime
from utils_telegram import send_telegram_alert

logger = logging.getLogger("risk_manager")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "portfolio_state.json")

MAX_DRAWDOWN_PCT = 0.15  # 개별 봇 MDD 15%
PORTFOLIO_MAX_DD = 0.10  # 포트폴리오 전체 MDD 10% (Kill-Switch 마지노선) ← 5%에서 완화
PAUSE_DURATION_SEC = 14400  # 4시간 정지 (← 12시간에서 단축)
GRADUAL_RECOVERY_SEC = 7200  # PAUSE 해제 후 2시간은 절반 사이즈로 운영
MAX_SAME_SIDE_POS = 3  # 동일 방향 최대 포지션 수

_PORTFOLIO_KEY = "__PORTFOLIO__"


def _load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"portfolio_state 로드 실패: {e}")
    return {}


def _save_state(state: dict):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"portfolio_state 저장 실패: {e}")


def _check_portfolio_panic(state: dict, now: float) -> bool:
    """
    포트폴리오 전체 equity 기반 MDD를 추적합니다.
    PORTFOLIO_MAX_DD(15%) 초과 시 __PORTFOLIO__ 키에 pause_until 기록 후 True 반환.
    """
    portfolio = state.get(_PORTFOLIO_KEY, {})
    
    if portfolio.get("emergency_stop", False):
        logger.critical("[PORTFOLIO RISK] EMERGENCY STOP 발동 중입니다. (시스템 잠금)")
        return True

    pause_until = portfolio.get("pause_until", 0.0)

    if now < pause_until:
        rem_h = (pause_until - now) / 3600
        logger.warning(f"[PORTFOLIO RISK] MDD로 PAUSED: {rem_h:.1f}h 남음")
        return True

    # 포트폴리오 전체 equity 합산
    total_equity = sum(
        s.get("equity", 0.0) for k, s in state.items() if k != _PORTFOLIO_KEY
    )
    if total_equity <= 0:
        return False

    peak = portfolio.get("peak_equity", total_equity)
    if total_equity > peak:
        peak = total_equity
        portfolio["peak_equity"] = peak
        state[_PORTFOLIO_KEY] = portfolio

    dd_pct = (peak - total_equity) / peak if peak > 0 else 0.0

    if dd_pct >= PORTFOLIO_MAX_DD:
        portfolio["emergency_stop"] = True
        portfolio["peak_equity"] = peak
        state[_PORTFOLIO_KEY] = portfolio
        
        err_msg = (
            f"🚨 [EMERGENCY STOP] 시스템 계좌 MDD {dd_pct*100:.1f}% 초과 (한계선: {PORTFOLIO_MAX_DD*100:.0f}%).\n"
            f"즉각 모든 API 송신을 차단하고 봇들을 정지시킵니다. 관리자의 'Re-Enable' 명령 전까지 가동 불가."
        )
        logger.critical(err_msg)
        send_telegram_alert(err_msg)
        return True

    state[_PORTFOLIO_KEY] = portfolio
    return False


def check_drawdown(bot_name: str, current_equity: float) -> dict:
    """
    드로다운 체크. Equity 기반 MDD를 추적합니다.

    Returns:
        {
            "status": "NORMAL" | "PANIC_MODE" | "PAUSED",
            "peak_equity": float,
            "current_dd_pct": float,
            "pause_until": float (timestamp)
        }
    """
    if current_equity <= 0:
        return {"status": "NORMAL"}

    state = _load_state()
    bot_state = state.get(
        bot_name,
        {
            "initial_equity": current_equity,
            "peak_equity": current_equity,
            "pause_until": 0.0,
        },
    )

    initial_equity = bot_state.get("initial_equity", current_equity)
    peak_equity = bot_state.get("peak_equity", current_equity)
    pause_until = bot_state.get("pause_until", 0.0)
    now = time.time()

    # PAUSE 상태 확인
    if now < pause_until:
        rem_hours = (pause_until - now) / 3600
        logger.warning(f"[RISK] {bot_name} is PAUSED: {rem_hours:.1f} hours remaining.")
        return {
            "status": "PAUSED",
            "peak_equity": peak_equity,
            "current_dd_pct": 0.0,
            "pause_until": pause_until,
        }
    elif pause_until > 0 and now >= pause_until:
        logger.info(f"[RISK] {bot_name} PAUSE 해제. 트레이싱 리셋.")
        bot_state["pause_until"] = 0.0
        bot_state["peak_equity"] = current_equity
        peak_equity = current_equity

    # Peak 갱신
    if current_equity > peak_equity:
        peak_equity = current_equity
        bot_state["peak_equity"] = peak_equity

    # Trailing Drawdown (V1.1)
    total_profit_pct = (peak_equity - initial_equity) / initial_equity if initial_equity > 0 else 0.0
    dynamic_mdd = 0.15 if total_profit_pct >= 0.20 else MAX_DRAWDOWN_PCT

    # Drawdown 계산
    dd_amount = peak_equity - current_equity
    dd_pct = dd_amount / peak_equity if peak_equity > 0 else 0.0

    status = "NORMAL"

    # MDD 초과 시 PANIC_MODE
    if dd_pct >= dynamic_mdd:
        status = "PANIC_MODE"
        bot_state["pause_until"] = now + PAUSE_DURATION_SEC
        logger.critical(
            f"[RISK] {bot_name} MAX DRAWDOWN 초과! ({dd_pct*100:.1f}% >= Limit {dynamic_mdd*100:.1f}%) "
            f"→ {PAUSE_DURATION_SEC//3600}h 정지 & 매매 중지"
        )

    # equity 저장 (MDD 추적용)
    bot_state["equity"] = current_equity
    state[bot_name] = bot_state

    # 포트폴리오 MDD 체크
    if status == "NORMAL" and _check_portfolio_panic(state, now):
        status = "PAUSED"
        bot_state["pause_until"] = state[_PORTFOLIO_KEY].get(
            "pause_until", now + PAUSE_DURATION_SEC
        )
        state[bot_name] = bot_state

    _save_state(state)

    if dd_pct >= 0.03 and status == "NORMAL":
        logger.info(
            f"[RISK] {bot_name} Drawdown: -{dd_pct*100:.1f}% "
            f"(Limit: -{dynamic_mdd*100:.1f}%, Peak: {peak_equity:,.0f})"
        )

    return {
        "status": status,
        "peak_equity": peak_equity,
        "current_dd_pct": dd_pct,
        "pause_until": bot_state["pause_until"],
        "scale_factor": _get_scale_factor(bot_state, now),
    }


def _get_scale_factor(bot_state: dict, now: float) -> float:
    """
    PAUSE 해제 후 점진적 복귀를 위한 포지션 스케일 팩터.
    PAUSE 해제 후 GRADUAL_RECOVERY_SEC(기본 2시간) 동안은 0.5~1.0 사이로 선형 복귀.
    평상시 1.0 반환.
    """
    pause_until = bot_state.get("pause_until", 0.0)
    if pause_until <= 0:
        return 1.0  # PAUSE 이력 없음
    if now < pause_until:
        return 0.0  # 아직 PAUSED 중
    elapsed = now - pause_until  # PAUSE 해제 후 경과 시간
    if elapsed >= GRADUAL_RECOVERY_SEC:
        return 1.0  # 복귀 완료
    # 0.5 → 1.0 선형 복귀
    return 0.5 + 0.5 * (elapsed / GRADUAL_RECOVERY_SEC)


def check_correlation_risk(positions: dict, new_symbol: str, new_side: str = "LONG") -> bool:
    """
    동일 방향 포지션 수를 제한합니다.
    코인 변동성이 높으므로 방향(LONG/SHORT)별로 제한합니다.

    Returns:
        True if safe (진입 허용), False if too concentrated (진입 금지)
    """
    if not positions:
        return True

    same_side_count = 0
    for sym, pos_data in positions.items():
        if pos_data.get("side", "LONG").upper() == new_side.upper():
            same_side_count += 1

    if same_side_count >= MAX_SAME_SIDE_POS:
        logger.warning(
            f"[RISK] 동일 방향 포지션 초과 (현재 {same_side_count}개). {new_symbol} 진입 금지."
        )
        return False

    return True

def is_emergency_stop() -> bool:
    """
    Emergency Stop 상태 확인용 퍼블릭 API. 
    봇들은 매 진입/청산 전 이 함수를 호출하여 True일 경우 API 전송을 차단해야 함.
    """
    state = _load_state()
    portfolio = state.get(_PORTFOLIO_KEY, {})
    return portfolio.get("emergency_stop", False)

def check_atr_volatility(recent_candles: list) -> dict:
    """
    ATR 20 기간 계산 로직 및 동적 변동성 엔진.
    recent_candles: [{'high': h, 'low': l, 'close': c}, ...]
    
    ATR 지표가 평균 대비 1.5배 이상 벌어지는 순간,
    진입 허들을 0.1% 하향 조정하고, 진입 규모를 1.2배 상향 조정.
    """
    if not recent_candles or len(recent_candles) < 21:
        return {"hurdle_mod": 0.0, "size_mod": 1.0}
    
    atrs = []
    for i in range(1, len(recent_candles)):
        h = recent_candles[i]['high']
        l = recent_candles[i]['low']
        pc = recent_candles[i-1]['close']
        tr = max(h - l, abs(h - pc), abs(l - pc))
        atrs.append(tr)
    
    if len(atrs) < 20:
        return {"hurdle_mod": 0.0, "size_mod": 1.0}
    
    current_atr = sum(atrs[-20:]) / 20.0
    avg_atr = sum(atrs) / len(atrs)
    
    if avg_atr > 0 and current_atr >= 1.5 * avg_atr:
        msg = f"⚡ [VOLATILITY ENGINE] ATR 과열 포착 (현재: {current_atr:.2f}, 평균: {avg_atr:.2f}). 진입 허들 -0.1%, 규모 1.2배 상향."
        logger.warning(msg)
        return {"hurdle_mod": -0.001, "size_mod": 1.2}
    
    return {"hurdle_mod": 0.0, "size_mod": 1.0}


def classify_atr_regime(recent_candles: list) -> dict:
    """
    ATR 기반 시장 레짐 분류.

    Returns:
        {
            "regime": "EXPANSION" | "NORMAL" | "CONTRACTION",
            "hurdle_mod": float,
            "size_mod": float,
            "atr_ratio": float,
        }
    """
    if not recent_candles or len(recent_candles) < 21:
        return {"regime": "NORMAL", "hurdle_mod": 0.0, "size_mod": 1.0, "atr_ratio": 1.0}

    atrs = []
    for i in range(1, len(recent_candles)):
        h = recent_candles[i]['high']
        l = recent_candles[i]['low']
        pc = recent_candles[i-1]['close']
        tr = max(h - l, abs(h - pc), abs(l - pc))
        atrs.append(tr)

    if len(atrs) < 20:
        return {"regime": "NORMAL", "hurdle_mod": 0.0, "size_mod": 1.0, "atr_ratio": 1.0}

    current_atr = sum(atrs[-20:]) / 20.0
    avg_atr = sum(atrs) / len(atrs)
    ratio = (current_atr / avg_atr) if avg_atr > 0 else 1.0

    if ratio >= 1.5:
        return {"regime": "EXPANSION", "hurdle_mod": -0.001, "size_mod": 1.2, "atr_ratio": ratio}
    if ratio <= 0.8:
        return {"regime": "CONTRACTION", "hurdle_mod": 0.001, "size_mod": 0.8, "atr_ratio": ratio}
    return {"regime": "NORMAL", "hurdle_mod": 0.0, "size_mod": 1.0, "atr_ratio": ratio}

