#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
strategy_common.py — Shared base for OKX strategy brains.
okx_venture_strategy.py, okx_major_strategy.py 등이
BaseStrategyBrain을 상속해 TA + DCA 로직을 공유한다.
"""
import os
import sys
import time
import json
import logging
import logging.handlers
import asyncio
from collections import defaultdict
import aiohttp
from datetime import datetime
from typing import List
import pytz
from dotenv import load_dotenv
import pandas as pd
import ccxt.async_support as ccxt_async

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

try:
    from webhook_spec import WebhookPayload, ActionType, SideType, sign_payload, WEBHOOK_SIGNATURE_HEADER
except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    sys.exit(1)

try:
    from bot_config import BotConfig
except ImportError:
    BotConfig = None


def setup_logger(name: str, log_file: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # root logger 전파 차단
    if not logger.handlers:
        fh = logging.handlers.RotatingFileHandler(
            os.path.join(BASE_DIR, log_file), encoding="utf-8",
            maxBytes=10*1024*1024, backupCount=3,
        )
        fh.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s'))
        logger.addHandler(fh)
        # [Fix #2] StreamHandler 제거: systemd가 stderr→같은 로그파일로 리다이렉트하므로
        # StreamHandler + FileHandler가 같은 파일에 2번 쓰는 것이 중복의 근본 원인이었음
    return logger


def calc_supertrend(df, period=10, multiplier=3.0):
    hl2 = (df['h'] + df['l']) / 2
    # [Fix] 올바른 3항목 True Range ATR (기존 H-L 단순화 → calc_atr()와 동일한 방식)
    h, l, c = df['h'], df['l'], df['c']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    final_upperband = hl2 + (multiplier * atr)
    final_lowerband = hl2 - (multiplier * atr)

    st_dir = pd.Series(1, index=df.index, dtype='int')
    st_val = pd.Series(0.0, index=df.index, dtype='float64')

    for i in range(period, len(df)):
        if df['c'].iloc[i] > final_upperband.iloc[i-1]:
            st_dir.iloc[i] = 1
        elif df['c'].iloc[i] < final_lowerband.iloc[i-1]:
            st_dir.iloc[i] = -1
        else:
            st_dir.iloc[i] = st_dir.iloc[i-1]
            if st_dir.iloc[i] == 1 and final_lowerband.iloc[i] < final_lowerband.iloc[i-1]:
                final_lowerband.iloc[i] = final_lowerband.iloc[i-1]
            if st_dir.iloc[i] == -1 and final_upperband.iloc[i] > final_upperband.iloc[i-1]:
                final_upperband.iloc[i] = final_upperband.iloc[i-1]

        if st_dir.iloc[i] == 1:
            st_val.iloc[i] = final_lowerband.iloc[i]
        else:
            st_val.iloc[i] = final_upperband.iloc[i]

    return st_dir, st_val


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calc_stoch_rsi(series, period=14, smooth_k=3, smooth_d=3):
    rsi = calc_rsi(series, period)
    stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    k = stoch_rsi.rolling(smooth_k).mean() * 100
    d = k.rolling(smooth_d).mean()
    return k, d


def calc_atr(df, period=14):
    """ATR(평균 진폭 범위) — Chandelier 트레일링 및 위험조정 모멘텀에 사용."""
    h, l, c = df['h'], df['l'], df['c']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def calc_adx(df, period=14):
    """
    ADX (Average Directional Index) — 추세 강도 지표.
    20 미만 = 방향성 없는 횡보장(휩소 구간)으로 해석.
    """
    h, l, c = df['h'], df['l'], df['c']
    up = h.diff()
    down = -l.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr
    denom = (plus_di + minus_di).where((plus_di + minus_di) != 0)
    dx = 100 * (plus_di - minus_di).abs() / denom
    adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    return adx.fillna(0.0)


class BaseStrategyBrain:
    """
    공통 전략 브레인 베이스 클래스.
    서브클래스는 get_target_symbols()와 클래스 속성(CONFIG)만 오버라이드하면 됨.
    """
    STRATEGY_NAME = "BaseStrategy"
    LOG_FILE = "base_strategy.log"
    LOGGER_NAME = "BaseStrategy"
    SUPERTREND_MULT_TIGHT = 2.0
    # [CRITICAL FIX] 계정별 마스터 포트 분리: 클론 디렉토리 간 파일 동기화 시
    # 하드코딩 포트가 함께 복사되어 타 계정으로 신호가 새는 회귀를 방지한다.
    # 각 시스템의 .env에서 MASTER_WEBHOOK_URL을 반드시 지정할 것 (8009/8010/8012).
    MASTER_WEBHOOK_URL = os.getenv("MASTER_WEBHOOK_URL", "http://localhost:8009/webhook")
    # [Review Fix] env 미설정 폴백은 타 계정 신호 유출 사고로 직결되므로 반드시 경고
    if os.getenv("MASTER_WEBHOOK_URL") is None:
        import logging as _logging
        _logging.getLogger().critical(
            "🚨 MASTER_WEBHOOK_URL 미설정 — 기본값 8009(Main)로 폴백합니다. "
            "클론 계정이라면 즉시 .env에 MASTER_WEBHOOK_URL을 지정하세요 (신호 유출 위험)."
        )
        print(
            "🚨 [CRITICAL] MASTER_WEBHOOK_URL 미설정 — 기본값 8009(Main) 폴백. "
            "클론 계정은 즉시 .env 지정 필요 (신호 유출 위험).",
            flush=True,
        )
    AUTO_TRADE_INTERVAL = 60.0
    STOCK_KEYWORDS = []
    BLACKLIST = []
    TIMEFRAME = "15m"
    TIMEFRAME_MINUTES = 15
    # [Fix] HTF(상위 타임프레임) 추세 필터 설정
    HTF_TIMEFRAME = "1h"     # 상위 타임프레임
    HTF_EMA_PERIOD = 50      # HTF EMA 기간
    # ── [K안] 롱 이중 게이트 (백테스트 +15.5K, MDD 41.7% 검증) ──
    # 롱 진입 조건: BTC > BTC 1h EMA50 (시장 게이트) AND 심볼 > 심볼 1h EMA50 (종목 게이트)
    LONG_DUAL_GATE = os.getenv("OKX_LONG_DUAL_GATE", "true").lower() == "true"
    SHORT_DUAL_GATE = os.getenv("OKX_SHORT_DUAL_GATE", "true").lower() == "true"
    HTF_SLOPE_THRESHOLD = 0.005  # HTF EMA 기울기 임계값 (0.5%)
    # [개선안 #1] Volume 확인 배수 — 서브클래스에서 오버라이드 가능
    # 추가 전략 파라미터 (AI 토너먼트 1등 Alpha_Trend 파라미터 적용)
    PROFIT_THRESHOLD = 1.03  # 3% 수익 구간부터 청산 고려
    VOL_CONFIRM_MULT = 1.0   # 거래량 급증 확인 배수 (저변동 장세 대응 1.2→1.0 완화)
    MIN_HOLD_CANDLES = 3     # 최소 보유 캔들 수
    EMA_PERIOD = 50          # 추세 필터 기간
    # [안전망 추가] 긴급 하드 스탑로스: 현물 기준 -5% (레버리지 10x 적용 시 PnL -50%) 
    HARD_STOP_LOSS_PCT = float(os.getenv("OKX_HARD_STOP_LOSS", "-0.30"))  # 손익비 개선: -50%→-30%
    SOFT_STOP_LOSS_PCT = float(os.getenv("OKX_SOFT_STOP_LOSS", "-0.12"))  # [Deprecated] ATR 스탑으로 대체됨 (하위호환용 유지)
    HARD_STOP_COOLDOWN_HOURS = int(os.getenv("OKX_HARD_STOP_COOLDOWN_HOURS", "24"))  # [Fix] 12→24h: 손절 후 재진입 왕복 차단
    # 신규 진입 차단: Master가 max_active_subpositions 초과 시 신호를 거부하므로
    # 각 전략 뇌도 로컬에서 동일 제한을 사전 체크 (중복 신호 억제)
    MAX_DCA_ENTRIES = int(os.getenv("OKX_MAX_DCA_ENTRIES", "8"))
    POSITION_PORTION = float(os.getenv("OKX_POSITION_PORTION", "0.20"))
    SCALE_OUT_EXITS = True
    MAX_OPEN_POSITIONS = int(os.getenv("OKX_BOT_MAX_POSITIONS", "15"))
    NEW_LISTING_SLOTS = int(os.getenv("OKX_NEW_LISTING_SLOTS", "3"))
    NEW_LISTING_DAYS = 60
    # [백테스트 검증] 재진입: 전량 청산 후 같은 방향 추세 유지 시 쿨다운 후 재진입
    # (Venture 15m에서 PF 1.15→1.20 개선 확인. 서브클래스에서만 활성화 권장)
    REENTRY_ENABLED = False
    REENTRY_COOLDOWN_CANDLES = int(os.getenv("OKX_REENTRY_COOLDOWN_CANDLES", "16"))  # [Fix] 4→16: 1h 재진입이 메이저 왕복(ADA/BNB 4회/5일) 유발
    
    # [포트폴리오 리밸런싱] 자본 배분 가중치 및 불타기(Pyramid) 비율
    PORTFOLIO_WEIGHT = 1.0
    PYRAMID_RATIO = 0.50  # [Walk-Forward OOS] 불타기 비율 0.50 (백테스트 FJ 최적)

    # [백테스트 검증] 트레일링 전량청산 시 즉시 반대방향 진입
    # Major 30m: PF 0.76→1.22 (흑자 전환) / Venture 15m: PF 1.23→1.13 (악화)
    # → 메이저처럼 박스권 성격 심볼에만 활성화
    FLIP_ON_TRAILING_CLOSE = False
    # [Fix] DCA 추가 진입 최소 간격 (캔들 수) — 매 캔들 물타기는 수수료 출혈
    DCA_MIN_CANDLES = 4
    # [검증] 물타기/피라미딩 백테스트(6~8월): 현행 조합 최악(-3.7K, MDD 83.7%) vs 둘다없음 최고(+18.1K, 48%)
    # DCA는 평균손실 -199→-274 확대, 피라미딩+20%가드는 승자 절단 → 기본 비활성 (env로 재활성 가능)
    # [Fix] 중복 선언 제거: SCALE_OUT_STEPS는 env 값 하나로 통일
    SCALE_OUT_STEPS = int(os.getenv("OKX_SCALE_OUT_STEPS", "3"))
    PYRAMIDING_ENABLED = os.getenv("OKX_PYRAMIDING", "false").lower() == "true"
    # ── 켈리 공식 포지션 사이징 ──
    # portion = Half-Kelly × f* / 평균손실률, f* = p - (1-p)/b
    KELLY_FRACTION = 0.50          # Half-Kelly (적극적 자본배치)
    KELLY_MIN_TRADES = 20          # 이 건수 미만이면 백테스트 사전값 사용
    KELLY_PRIOR_WR = 0.36          # 180일 Venture 15m 백테스트 승률
    KELLY_PRIOR_PL = 2.0           # 백테스트 평균손익비 (PF 1.15 기반)
    KELLY_PRIOR_LOSS = 0.20        # 평균 손실률 (마진 기준, 하드스탑 -30% 이내)
    PORTION_MIN = 0.15
    PORTION_MAX = 0.45
    # [긴급 패치] 24시간 거래대금 최소 기준 (잡코인 원천 차단) — 주식 토큰처럼
    # 거래량이 구조적으로 낮은 전략은 서브클스에서 오버라이드
    MIN_QUOTE_VOLUME = float(os.getenv("OKX_MIN_QUOTE_VOLUME", "10000000"))
    # ── [Fix] 포지션 사이징 개선: 최소 마진 하한선 ──
    # free_usdt 기반 사이징은 포지션 수 증가 시 사이즈가 0에 수렴하는 문제 해결
    # total_equity / MAX_OPEN_POSITIONS 기반 균등 분할 + 최소 마진 하한선
    MIN_POSITION_MARGIN = float(os.getenv("OKX_MIN_POSITION_MARGIN", "100"))
    # ── [Fix #1] 횡보장(Chop) 차단 필터 ──
    # BTC 1h ADX가 임계값 미만이면 신규 자본 투입(진입/DCA/불타기/재진입/플립) 전면 차단.
    # 08-21 꼭지 후 횡보장에서 롱 바이어스 + 물타기 + 재진입 휩소로 -27% 손실 → 재발 방지.
    # 청산/스탑/익절은 계속 동작 (기존 포지션 관리 유지).
    CHOP_FILTER_ENABLED = os.getenv("OKX_CHOP_FILTER", "true").lower() == "true"
    CHOP_ADX_THRESHOLD = float(os.getenv("OKX_CHOP_ADX", "20"))
    CHOP_ADX_BLOCK_THRESHOLD = float(os.getenv("OKX_CHOP_ADX_BLOCK", "20.0"))
    # 연속 배포: 하드 차단 없음, 사이즈만 호흡 (철칙: 거래가 없으면 기회도 없다)
    CHOP_FLOOR = float(os.getenv("OKX_CHOP_FLOOR", "0.15"))       # 배율 바닥
    CHOP_FLOOR_ADX = float(os.getenv("OKX_CHOP_FLOOR_ADX", "8"))  # 이 아래선 바닥 배율 고정
    # 신호 보너스 게이트: 심충보(저ADX)에서는 순행 신호(스퀴즈 등)가 오히려 함정 → 발동 하한
    SQUEEZE_MIN_ADX = float(os.getenv("OKX_SQUEEZE_MIN_ADX", "18"))
    MOM_MIN_ADX = float(os.getenv("OKX_MOM_MIN_ADX", "15"))
    BETA_MIN_ADX = float(os.getenv("OKX_BETA_MIN_ADX", "20"))
    # ── 섹터별 파라미터 (SECTOR_PARAMS) ──
    # 실적 데이터(08-16~) 기반: 밈 승률 66% 최고 / 메이저 42%(ETH류 체인 과다) /
    # 신규상장 순손실(-292, CAP -1.5k) → 섹터별 임계값·사이즈 차등화.
    SECTOR_MAJORS = frozenset({'BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'AVAX', 'LINK', 'DOT', 'BNB', 'TRX'})
    SECTOR_MEMES = frozenset({
        'DOGE', 'SHIB', 'PEPE', 'BOME', 'WIF', 'BONK', 'FLOKI', 'FARTCOIN', 'PUMP',
        'PEOPLE', 'MOODENG', 'PNUT', 'ACT', 'NEIRO', 'TURBO', 'MEW', 'POPCAT',
        'GIGA', 'BRETT', 'TRUMP', 'MELANIA', 'PENGU', 'AI16Z',
    })
    SECTOR_STOCKS = frozenset({
        'TSLA', 'NVDA', 'AAPL', 'AMZN', 'MSFT', 'META', 'GOOG', 'GOOGL', 'COIN', 'SPCX',
        'OPENAI', 'ANTHROPIC', 'RDDT', 'MU', 'SNDK', 'SOXL', 'SOXS', 'XAU', 'CL', 'SKHY',
        'KORU', 'CBRS', 'AEON', 'PLTR', 'AMD', 'INTC', 'QCOM', 'BABA', 'UBER', 'ABNB',
        'SNAP', 'MSTR', 'HOOD', 'RIVN', 'NIO', 'PYPL', 'SQ', 'SHOP', 'SPY', 'QQQ', 'IWM',
        'DIA', 'GLD', 'SLV', 'XAG', 'AXTI', 'CRCL', 'UNITREE', 'XIAOMI', 'LITE', 'UB',
        'KR200', 'ISRG', 'MRVL', 'SKUU', 'HOME',
    })
    # thr_long: 롱 진입 임계값 (숏 = +20 비대칭 유지)
    # size_mult: 진입 목표 마진 배수
    SECTOR_PARAMS = {
        'major':       {'thr_long': int(os.getenv("OKX_THR_MAJOR", "80")),  'size_mult': 1.0},
        'alt':         {'thr_long': int(os.getenv("OKX_THR_ALT", "70")),    'size_mult': 1.0},
        'meme':        {'thr_long': int(os.getenv("OKX_THR_MEME", "65")),   'size_mult': 1.0},
        'new_listing': {'thr_long': int(os.getenv("OKX_THR_NEW", "75")),
                        'size_mult': float(os.getenv("OKX_NEW_LISTING_SIZE_MULT", "0.75"))},
        'stock':       {'thr_long': 999, 'size_mult': 0.0},   # 원천 제외
    }
    # ── 왕복 필터 (Churn Filter) ──
    # 최근 N청산 승률 < 기준 & 순손실인 종목을 일정 시간 신규 진입 제외 (CAP류 출혈 차단).
    CHURN_LOOKBACK_CLOSES = int(os.getenv("OKX_CHURN_LOOKBACK", "6"))
    CHURN_MIN_WINRATE = float(os.getenv("OKX_CHURN_MIN_WINRATE", "0.40"))
    CHURN_COOLDOWN_HOURS = float(os.getenv("OKX_CHURN_COOLDOWN_HOURS", "12"))
    # ── Alpha Stack ──
    # ① ATR Chandelier 트레일링: 수익 구간에서 고점 − k×ATR 이탈 시 청산 (변동성 클수록 타이트)
    ATR_TRAILING_ENABLED = os.getenv("OKX_ATR_TRAILING", "true").lower() == "true"
    ATR_TRAIL_K = float(os.getenv("OKX_ATR_TRAIL_K", "3.0"))
    # [Fix] 가동 임계 0.20→0.06: 3x에서 마진 +20% = 가격 +6.7%는 사실상 미도달 → 트레일링이 비활성 상태였음. +6% ≈ 1R
    ATR_TRAIL_ARM_PNL = float(os.getenv("OKX_ATR_TRAIL_ARM_PNL", "0.06"))
    # ② 스퀴즈 브레이크아웃: BB 폭 압축 해제 + 방향성 돌파 → 점수 보너스
    SQUEEZE_SIGNAL_ENABLED = os.getenv("OKX_SQUEEZE_SIGNAL", "true").lower() == "true"
    SQUEEZE_BONUS = int(os.getenv("OKX_SQUEEZE_BONUS", "30"))
    # ③ 모멘텀 로테이션: 위험조정 모멘텀(ROC/ATR%) 강한 종목 점수 가중 (리더 집중)
    MOM_ROTATION_ENABLED = os.getenv("OKX_MOM_BONUS_ENABLED", "true").lower() == "true"
    MOM_ROC_LOOKBACK = int(os.getenv("OKX_MOM_LOOKBACK", "24"))
    MOM_RISK_ADJ_THRESHOLD = float(os.getenv("OKX_MOM_RA_THRESH", "1.5"))
    MOM_BONUS = int(os.getenv("OKX_MOM_BONUS", "15"))
    # ④ BTC 베타 래그: BTC 직전 봉 급등락 시 고베타 알트 동방향 보너스
    BTC_BETA_LAG_ENABLED = os.getenv("OKX_BTC_BETA_LAG", "true").lower() == "true"
    BTC_LAG_MOVE_PCT = float(os.getenv("OKX_BTC_MOVE_PCT", "0.8"))   # 단일 봉 ±0.8%
    BTC_LAG_BONUS = int(os.getenv("OKX_BETA_BONUS", "15"))
    # ⑤ 펀딩비 정렬: 극단 펀딩에서 유리한 방향 사이즈 확대 / 불리한 방향 축소
    FUNDING_ADJUST_ENABLED = os.getenv("OKX_FUNDING_ADJUST", "true").lower() == "true"
    FUNDING_EXTREME_POS = float(os.getenv("OKX_FUNDING_POS", "0.0015"))   # +0.15%
    FUNDING_EXTREME_NEG = float(os.getenv("OKX_FUNDING_NEG", "-0.0010"))  # -0.10%
    # ── [수익성 개선] 평균회귀(Mean-Reversion) 모드 — 레인지 시장용 ──
    # ADX < 20(횡보장)에서 RSI/Stoch RSI 기반 반대편 진입으로 수익 창출
    MEAN_REVERSION_ENABLED = os.getenv("OKX_MEAN_REVERSION", "true").lower() == "true"
    MEAN_REVERSION_RSI_OVERSOLD = int(os.getenv("OKX_MR_RSI_OVERSOLD", "30"))
    MEAN_REVERSION_RSI_OVERBOUGHT = int(os.getenv("OKX_MR_RSI_OVERBOUGHT", "70"))
    MEAN_REVERSION_SIZE_MULT = float(os.getenv("OKX_MR_SIZE_MULT", "0.5"))   # 레인지 시 사이즈 50%
    MEAN_REVERSION_ATR_K = float(os.getenv("OKX_MR_ATR_K", "1.5"))          # 더 타이트한 스탑
    MEAN_REVERSION_PROFIT_TARGET = float(os.getenv("OKX_MR_PROFIT_TARGET", "0.10"))  # 빠른 익절 (마진 +10%)
    MEAN_REVERSION_MAX_HOLD_HOURS = float(os.getenv("OKX_MR_MAX_HOLD", "6"))  # 최대 보유 시간
    # ── [Fix #2] 일손실 서킷 브레이커 ──
    # 당일 자산이 기준(일 시작 자산) 대비 임계값 이하로 하락하면 신규 진입 차단.
    # 회복(임계값의 절반 이상) 또는 다음 날(UTC) 자동 해제. 상태는 파일로 영속화(재시작 대비).
    CIRCUIT_BREAKER_ENABLED = os.getenv("OKX_CIRCUIT_BREAKER", "true").lower() == "true"
    CIRCUIT_BREAKER_ROE = float(os.getenv("OKX_CIRCUIT_BREAKER_ROE", "-6"))  # 일 시작 자산 대비 %
    CIRCUIT_BREAKER_COOLDOWN_HOURS = int(os.getenv("OKX_CIRCUIT_BREAKER_COOLDOWN_HOURS", "12"))  # [승부사] 48→12
    GLOBAL_MDD_LIMIT = float(os.getenv("OKX_GLOBAL_MDD_LIMIT", "-10"))
    # ── 포지션별 손실 한도 ──
    # 포지션 레벨 손실이 이 수준 이하로 떨어지면 추세/레짐 무관하게 무조건 청산 (빠른 차단).
    # 하드스탑(-30%)보다 빨리 발동해 횡보장 물타기 누적을 방지. 백테스트(F 변형) 검증 완료.
    POSITION_LOSS_LIMIT = float(os.getenv("OKX_POSITION_LOSS_LIMIT", "-0.15"))
    # ── [FJ 개선] 포지션 수명 제한 (Time Stop) ──
    # 진입 후 TIME_STOP_HOURS 경과 & 여전히 손실 중이면 강제 청산.
    # FJ Investment 카피봇 분석: 좀비 포지션 장기 방치가 자금 고갈의 핵심 원인.
    # 자금을 회수하여 새로운 추세 기회에 재배치. (기본 48시간)
    TIME_STOP_ENABLED = os.getenv("OKX_TIME_STOP", "true").lower() == "true"
    TIME_STOP_HOURS = float(os.getenv("OKX_TIME_STOP_HOURS", "48"))
    TIME_STOP_PROFIT_THRESHOLD = float(os.getenv("OKX_TIME_STOP_PROFIT", "0.02"))  # 마진 +2% 이하면 손실 취급
    # ── [Fix #3] 노출 스케일 ──
    # 심볼당 목표 마진에 곱하는 전역 배율. 정점 대비 -27% 손실 후 총 노출 축소(마진 사용률 84% → ~50% 목표).
    EXPOSURE_SCALE = float(os.getenv("OKX_EXPOSURE_SCALE", "0.6"))
    # ── [수정 4] 시즌별 모드 (Season Mode) ──
    # 시장 상태 자동 감지 → 전략 파라미터 자동 전환
    SEASON_MODE_ENABLED = os.getenv("OKX_SEASON_MODE", "true").lower() == "true"
    SEASON_MODE_STATE = "normal"  # normal | trend_up | trend_down | chop | crash
    SEASON_MODE_COOLDOWN = 3600  # 모드 전환 쿨다운 (1시간)
    # 추세 모드 파라미터
    SEASON_TREND_SIZE_MULT = 1.5    # 추세 시 마진 ×1.5
    SEASON_TREND_MAX_POS = int(os.getenv("OKX_SEASON_MAX_POS", "5"))       # 추세 시 최대 포지션 (기본 5개 집중)
    SEASON_CHOP_SIZE_MULT = 0.5     # 횡보 시 마진 ×0.5
    SEASON_CHOP_MAX_POS = min(4, int(os.getenv("OKX_SEASON_MAX_POS", "5")))         # 횡보 시 최대 포지션
    SEASON_CRASH_SIZE_MULT = 0.3    # 급락 시 마진 ×0.3
    SEASON_CRASH_MAX_POS = 2        # 급락 시 최대 포지션 2개
    SEASON_SIZE_MULT = float(os.getenv("OKX_SEASON_SIZE_MULT", "1.0"))
    SEASON_MAX_POS = int(os.getenv("OKX_SEASON_MAX_POS", "5"))

    # ── 수익성 향상: 컨빅션 사이징 ──
    # 진입 점수에 비례해 포지션 크기를 0.5x~2x로 조절. 고점수 셋업(강한 추세)에 몰빵 → 대수익 극대화.
    # 백테스트(08-18~23) 검증: 컨빙션+베어숏 조합 시 전체 +90 → +80,721.
    CONVICTION_SIZING_ENABLED = os.getenv("OKX_CONVICTION_SIZING", "true").lower() == "true"
    CONVICTION_MAX_MULT = float(os.getenv("OKX_CONVICTION_MAX_MULT", "2.0"))
    CONVICTION_MIN_MULT = float(os.getenv("OKX_CONVICTION_MIN_MULT", "0.5"))
    # ── 수익성 향상: 베어 숏 게이팅 ──
    # 불장(BTC>=EMA200)에서 숏 차단 → 불장 숏 손실 원천 방지. 베어장에서만 숏 허용.
    BEAR_SHORT_ENABLED = os.getenv("OKX_BEAR_SHORT", "true").lower() == "true"
    # ── [Fix] ATR 스탑 & 트레이드당 리스크 예산 ──
    # 실거래(08-30~09-03) 51건 승률 5.9%, PF 0.19: 고정 마진% 스탑(3x에서 가격 -2%)이 종목 ATR 안에 위치해 노이즈 손절 반복.
    # 스탑 = 진입가 ∓ ATR_STOP_K×ATR(확정 캔들), 사이즈 = 자산×RISK_PER_TRADE / 스탑거리 (레버리지·종목 무관 손실 금액 균일화)
    ATR_STOP_K = float(os.getenv("OKX_ATR_STOP_K", "3.5"))  # [Fix] 2.5→3.5: BTC 30m ATR 기준 1.3%→1.8% 스탑거리, 노이즈 손절 방지
    ATR_STOP_MIN_PCT = float(os.getenv("OKX_ATR_STOP_MIN_PCT", "0.02"))
    RISK_PER_TRADE = float(os.getenv("OKX_RISK_PER_TRADE", "0.005"))
    # ── [Fix] 진입 속도 제한 / 주말 차단 / 호가 깊이 가드 ──
    # 08-31(일) 하루 28건 진입 -469 USDT(총손실 71%), 라운드트립당 평균 14.9개 체결(얇은 호가 스윙) 재발 방지
    MAX_ENTRIES_PER_HOUR = int(os.getenv("OKX_MAX_ENTRIES_PER_HOUR", "2"))
    MAX_ENTRIES_PER_DAY = int(os.getenv("OKX_MAX_ENTRIES_PER_DAY", "6"))
    BLOCK_WEEKEND = os.getenv("OKX_BLOCK_WEEKEND", "true").lower() == "true"
    LIQUIDITY_DEPTH_MULT = float(os.getenv("OKX_LIQUIDITY_DEPTH_MULT", "2.0"))
    LIQUIDITY_MAX_SLIP = float(os.getenv("OKX_LIQUIDITY_MAX_SLIP", "0.001"))
    # ── [Fix] 숏 전면 스위치 (실거래 숏 22건 PF 0.09 → 엣지 확인 전까지 기본 비활성) ──
    SHORTS_ENABLED = os.getenv("OKX_SHORTS_ENABLED", "false").lower() == "true"
    # ── [수정 P1] 전략 건강도 서킷 브레이커 ──
    HEALTH_CB_ENABLED = os.getenv("OKX_HEALTH_CB", "true").lower() == "true"
    HEALTH_CB_LOOKBACK = int(os.getenv("OKX_HEALTH_CB_LOOKBACK", "20"))
    HEALTH_CB_MIN_PF = float(os.getenv("OKX_HEALTH_CB_MIN_PF", "0.8"))
    HEALTH_CB_MIN_WR = float(os.getenv("OKX_HEALTH_CB_MIN_WR", "0.15"))
    # ── [승부사] 동적 방향 추종 모드 ──
    # BTC 추세 방향에 따라 롱/숏 동적 전환: 상승=롱만, 하락=숏만, 횡보=차단
    AGGRESSIVE_MODE = os.getenv("OKX_AGGRESSIVE_MODE", "true").lower() == "true"
    DYNAMIC_DIRECTION = os.getenv("OKX_DYNAMIC_DIRECTION", "true").lower() == "true"
    BTC_DIRECTION_PERIOD = int(os.getenv("OKX_BTC_DIRECTION_PERIOD", "24"))  # 캔들 수
    BTC_DIRECTION_THRESHOLD = float(os.getenv("OKX_BTC_DIRECTION_THRESHOLD", "0.3"))  # EMA 기울기 %
    # ── [승부사] 서킷 브레이커 완화: -6% (수정 전 수준)
    CIRCUIT_BREAKER_ROE = float(os.getenv("OKX_CIRCUIT_BREAKER_ROE", "-6"))
    CIRCUIT_BREAKER_COOLDOWN_HOURS = int(os.getenv("OKX_CIRCUIT_BREAKER_COOLDOWN_HOURS", "12"))

    def __init__(self):
        self.session = None
        self.exchange = None
        self.auto_active_pos = {}
        self.dca_state = {}
        self.config = BotConfig() if BotConfig else None
        self.logger = setup_logger(self.LOGGER_NAME, self.LOG_FILE)
        self._long_regime_ok = True
        self._short_regime_ok = True  # [Fix] 숏 레짐 필터 (BTC 상승 시 알트숏 억제)
        self._htf_cache = {}  # [Fix] HTF EMA 캐시 {symbol: (timestamp, slope)}
        self._kelly_cache = None  # (timestamp, portion)
        # [Fix] 사이클 내 예약 마진 추적: 같은 사이클에 여러 진입 신호가 동시에
        # 발송되어 각자 동일 free 잔고 기준으로 주문 → 51008(증거금 부족) 연쇄 실패 방지
        self._reserved_margin = 0.0
        self._margin_reject_logged = False
        # [Fix #1/#2] 횡보장 필터 & 서킷 브레이커 상태
        self._chop_block = False
        self._deploy_scale = 1.0   # 단계적 배포 배율 (1.0=풀, 0.7=소프트존, 0.0=차단)
        self._deploy_state = None
        self._circuit_open = False
        # 섹터 분류 캐시 + 왕복 필터 상태
        self._listtime_cache = None       # {sym: ms}
        self._churn_blacklist = {}        # sym -> 제외 만료 epoch초
        self._churn_last_refresh = 0.0
        # [승부사] 동적 방향 추종 상태
        self._btc_direction = 'long'  # 'long' | 'short' | 'chop'
        self._btc_direction_ts = 0
        self._btc_ema_slope = 0.0
        self._btc_adx = 0.0
        # Alpha Stack 상태
        self._funding_cache = {}          # sym -> (ts, rate) 30분 TTL
        self._btc_move_15m = 0.0          # 직전 확정 봉 BTC 변동률% (베타 래그용)
        self._btc_above_ema50_1h = True   # [K안] 시장 게이트 상태 (첫 갱신 전 fail-open)
        self._cb_state = {}
        # [Fix] ATR 스탑 사이징용 최신 OHLCV 캠시 {symbol: df}, 진입 속도 예산 로그
        self._df_cache = {}
        self._entry_log = []
        self._entry_budget_logged = False
        # [Fix] auto_tune_config 파라미터 인스턴스 변수로 로드
        # daily_analyzer.py가 자동 튜닝한 값을 실제 전략 로직에 반영
        self._apply_auto_tune_params()

        # [Jev AI Integration] Sub-second orderbook prediction filter
        try:
            from jev.jev_signal_filter import JevSignalFilter
            self.jev_filter = JevSignalFilter()
            for h in self.logger.handlers:
                logging.getLogger("JevSignalFilter").addHandler(h)
                logging.getLogger("OKX_LOB_Feed").addHandler(h)
                logging.getLogger("Typesafe_Jev").addHandler(h)
            logging.getLogger("JevSignalFilter").setLevel(logging.INFO)
            logging.getLogger("OKX_LOB_Feed").setLevel(logging.INFO)
            logging.getLogger("Typesafe_Jev").setLevel(logging.INFO)
            self.logger.info("⚡ [Jev AI] 호가창 서브세컨드 예측 필터 및 웹소켓 피드 장착 완료")
        except Exception as e:
            self.logger.warning(f"⚠️ JevSignalFilter 초기화 예외: {e}")
            self.jev_filter = None

    def _apply_auto_tune_params(self):
        """
        [Fix] daily_analyzer.py가 생성한 auto_tune_config.json의 파라미터를
        인스턴스 변수로 로드해 실제 전략 로직에 반영.
        기존엔 bot_config에 값이 저장돼도 전략이 env 값만 읽어 자동 튜닝 효과가 없었음.

        [Fix v2] 서브클래스가 클래스 속성으로 명시 오버라이드한 값은 건드리지 않음.
        - 기본값과 같은 경우에만 auto_tune 값 적용 (서브클래스 의도 존중)
        - 잘못된 ×10 변환 제거 (HARD_SL_PCT는 현물 비율 그대로 사용)
        """
        if not self.config:
            return
        try:
            # 클래스 선언 기본값 (BaseStrategyBrain에서 env로 정의된 값)
            base_hard_sl = float(os.getenv("OKX_HARD_STOP_LOSS", "-0.30"))

            tuned_sl = self.config.okx_hard_sl_pct  # 양수 (예: 0.05 = 5%)
            if tuned_sl and tuned_sl > 0:
                tuned_val = -tuned_sl  # 음수로 변환 (-0.05)
                # [Fix] 서브클래스가 명시 오버라이드(클래스 속성)한 경우 건드리지 않음
                # → 현재 인스턴스 값이 base 기본값과 같을 때만 auto_tune 적용
                # [안전 가드] ATR 스탑의 정상 변동성을 보장하기 위해 base_hard_sl(-30%)보다 타이트하게 조여지지 않도록 보장
                if abs(self.HARD_STOP_LOSS_PCT - base_hard_sl) < 0.001:
                    self.HARD_STOP_LOSS_PCT = min(tuned_val, base_hard_sl)

            tuned_trail = self.config.okx_trailing_pct  # 양수 (예: 0.02 = 2%)
            if tuned_trail and tuned_trail > 0:
                # ATR_TRAIL_ARM_PNL도 클래스 기본값(0.20)과 같을 때만 적용
                base_arm = float(os.getenv("OKX_ATR_TRAIL_ARM_PNL", "0.06"))
                if abs(self.ATR_TRAIL_ARM_PNL - base_arm) < 0.01:
                    self.ATR_TRAIL_ARM_PNL = tuned_trail * 3  # 트레일링 비율 × 3배 (마진 기준)

            self.logger.info(
                f"⚙️ [AutoTune] 파라미터 확인: "
                f"HardSL={self.HARD_STOP_LOSS_PCT*100:.1f}%, "
                f"ATR_ARM={self.ATR_TRAIL_ARM_PNL*100:.1f}%"
            )
        except Exception as e:
            self.logger.warning(f"⚠️ [AutoTune] 파라미터 로드 실패(기본값 유지): {e}")
        self._load_survival_state()

    def _is_trading_hour_allowed(self) -> bool:
        """DEPRECATED: Bots now run 24/7 relying purely on technical indicators."""
        return True

    # ── [수정 1] 동적 레버리지 조절 ──
    # BTC 1h 변동성에 따라 레버리지를 자동 조절
    # 저변동 = 높은 레버리지, 고변동 = 낮은 레버리지
    _btc_vol_cache = {'ts': 0, 'vol': 0.5, 'lev': 2}

    async def _get_dynamic_leverage(self) -> int:
        """BTC 1h 변동성 기반 동적 레버리지 계산.
        변동성 < 0.3% → 3x, 0.3~0.5% → 2x, > 0.5% → 1x"""
        now = time.time()
        if now - self._btc_vol_cache['ts'] < 3600:  # 1시간 캐시
            return self._btc_vol_cache['lev']
        try:
            ohlcv = await self.exchange.fetch_ohlcv('BTC/USDT:USDT', '1h', limit=48)
            if ohlcv and len(ohlcv) >= 24:
                closes = [c[4] for c in ohlcv[-24:]]
                returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
                import statistics
                vol = statistics.stdev(returns) * 100  # %로 변환
                if vol < 0.3:
                    lev = 3
                elif vol < 0.5:
                    lev = 2
                else:
                    lev = 1
                self._btc_vol_cache = {'ts': now, 'vol': vol, 'lev': lev}
                self.logger.info(f"📊 [동적레버리지] BTC 변동성 {vol:.2f}% → {lev}x")
                return lev
        except Exception:
            pass
        return self._btc_vol_cache['lev'] if self._btc_vol_cache['ts'] > 0 else getattr(self, 'STRATEGY_LEVERAGE', 2)

    # ── [Fix] ATR 스탑 / 진입 예산 / 호가 깊이 / 전략 건강도 헬퍼 ──
    def _stop_distance_pct(self, df) -> float:
        """확정 캔들(-2) ATR 기준 스탑 거리(가격 비율). 최소 ATR_STOP_MIN_PCT.
        고정 마진% 스탑(3x에서 가격 -2%)은 알트 30m ATR(1.5~2.5%) 안에 위치해 노이즈로 손절됐음."""
        try:
            atr = float(df['atr'].iloc[-2])
            px = float(df['c'].iloc[-2])
            if px > 0 and atr > 0:
                return max(self.ATR_STOP_MIN_PCT, self.ATR_STOP_K * atr / px)
        except Exception:
            pass
        return self.ATR_STOP_MIN_PCT

    def _entry_budget_ok(self) -> bool:
        """신규 자본 투입 속도 제한: 시간당/일일 상한 + 주말(UTC 토·일) 차단.
        08-31(일) 하루 28건 진입 -469 USDT(총손실 71%) 재발 방지."""
        now = time.time()
        self._entry_log = [t for t in self._entry_log if now - t < 86400]
        if self.BLOCK_WEEKEND and datetime.utcnow().weekday() >= 5:
            return False
        if sum(1 for t in self._entry_log if now - t < 3600) >= self.MAX_ENTRIES_PER_HOUR:
            return False
        return len(self._entry_log) < self.MAX_ENTRIES_PER_DAY

    async def _liquidity_ok(self, symbol: str, side, notional: float) -> bool:
        """최우선호가 ±LIQUIDITY_MAX_SLIP 안에 내 노셔널의 LIQUIDITY_DEPTH_MULT배 깊이가 있어야 진입.
        (실거래: 라운드트립당 평균 14.9개 체결 = 시장가가 얇은 호가창을 여러 단계 스윙)"""
        try:
            ob = await self.exchange.fetch_order_book(symbol, limit=20)
            is_buy = side == SideType.BUY
            levels = ob.get('asks' if is_buy else 'bids') or []
            if not levels:
                return False
            best = float(levels[0][0])
            cap = best * (1 + self.LIQUIDITY_MAX_SLIP) if is_buy else best * (1 - self.LIQUIDITY_MAX_SLIP)
            cs = float((self.exchange.markets.get(symbol) or {}).get('contractSize') or 1)
            depth = sum(float(item[0]) * float(item[1]) * cs for item in levels
                        if len(item) >= 2 and (float(item[0]) <= cap if is_buy else float(item[0]) >= cap))
            need = notional * self.LIQUIDITY_DEPTH_MULT
            if depth < need:
                self.logger.info(
                    f"💧 [Liquidity] {symbol} 호가 깊이 부족 ({depth:.0f} < {need:.0f} USDT "
                    f"@ ±{self.LIQUIDITY_MAX_SLIP*100:.2f}%) — 진입 스킵"
                )
                return False
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ [Liquidity] {symbol} 호가 조회 실패 — 진입 스킵: {e}")
            try:
                from utils_telegram import send_telegram_alert
                send_telegram_alert(f"⚠️ [{self.STRATEGY_NAME}] {symbol} 호가 조회 예외 — 진입 스킵: {e}")
            except Exception:
                pass
            return False

    def _recent_roundtrip_pnls(self, n: int, since_ts: float = 0.0) -> list:
        """state/trades.jsonl에서 (symbol, side)별 가중평균가로 재구성한 청산 PnL(가격 비율, 시간순) 최근 n건.
        since_ts 이후 청산만 포함 (Health CB 재발동 시 새 표본만 평가)."""
        path = os.path.join(BASE_DIR, "state", "trades.jsonl")
        out = []
        if not os.path.exists(path):
            return out
        pos = {}
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    px, amt = float(r.get("price") or 0), float(r.get("amount") or 0)
                    if not px:
                        continue
                    sk = "short" if r["side"] in ("SELL", "CLOSE_SHORT") else "long"
                    key = (r["symbol"], sk)
                    qty, avg = pos.get(key, (0.0, 0.0))
                    if r["side"] in ("BUY", "SELL"):
                        nq = qty + amt
                        pos[key] = (nq, (avg * qty + px * amt) / nq if nq else 0.0)
                    else:
                        cq = qty if amt == 0 else min(amt, qty)
                        if cq > 0 and avg > 0 and float(r.get("ts", 0)) > since_ts:
                            out.append((px - avg) / avg * (1 if sk == "long" else -1))
                        nq = qty - cq
                        pos[key] = (max(nq, 0.0), avg if nq > 1e-9 else 0.0)
        except Exception as e:
            self.logger.warning(f"⚠️ 라운드트립 PnL 계산 실패: {e}")
        return out[-n:]

    def _get_dynamic_blacklist(self) -> List[str]:
        """Merge hardcoded BLACKLIST with auto-tuned blacklisted_symbols from OKX."""
        bl = list(self.BLACKLIST)
        if self.config:
            bl.extend(self.config.blacklisted_symbols_okx)
        return bl

    def _is_new_listing(self, symbol: str) -> bool:
        """Check if the symbol was listed within the last NEW_LISTING_DAYS."""
        if not self.exchange or not hasattr(self.exchange, 'markets') or not self.exchange.markets:
            return False
        m_info = self.exchange.markets.get(symbol, {})
        info = m_info.get('info', {})
        list_time_str = info.get('listTime', '0')
        if not list_time_str:
            return False
        try:
            list_time = int(list_time_str)
            now_ms = time.time() * 1000
            if (now_ms - list_time) < (self.NEW_LISTING_DAYS * 24 * 60 * 60 * 1000):
                return True
        except Exception:
            pass
        return False

    def _compute_kelly_stats(self):
        """state/trades.jsonl 90일 체결 기록 → (승률, 손익비, 평균손실률, 샘플수)"""
        path = os.path.join(BASE_DIR, "state", "trades.jsonl")
        pnls = []
        if os.path.exists(path):
            pos = {}  # (symbol, side) -> [qty, avg]
            cutoff = time.time() - 90 * 86400
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if r.get("ts", 0) < cutoff or not r.get("price"):
                            continue
                        side_key = "short" if r["side"] in ("SELL", "CLOSE_SHORT") else "long"
                        key = (r["symbol"], side_key)
                        qty, avg = pos.get(key, (0.0, 0.0))
                        px, amt = float(r["price"]), float(r["amount"])
                        if r["side"] in ("BUY", "SELL"):
                            nq = qty + amt
                            pos[key] = (nq, (avg * qty + px * amt) / nq if nq else 0.0)
                        else:
                            cq = qty if amt == 0 else min(amt, qty)
                            if cq > 0 and avg > 0:
                                sgn = 1 if side_key == "long" else -1
                                lev = int(os.getenv("OKX_LEVERAGE", "10"))
                                pnls.append((px - avg) / avg * sgn * lev)
                            nq = qty - cq
                            pos[key] = (max(nq, 0.0), avg if nq > 1e-9 else 0.0)
            except Exception as e:
                self.logger.warning(f"⚠️ 켈리 통계 계산 실패: {e}")
        wins = [x for x in pnls if x > 0]
        losses = [x for x in pnls if x <= 0]
        if len(pnls) >= self.KELLY_MIN_TRADES and losses:
            p = len(wins) / len(pnls)
            avg_win = sum(wins) / len(wins) if wins else 0.0
            avg_loss = abs(sum(losses) / len(losses))
            b = avg_win / avg_loss if avg_loss > 0 else self.KELLY_PRIOR_PL
            return p, b, max(avg_loss, 0.05), len(pnls)
        return self.KELLY_PRIOR_WR, self.KELLY_PRIOR_PL, self.KELLY_PRIOR_LOSS, len(pnls)

    def get_kelly_portion(self) -> float:
        """쿼터 켈리 기준 심볼당 총 비중 (10분 캐시)"""
        now = time.time()
        if self._kelly_cache and now - self._kelly_cache[0] < 600:
            return self._kelly_cache[1]
        p, b, avg_loss, n = self._compute_kelly_stats()
        f_star = max(0.0, p - (1 - p) / b) if b > 0 else 0.0
        portion = self.KELLY_FRACTION * f_star / avg_loss
        portion = max(self.PORTION_MIN, min(self.PORTION_MAX, portion))
        src = "실거래" if n >= self.KELLY_MIN_TRADES else f"백테스트사전값(실거래 {n}건)"
        self.logger.info(
            f"🎯 [Kelly] f*={f_star:.3f} (p={p:.2f}, b={b:.2f}, avgLoss={avg_loss:.2f}, {src}) "
            f"→ 심볼당 비중 {portion*100:.1f}%"
        )
        self._kelly_cache = (now, portion)
        return portion

    # ── DCA 상태 영속화 (재시작 시 폭주/기억상실 방지) ──
    def _dca_state_path(self) -> str:
        return os.path.join(BASE_DIR, "state", f"dca_state_{self.STRATEGY_NAME}.json")

    def _load_dca_state(self):
        try:
            with open(self._dca_state_path(), encoding="utf-8") as f:
                self.dca_state = json.load(f)
            self.logger.info(f"💾 DCA 상태 복원: {len(self.dca_state)}개 심볼")
        except FileNotFoundError:
            pass
        except Exception as e:
            self.logger.warning(f"⚠️ DCA 상태 복원 실패: {e}")

    def _save_dca_state(self):
        try:
            # [Fix] 상태 파일 비대화 방지: 30일 이상 활동 없는 심볼 프루닝
            cutoff = time.time() * 1000 - 30 * 86400 * 1000
            stale = [
                s for s, d in self.dca_state.items()
                if max(d.get('last_entry_t', 0), d.get('last_exit_t', 0), d.get('last_close_t', 0)) < cutoff
            ]
            for s in stale:
                del self.dca_state[s]

            path = self._dca_state_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.dca_state, f)
            os.replace(tmp, path)
        except Exception as e:
            self.logger.warning(f"⚠️ DCA 상태 저장 실패: {e}")

    def _dca_ready(self, dca: dict, t_curr) -> bool:
        """DCA 추가 진입 가능 여부: 마지막 진입 후 DCA_MIN_CANDLES 캔들 경과"""
        last = dca.get('last_entry_t', 0)
        interval_ms = self.DCA_MIN_CANDLES * self.TIMEFRAME_MINUTES * 60 * 1000
        return (t_curr - last) >= interval_ms

    # ── 섹터 분류 & 파라미터 ──
    def _symbol_sector(self, symbol: str) -> str:
        """심볼을 major/alt/meme/new_listing/stock 로 분류 (listTime 60일 기준)."""
        base = symbol.split('/')[0]
        if base in self.SECTOR_STOCKS:
            return 'stock'
        if base in self.SECTOR_MAJORS:
            return 'major'
        lt = self._get_list_time(symbol)
        if lt and (time.time() * 1000 - lt) < self.NEW_LISTING_DAYS * 86400 * 1000:
            return 'new_listing'
        if base in self.SECTOR_MEMES:
            return 'meme'
        return 'alt'

    def _sector_params(self, symbol: str) -> dict:
        sec = self._symbol_sector(symbol)
        return self.SECTOR_PARAMS.get(sec, {'thr_long': 70, 'size_mult': 1.0})

    async def _long_dual_gate_ok(self, symbol: str) -> bool:
        """
        [K안] 롱 이중 게이트: BTC > 1h EMA50 (시장) AND 심볼 > 1h EMA50 (종목).
        백테스트(6/23~8/26): 게이트 없음 +9.3K → 이중게이트 +15.5K, MDD 53.5→41.7%.
        """
        # [Fix] Falling Knife 방지 (최근 24h 하락률 -10% 이상 시 롱 금지)
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            if ticker and ticker.get('percentage'):
                if float(ticker['percentage']) <= -10.0:
                    now_ts = time.time()
                    if not hasattr(self, '_fk_log_ts'):
                        self._fk_log_ts = {}
                    if now_ts - self._fk_log_ts.get(symbol, 0) > 300:
                        self.logger.warning(f"🚨 [Falling Knife] {symbol} 24h {ticker['percentage']}% 급락 중 — 롱 진입 강제 차단")
                        self._fk_log_ts[symbol] = now_ts
                    return False
        except Exception as e:
            self.logger.warning(f"⚠️ {symbol} ticker 조회 실패 (Falling Knife 체크 패스): {e}")

        if not self.LONG_DUAL_GATE:
            return True
            
        sec_name = self._symbol_sector(symbol)
        exempt_btc_gate = sec_name in ('meme', 'new_listing', 'alt')
        
        if not exempt_btc_gate and not getattr(self, '_btc_above_ema50_1h', True):
            return False
            
        htf = await self._check_htf_trend(symbol)
        return bool(htf.get('above_ema50', True))

    async def _short_dual_gate_ok(self, symbol: str) -> bool:
        """
        [K안 추가] 숏 이중 게이트: BTC < 1h EMA50 AND 심볼 < 1h EMA50.
        """
        if not self.SHORT_DUAL_GATE:
            return True
            
        sec_name = self._symbol_sector(symbol)
        exempt_btc_gate = sec_name in ('meme', 'new_listing', 'alt')
        
        if not exempt_btc_gate and getattr(self, '_btc_above_ema50_1h', True):
            return False
            
        htf = await self._check_htf_trend(symbol)
        return not bool(htf.get('above_ema50', True))

    async def _get_funding_rate(self, symbol: str):
        """[Alpha ⑤] 펀딩비 캐시 조회 (30분 TTL — 펀딩은 8h 주기라 충분)."""
        now = time.time()
        hit = self._funding_cache.get(symbol)
        if hit and now - hit[0] < 1800:
            return hit[1]
        try:
            f = await self.exchange.fetch_funding_rate(symbol)
            rate = float(f.get('fundingRate') or 0)
            self._funding_cache[symbol] = (now, rate)
            return rate
        except Exception:
            return None

    def _get_list_time(self, symbol: str):
        """상장시각(ms) 캐시 조회. exchange.markets 로딩 전이면 None."""
        if self._listtime_cache is None:
            self._listtime_cache = {}
            try:
                for sym, m in (getattr(self, 'exchange', None) and self.exchange.markets or {}).items():
                    if not m.get('swap'):
                        continue
                    try:
                        self._listtime_cache[sym] = int((m.get('info') or {}).get('listTime') or 0)
                    except Exception:
                        self._listtime_cache[sym] = 0
            except Exception:
                pass
        return self._listtime_cache.get(symbol) or 0

    def _refresh_churn_blacklist(self):
        """
        왕복 필터: trades.jsonl 최근 청산 성적 기준.
        최근 N청산 승률 < 기준 & 순손실 → CHURN_COOLDOWN_HOURS 동안 신규 진입 제외.
        10분 캐시, 변화 시에만 로깅.
        """
        now = time.time()
        if now - self._churn_last_refresh < 600:
            return
        self._churn_last_refresh = now
        try:
            path = os.path.join(BASE_DIR, "state", "trades.jsonl")
            if not os.path.exists(path):
                return
            cutoff = now - 7 * 86400
            markets = getattr(self, 'exchange', None) and getattr(self.exchange, 'markets', {}) or {}
            pos = {}
            closes = defaultdict(list)
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    ts = r.get('ts', 0)
                    if ts < cutoff:
                        continue
                    px, amt = float(r.get('price') or 0), float(r.get('amount') or 0)
                    if not px:
                        continue
                    m = markets.get(r['symbol']) or {}
                    cs = float(m.get('contractSize') or 1)
                    side = r['side']
                    sk = "short" if side in ("SELL", "CLOSE_SHORT") else "long"
                    key = (r['symbol'], sk)
                    qty, avg = pos.get(key, (0.0, 0.0))
                    if side in ("BUY", "SELL"):
                        nq = qty + amt
                        pos[key] = (nq, (avg * qty + px * amt) / nq if nq else 0.0)
                    else:
                        cq = qty if amt == 0 else min(amt, qty)
                        if cq > 0 and avg > 0:
                            sgn = 1 if sk == "long" else -1
                            closes[r['symbol']].append((ts, (px - avg) * cq * cs * sgn))
                        nq = qty - cq
                        pos[key] = (max(nq, 0.0), avg if nq > 1e-9 else 0.0)

            new_bl = {}
            for sym, cl in closes.items():
                base = sym.split('/')[0]
                if base in self.SECTOR_MAJORS or base in self.SECTOR_STOCKS:
                    continue  # 메이저(추세캡처 본업)/주식(원천제외)는 왕복필터 대상 아님
                recent = cl[-self.CHURN_LOOKBACK_CLOSES:]
                if len(recent) < 4:
                    continue  # 표본 부족 → 제외 안 함
                wr = sum(1 for _, p in recent if p > 0) / len(recent)
                netp = sum(p for _, p in recent)
                if wr < self.CHURN_MIN_WINRATE and netp < 0:
                    new_bl[sym] = now + self.CHURN_COOLDOWN_HOURS * 3600

            active_old = {s for s, u in self._churn_blacklist.items() if u > now}
            added = set(new_bl) - active_old
            removed = active_old - set(new_bl)
            for s in sorted(added):
                self.logger.warning(
                    f"🚫 [왕복필터] {s} 최근 청산 승률 저조 — "
                    f"{self.CHURN_COOLDOWN_HOURS:.0f}시간 신규 진입 제외"
                )
            for s in sorted(removed):
                self.logger.info(f"✅ [왕복필터] {s} 제외 해제")
            self._churn_blacklist = new_bl
        except Exception as e:
            self.logger.warning(f"⚠️ [{self.STRATEGY_NAME}] 왕복필터 갱신 실패(기존 유지): {e}")

    def _churn_blocked(self, symbol: str) -> bool:
        return time.time() < self._churn_blacklist.get(symbol, 0)

    def _get_dynamic_portion(self, symbol: str) -> float:
        """[레거시 호환] 섹터별 사이즈 배수 반영한 켈리 포션."""
        base = self.get_kelly_portion()
        mult = self._sector_params(symbol).get('size_mult', 1.0)
        return max(0.05, base * mult)

    def _calc_target_margin(self, free_usdt: float, total_equity: float, entry_type: str = "new") -> float:
        """
        [Fix] 균등 분할 기반 심볼당 목표 마진 계산.
        기존 free_usdt × portion 방식 → total_equity / MAX_OPEN_POSITIONS 기반.

        entry_type:
          - "new": 신규 진입 (목표 마진의 50%)
          - "dca": DCA 물타기 (목표 마진의 25%)
          - "pyramid": 승자 불타기 (목표 마진의 35%)
          - "flip": 반대방향 전환 (목표 마진의 30%)
          - "reentry": 재진입 (목표 마진의 30%)
        """
        # 심볼당 목표 마진 = (총 자산 / 최대 포지션 수) * 포트폴리오 가중치 * 노출 스케일
        base_margin = (total_equity / max(1, self.MAX_OPEN_POSITIONS)) * self.PORTFOLIO_WEIGHT * self.EXPOSURE_SCALE

        # 진입 유형별 비율
        type_ratios = {
            "new": 0.50,       # 신규 진입: 목표의 50%
            "dca": 0.25,       # DCA 물타기: 목표의 25%
            "pyramid": self.PYRAMID_RATIO,   # 승자 불타기: 동적 비율 적용
            "flip": 0.30,      # 반대방향 전환: 목표의 30%
            "reentry": 0.30,   # 재진입: 목표의 30%
            "mean_rev": 0.30,  # 평균회귀: 목표의 30%
        }
        ratio = type_ratios.get(entry_type, 0.50)
        target_margin = base_margin * ratio

        # [수정 3] 서바이벌 모드: 연속 손실 시 마진 50% 축소
        if self._survival_state.get('active') and entry_type == "new":
            target_margin *= 0.5
            self.logger.info(f"🛡️ [서바이벌] 마진 50% 축소 ({target_margin:.1f} USDT)")

        # [수정 4] 시즌 모드: 시장 상태별 마진 조절
        if hasattr(self, '_season_mode_cached') and self._season_mode_cached != "normal":
            target_margin *= self.SEASON_SIZE_MULT
            self.logger.info(f"🌸 [시즌] {self._season_mode_cached} 모드 — 마진 ×{self.SEASON_SIZE_MULT:.1f}")

        # Free 잔고 안전 가드: free의 95% 이내
        target_margin = min(target_margin, free_usdt * 0.95)

        # 최소 마진 하한선 적용
        if target_margin < self.MIN_POSITION_MARGIN:
            # free가 충분하면 최소 마진으로 상향, 아니면 진입 거부(0 반환)
            if free_usdt >= self.MIN_POSITION_MARGIN * 1.1:
                target_margin = self.MIN_POSITION_MARGIN
            else:
                return 0.0  # 자금 부족으로 진입 불가

        return target_margin

    async def get_target_symbols(self):
        import traceback
        for attempt in range(3):
            try:
                tickers = await self.exchange.fetch_tickers()
                markets = await self.exchange.load_markets()
                dynamic_blacklist = self._get_dynamic_blacklist()
                data = []
                for s, t in tickers.items():
                    if s in markets and markets[s].get('swap') and 'USDT' in s:
                        raw_vol = t.get('quoteVolume')
                        if raw_vol is None:
                            # [Fix] OKX swap 티커에서 quoteVolume이 None으로 날아오는 경우
                            # info.volCcy24h(기준통화 거래량) × last 가격으로 USDT 환산
                            try:
                                vol_ccy = float(t.get('info', {}).get('volCcy24h') or 0)
                                last = float(t.get('last') or 0)
                                raw_vol = vol_ccy * last if vol_ccy and last else None
                            except (ValueError, TypeError):
                                raw_vol = None
                        try:
                            vol = float(raw_vol) if raw_vol is not None else 0.0
                        except (ValueError, TypeError):
                            vol = 0.0
                        # [긴급 패치] 24시간 거래대금 MIN_QUOTE_VOLUME 미만인 잡코인 원천 차단
                        if vol >= self.MIN_QUOTE_VOLUME:
                            # [왕복필터] 저승률 왕복 종목 신규 선정 제외
                            # (메이저 제외: 단기 승률 낮아도 추세 캡처가 본업 → SOL 사례)
                            if self._churn_blocked(s) and s.split('/')[0] not in self.SECTOR_MAJORS:
                                continue
                            if self._symbol_matches(s, t, markets) and not any(b in s for b in dynamic_blacklist):
                                data.append({'symbol': s, 'vol': vol})
                if not data:
                    return []
                df = pd.DataFrame(data).sort_values(by='vol', ascending=False)
                return df['symbol'].tolist()
            except Exception as e:
                if attempt < 2:
                    self.logger.warning(f"⚠️ [{self.STRATEGY_NAME}] 심볼 로드 재시도 ({attempt+1}/3): {e}")
                    await asyncio.sleep(2)
                else:
                    self.logger.error(f"❌ [{self.STRATEGY_NAME}] 심볼 로드 최종 실패: {e}\n{traceback.format_exc()}")
                    return []

    async def init_session(self):
        self.session = aiohttp.ClientSession()
        self.exchange = ccxt_async.okx({
            "apiKey": os.getenv("OKX_API_KEY", ""),
            "secret": os.getenv("OKX_SECRET", "") or os.getenv("OKX_API_SECRET", ""),
            "password": os.getenv("OKX_PASSPHRASE", "") or os.getenv("OKX_PASSWORD", ""),
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })

    async def close_session(self):
        if self.session:
            await self.session.close()
        if self.exchange:
            await self.exchange.close()

    async def send_webhook(self, side: SideType, symbol: str, qty: float, leverage: int = None, stop_pct: float = None,
                           order_type: str = "MARKET", target_price: float = None, jev_score: float = None, is_simulation: bool = False):
        if leverage is None:
            # [수정 1] 동적 레버리지 사용
            lev = await self._get_dynamic_leverage()
        else:
            lev = leverage
        payload = WebhookPayload(
            action=ActionType.EXEC,
            side=side,
            symbol=symbol,
            qty=qty,
            price=target_price,
            signal_strength="STRONG",
            leverage=lev,
            stop_pct=stop_pct,  # [Fix] 브레인 ATR 스탑 거리 → 실행봇 거래소 SL 정합
            order_type=order_type,
            target_price=target_price,
            jev_score=jev_score,
            is_simulation=is_simulation,
        )
        json_data = json.loads(payload.to_json())
        json_data["market"] = "okx_swap"
        body = json.dumps(json_data)
        headers = {"Content-Type": "application/json", WEBHOOK_SIGNATURE_HEADER: sign_payload(body)}
        try:
            async with self.session.post(
                self.MASTER_WEBHOOK_URL,
                data=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status in [200, 201]:
                    self.logger.info(f"✅ Webhook 발송 완료: {side.value} {qty} {symbol}")
                    # [Fix] 거래 알림 (기본 비활성: .env에서 TELEGRAM_TRADE_ALERTS=true로 활성화)
                    if os.getenv("TELEGRAM_TRADE_ALERTS", "false").lower() == "true":
                        try:
                            from utils_telegram import send_telegram_alert
                            send_telegram_alert(f"📈 [{self.STRATEGY_NAME}] {side.value} {qty} {symbol}")
                        except Exception:
                            pass
                else:
                    self.logger.error(f"❌ Webhook 발송 실패: {resp.status} - {await resp.text()}")
        except Exception as e:
            self.logger.error(f"❌ Webhook 전송 예외: {e}")

    def is_us_market_active(self):
        """DEPRECATED: OKX stock tokens trade 24/7. Use _is_trading_hour_allowed() instead.
        This is kept for backward compat but always returns True since crypto never sleeps."""
        return True

    def _symbol_matches(self, symbol: str, ticker_data: dict, markets: dict) -> bool:
        raise NotImplementedError

    async def check_auto_logic(self, symbol):
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, self.TIMEFRAME, limit=300)
            if not ohlcv or len(ohlcv) < 200:
                return
            df = pd.DataFrame(ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])

            st_d_loose, st_v_loose = calc_supertrend(df, 10, 3.0)
            st_d_tight, st_v_tight = calc_supertrend(df, 10, self.SUPERTREND_MULT_TIGHT)
            df['st_d_loose'] = st_d_loose
            df['st_v_loose'] = st_v_loose
            df['st_d_tight'] = st_d_tight
            df['st_v_tight'] = st_v_tight

            df['vol_ma'] = df['v'].rolling(20).mean()
            k, d = calc_stoch_rsi(df['c'], 14, 3, 3)
            df['stoch_k'] = k
            df['stoch_d'] = d
            df['vol_ma'] = df['v'].rolling(20).mean()
            df['ema_target'] = df['c'].ewm(span=self.EMA_PERIOD, adjust=False).mean()
            # [Alpha] ATR (Chandelier 트레일링·위험조정 모멘텀용)
            df['atr'] = calc_atr(df, 14)
            # [Alpha] BB 스퀴즈 감지용 밴드
            if self.SQUEEZE_SIGNAL_ENABLED:
                bb_mid = df['c'].rolling(20).mean()
                bb_std = df['c'].rolling(20).std()
                df['bb_upper'] = bb_mid + 2.0 * bb_std
                df['bb_lower'] = bb_mid - 2.0 * bb_std
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_mid.where(bb_mid != 0)

            # [Fix] 신호는 확정 캔들(-2)로 판정, 가격 참조(PnL/스탑/트레일링)는 진행 캔들(-1) 종가.
            # 기존엔 진행 중 캔들로 Supertrend 전환/청산을 판정해 봉 중간 일시 돌파에 진입하고 되돌림에 청산됨
            # (실거래 중앙 보유시간 1.8h, 4분·11분 라운드트립 다수). 백테스트(종가 기준)와도 불일치했음
            live = df.iloc[-1]
            prev, curr = df.iloc[-3], df.iloc[-2]
            t_curr = curr['t']
            px_now = float(live['c'])
            self._df_cache[symbol] = df
            # [Jev AI First-Class Engine] Jev가 1번 판단권자: Jev 활성화 시 레거시 횡보장 차단(_chop_block) 및 EMA50 이중게이트를 바이패스
            jev_active = bool(hasattr(self, 'jev_filter') and self.jev_filter and self.jev_filter.is_enabled)
            long_blocked = (self._chop_block if not jev_active else False) or self._circuit_open
            short_blocked = (self._chop_block if not jev_active else False)
            mr_long_blocked = self._circuit_open  # 평균회귀는 횡보장 차단 없이 진입 가능
            mr_short_blocked = self._circuit_open
            if self._circuit_open:
                if getattr(self, '_short_regime_ok', False) or jev_active:
                    short_blocked = False
                else:
                    short_blocked = True
            # [K안] 롱/숏 이중 게이트 (Jev 활성화 시 Jev 실시간 호가 수급이 1번이므로 바이패스)
            dual_gate = True if jev_active else await self._long_dual_gate_ok(symbol)
            short_dual_gate = True if jev_active else await self._short_dual_gate_ok(symbol)
            dca = self.dca_state.setdefault(symbol, {'entry_count': 0, 'exit_count': 0, 'last_entry_t': 0, 'last_exit_t': 0, 'first_entry_t': 0, 'max_pnl_pct': 0.0})

            pos_long = self.auto_active_pos.get((symbol, 'long'))
            has_long = pos_long is not None
            avg_price_long = pos_long['avgPrice'] if has_long else 0

            pos_short = self.auto_active_pos.get((symbol, 'short'))
            has_short = pos_short is not None
            avg_price_short = pos_short['avgPrice'] if has_short else 0

            # [수정 3] 서바이벌 모드 체크
            _survival_active = self._check_survival_mode()

            # [수정 4] 시즌 모드 감지
            self._season_mode_cached = await self._detect_season_mode()

            # [Fix] 방향 전환 잔여 상태 정리: 보유 방향과 기록 방향이 다르면 상태 리셋
            held_side = 'long' if has_long else ('short' if has_short else None)
            if held_side:
                if dca.get('side') and dca['side'] != held_side:
                    self.logger.info(
                        f"🧹 [DCA State] {symbol} 방향 전환 감지 ({dca['side']}→{held_side}) — 잔여 상태 리셋"
                    )
                    dca.clear()
                    dca.update({
                        'entry_count': 0, 'exit_count': 0, 'last_entry_t': 0,
                        'last_exit_t': 0, 'first_entry_t': 0, 'max_pnl_pct': 0.0,
                        'side': held_side,
                    })
                elif not dca.get('side'):
                    dca['side'] = held_side  # 레거시 상태 백필
            else:
                dca.pop('stop_pct', None)  # [Fix] 포지션 없음 → 이전 포지션의 ATR 스탑 폐기
                # [개선 #1] 거래소 실포지션 자동 동기화 (Auto-Reconcile): 실포지션 0인데 dca 기록이 남아있는 고스트 상태 클린
                last_entry_t = dca.get('last_entry_t', 0)
                if dca.get('entry_count', 0) > 0 and (t_curr - last_entry_t) > 60000:
                    self.logger.info(
                        f"🧹 [Auto-Reconcile] {symbol} 거래소 실포지션 없음(0) 확인 — 잔여 DCA 고스트 상태 자동 초기화"
                    )
                    dca.clear()
                    dca.update({
                        'entry_count': 0, 'exit_count': 0, 'last_entry_t': 0,
                        'last_exit_t': 0, 'first_entry_t': 0, 'max_pnl_pct': 0.0,
                    })

            # [버그 수정] curr['v']는 미완성 캔들이라 거래량이 턱없이 부족할 때가 많습니다. 
            # 따라서 직전 완성 캔들(prev['v'])에서 이미 거래량이 터졌거나, 
            # 현재 캔들에서 벌써 거래량 조건을 만족한 경우를 모두 인정합니다.
            # [Fix] Jev 활성화 시 캔들 거래량 지표(vol_cond)로 실시간 호가 수급을 차단하지 않음
            vol_cond = True if jev_active else (curr['v'] > prev['vol_ma'] * self.VOL_CONFIRM_MULT)

            # 기초 시그널 판단
            is_long_breakout = prev['st_d_loose'] == -1 and curr['st_d_loose'] == 1
            is_short_breakout = prev['st_d_loose'] == 1 and curr['st_d_loose'] == -1

            is_long_pullback = curr['st_d_loose'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
            is_short_pullback = curr['st_d_loose'] == -1 and prev['stoch_k'] > 80 and curr['stoch_k'] <= 80

            is_ema_trend_up = curr['c'] > curr['ema_target']
            is_ema_trend_down = curr['c'] < curr['ema_target']

            # [Fix #1] 추가 점수 항목: Supertrend 추세 지속 & StochRSI 모멘텀
            is_long_trend_cont = curr['st_d_loose'] == 1 and prev['st_d_loose'] == 1  # 추세 지속
            is_short_trend_cont = curr['st_d_loose'] == -1 and prev['st_d_loose'] == -1
            is_long_momentum = curr['stoch_k'] > curr['stoch_d'] and curr['stoch_k'] < 80  # 모멘텀 상승 중
            is_short_momentum = curr['stoch_k'] < curr['stoch_d'] and curr['stoch_k'] > 20  # 모멘텀 하락 중

            # 롱 진입 점수 계산 (기존 EMA 60 + Breakout/Pullback 40 + 추세지속 20 + 모멘텀 20 = 최대 140)
            long_score = 0
            if is_ema_trend_up: long_score += 50
            if is_long_breakout: long_score += 40
            if is_long_pullback: long_score += 30
            if is_long_trend_cont: long_score += 20
            if is_long_momentum: long_score += 20
            
            # 숏 진입 점수 계산
            short_score = 0
            if is_ema_trend_down: short_score += 50
            if is_short_breakout: short_score += 40
            if is_short_pullback: short_score += 30
            if is_short_trend_cont: short_score += 20
            if is_short_momentum: short_score += 20

            # [Alpha ③] 모멘텀 로테이션: 위험조정 모멘텀 강한 리더 종목 가중 (ADX≥15에서만)
            _adx_now = getattr(self, '_current_adx', 99.0)
            if self.MOM_ROTATION_ENABLED and _adx_now >= self.MOM_MIN_ADX and len(df) > self.MOM_ROC_LOOKBACK + 2:
                roc_ref = float(df['c'].iloc[-2 - self.MOM_ROC_LOOKBACK])  # [Fix] 확정 캔들 기준
                atr_pct = max(float(curr['atr']) / curr['c'], 1e-9)
                if roc_ref > 0 and atr_pct > 0:
                    ra_mom = ((curr['c'] - roc_ref) / roc_ref) / atr_pct
                    if ra_mom >= self.MOM_RISK_ADJ_THRESHOLD:
                        long_score += self.MOM_BONUS
                    elif ra_mom <= -self.MOM_RISK_ADJ_THRESHOLD:
                        short_score += self.MOM_BONUS

            # [Alpha ②] 스퀴즈 브레이크아웃: BB 폭 압축 해제 + 방향성 돌파 (ADX≥18 — 심충보에선 함정 신호)
            if self.SQUEEZE_SIGNAL_ENABLED and _adx_now >= self.SQUEEZE_MIN_ADX and 'bb_width' in df.columns and len(df) > 105:
                w_ref = df['bb_width'].iloc[-101:-4].min()  # [Fix] 확정 캔들 기준으로 한 칸 이동
                prev_squeeze = float(df['bb_width'].iloc[-5]) <= w_ref * 1.1
                if prev_squeeze and pd.notna(curr['bb_upper']):
                    if curr['c'] > curr['bb_upper']:
                        long_score += self.SQUEEZE_BONUS
                    elif curr['c'] < curr['bb_lower']:
                        short_score += self.SQUEEZE_BONUS

            # [Alpha ④] BTC 베타 래그: BTC 급등락 → 고베타 섹터 동방향 가중 (ADX≥20 추세 맥락 필요)
            if self.BTC_BETA_LAG_ENABLED and _adx_now >= self.BETA_MIN_ADX and abs(self._btc_move_15m) >= self.BTC_LAG_MOVE_PCT:
                _sec_name = self._symbol_sector(symbol)
                if _sec_name in ('alt', 'meme', 'new_listing'):
                    if self._btc_move_15m > 0:
                        long_score += self.BTC_LAG_BONUS
                    else:
                        short_score += self.BTC_LAG_BONUS

            # ── [수익성 개선] 평균회귀(Mean-Reversion) 점수 — 레인지 시장용 ──
            # ADX < CHOP_ADX_THRESHOLD이면 추세 신호가 아닌 반대편 RSI 신호로 진입
            is_mean_rev_long_sig = False
            is_mean_rev_short_sig = False
            if self.MEAN_REVERSION_ENABLED and _adx_now < self.CHOP_ADX_THRESHOLD:
                rsi_val = float(curr['rsi']) if 'rsi' in df.columns else 50.0
                stoch_k_val = float(curr['stoch_k']) if 'stoch_k' in df.columns else 50.0
                # RSI 과매수/과매도 기반 반대편 진입
                if rsi_val < self.MEAN_REVERSION_RSI_OVERSOLD:
                    is_mean_rev_long_sig = True
                elif rsi_val > self.MEAN_REVERSION_RSI_OVERBOUGHT:
                    is_mean_rev_short_sig = True
                # Stoch RSI 확인 (과매수/과매도 구간에서 반전 신호)
                if is_mean_rev_long_sig and stoch_k_val > 80:
                    is_mean_rev_long_sig = False  # Stoch RSI도 과매수면 신호 무시
                if is_mean_rev_short_sig and stoch_k_val < 20:
                    is_mean_rev_short_sig = False  # Stoch RSI도 과매도면 신호 무시
                # 볼린저 밴드 확인: 하한 터치 시 롱, 상한 터치 시 숏
                if 'bb_lower' in df.columns and is_mean_rev_long_sig:
                    if curr['c'] > curr.get('bb_lower', 0) * 1.001:
                        pass  # 이미 밴드 안에 있으면 신호 유지
                if 'bb_upper' in df.columns and is_mean_rev_short_sig:
                    if curr['c'] < curr.get('bb_upper', 0) * 0.999:
                        pass  # 이미 밴드 안에 있으면 신호 유지

            # [Fix] 비대칭 임계값: 롱 70, 숏 90 (숏은 구조적으로 위험하므로 엄격)
            # [섹터별 임계값] 종목 섹터에 따라 진입 점수 기준 차등 적용
            _sec_p = self._sector_params(symbol)
            ENTRY_THRESHOLD_LONG = _sec_p.get('thr_long', 70)
            ENTRY_THRESHOLD_SHORT = ENTRY_THRESHOLD_LONG + 30  # 숏 비대칭 강화 (기존 +20→+30): 숏 PF 0.95 적자 대응
            
            _sec_name = self._symbol_sector(symbol)
            is_independent = _sec_name in ('meme', 'new_listing', 'alt')
            # [Jev AI 결합] Jev가 1번 판단권자: Jev 활성화 시 매크로 EMA200/EMA50 레짐으로 롱/숏을 사전 차단하지 않음
            jev_active = bool(hasattr(self, 'jev_filter') and self.jev_filter and self.jev_filter.is_enabled)
            regime_ok = True if (is_independent or jev_active) else getattr(self, '_long_regime_ok', True)
            regime_short_ok = getattr(self, '_short_regime_ok', True) or jev_active
            dual_gate_short_ok = short_dual_gate or jev_active

            # [Jev AI First-Class Engine] 실시간 호가창 LOB 불균형 확인
            lob = self.jev_filter.lob_feed.get_lob(symbol) if (jev_active and hasattr(self.jev_filter, 'lob_feed')) else None
            lob_imb = lob.imbalance if (lob and not lob.is_stale) else 0.0

            if jev_active:
                # 1번 판단권자: Jev AI
                # 미세 모멘텀(점수 30점 이상) 또는 호가 매수 우위(imbalance >= +0.10) 시 즉시 Jev 호가창 AI로 이관하여 최종 진입 결정
                is_long_sig = (long_score >= 30 or lob_imb >= 0.10) and not self._circuit_open
                is_short_sig = (short_score >= 30 or lob_imb <= -0.10) and self.SHORTS_ENABLED and not self._circuit_open
            else:
                entry_long_thr = ENTRY_THRESHOLD_LONG
                entry_short_thr = ENTRY_THRESHOLD_SHORT
                is_long_sig = (long_score >= entry_long_thr) and vol_cond and regime_ok and dual_gate
                is_short_sig = (short_score >= entry_short_thr) and vol_cond and regime_short_ok and dual_gate_short_ok
                if self.BEAR_SHORT_ENABLED and is_short_sig and getattr(self, '_long_regime_ok', True):
                    is_short_sig = False
                if not self.SHORTS_ENABLED:
                    is_short_sig = False
            
            default_leverage_long = int(os.getenv("OKX_LEVERAGE", "10"))
            default_leverage_short = int(os.getenv("OKX_SHORT_LEVERAGE", str(default_leverage_long)))
            leverage_long = getattr(self, 'STRATEGY_LEVERAGE', default_leverage_long)
            leverage_short = getattr(self, 'STRATEGY_LEVERAGE_SHORT', default_leverage_short)
            
            pnl_pct_long = 0
            is_hard_stop_long = False
            take_profit_long_sig = False
            if has_long and avg_price_long > 0:
                pnl_pct_long = ((px_now - avg_price_long) / avg_price_long) * leverage_long
                spot_pnl_pct_long = (px_now - avg_price_long) / avg_price_long  # [Fix] 레버리지 제외 가격 비율 — Breakeven 비교용
                if pnl_pct_long > dca['max_pnl_pct']:
                    dca['max_pnl_pct'] = pnl_pct_long
                
                is_profit = px_now > avg_price_long * self.PROFIT_THRESHOLD
                st_v_long = curr['st_v_tight'] if is_profit else curr['st_v_loose']
                st_d_long = curr['st_d_tight'] if is_profit else curr['st_d_loose']
                close_long_sig = st_d_long == -1 or curr['c'] < st_v_long
                force_close_long = False

                # [Alpha ①] ATR Chandelier 트레일링: 수익 구간에서 고점 − k×ATR 이탈 시 청산
                # (철칙 2: 변동성 클수록 ATR이 커져 선이 넓어지는 대신, 가격 이탈 즉시 반응)
                if self.ATR_TRAILING_ENABLED:
                    dca['highest_px'] = max(dca.get('highest_px') or avg_price_long, float(curr['h']), float(live['h']))
                    _chand_l = dca['highest_px'] - self.ATR_TRAIL_K * float(curr['atr'])
                    if dca.get('max_pnl_pct', 0.0) >= self.ATR_TRAIL_ARM_PNL and px_now < _chand_l:
                        self.logger.info(
                            f"🎯 [Chandelier] 롱 트레일링 청산: {symbol} "
                            f"(고점 {dca['highest_px']:.6g} − {self.ATR_TRAIL_K}×ATR, "
                            f"최고수익 {dca.get('max_pnl_pct',0)*100:.0f}%)"
                        )
                        force_close_long = True

                # [Fix] ATR 스탑: 진입 시점 ATR로 고정한 stop_pct(가격 비율). SOFT_STOP/POSITION_LOSS_LIMIT 고정 마진% 스탑은
                # 종목 변동성을 무시해 노이즈 손절 반복(승률 5.9%) → 대체. HARD_STOP은 최후 방어선으로만 유지.
                stop_pct_long = dca.get('stop_pct') or self._stop_distance_pct(df)
                dca['stop_pct'] = stop_pct_long
                stop_px_long = avg_price_long * (1 - stop_pct_long)
                if pnl_pct_long <= self.HARD_STOP_LOSS_PCT:
                    force_close_long = True
                    is_hard_stop_long = True
                elif px_now <= stop_px_long:
                    self.logger.warning(
                        f"🚨 [ATR STOP] 롱 손절 {symbol} px={px_now:.6g} stop={stop_px_long:.6g} "
                        f"(ATR스탑 {stop_pct_long*100:.2f}%, ROE {pnl_pct_long*100:.1f}%)"
                    )
                    force_close_long = True
                    is_hard_stop_long = True  # 쿨다운 적용을 위해 hard stop 취급
                elif dca['max_pnl_pct'] >= 1.00 and pnl_pct_long <= dca['max_pnl_pct'] - 0.30:
                    # [Fix] +100% 이상: 고점 대비 30%p回落 시 전량 청산 (기존엔 +20%까지回落해야 청산)
                    force_close_long = True
                elif dca['max_pnl_pct'] >= 0.50 and pnl_pct_long <= dca['max_pnl_pct'] * 0.6:
                    # [Fix] +50% 이상: 최고 수익의 60% 유지선
                    force_close_long = True
                elif dca['max_pnl_pct'] >= 0.40 and pnl_pct_long <= 0.30:
                    force_close_long = True
                elif dca['max_pnl_pct'] >= 0.20 and pnl_pct_long <= 0.05:
                    force_close_long = True
                elif dca.get('exit_count', 0) >= 1 and spot_pnl_pct_long <= -0.005:
                    # [Fix] 레버리지 제외 가격 비율 비교: 진입가 대비 -0.5% 하락 시 본전 보호 청산
                    # (기존: ROE 0.5%로 비교 → 20x에서 가격 0.025% 하락에도 발동하는 버그)
                    self.logger.info(f"🛡️ [Breakeven Stop] 롱 1차 익절 후 본전 보호 전량 청산 (Spot: {spot_pnl_pct_long*100:+.2f}%, ROE: {pnl_pct_long*100:+.2f}%): {symbol}")
                    force_close_long = True
                # ── [수익성 개선] 평균회귀 모드 전용 익절/스탑 — 빠른 턴오버 ──
                if dca.get('mr_mode') and self.MEAN_REVERSION_ENABLED:
                    # 빠른 익절: 마진 +10% 전량 청산
                    if pnl_pct_long >= self.MEAN_REVERSION_PROFIT_TARGET:
                        force_close_long = True
                        self.logger.info(f"🎯 [MR 익절] 롱 {symbol} (목표 도달: +{pnl_pct_long*100:.1f}%)")
                    # 최대 보유 시간 초과 시 청산
                    elif dca.get('mr_entry_t'):
                        mr_hours = (t_curr - dca['mr_entry_t']) / (3600 * 1000)
                        if mr_hours >= self.MEAN_REVERSION_MAX_HOLD_HOURS:
                            force_close_long = True
                            self.logger.warning(f"⏰ [MR Time Stop] 롱 {symbol} (보유 {mr_hours:.1f}h ≥ {self.MEAN_REVERSION_MAX_HOLD_HOURS}h)")
                    # 타이트한 트레일링: 고점 대비 5%p回落
                    elif dca['max_pnl_pct'] >= 0.05 and pnl_pct_long <= dca['max_pnl_pct'] - 0.05:
                        force_close_long = True
                        self.logger.info(f"🎯 [MR 트레일링] 롱 {symbol} (고점 대비 5%p回落)")
                elif not dca.get('mr_mode'):
                    # ── [FJ 개선] Time Stop: 장기 보유 손실 포지션 자동 청산 ──
                    if not force_close_long and self.TIME_STOP_ENABLED and dca.get('first_entry_t'):
                        hours_held = (t_curr - dca['first_entry_t']) / (3600 * 1000)
                        if hours_held >= self.TIME_STOP_HOURS and pnl_pct_long <= self.TIME_STOP_PROFIT_THRESHOLD:
                            self.logger.warning(
                                f"⏰ [Time Stop] 롱 수명 초과 강제 청산: {symbol} "
                                f"(보유 {hours_held:.1f}h ≥ {self.TIME_STOP_HOURS}h, PnL: {pnl_pct_long*100:+.2f}%)"
                            )
                            force_close_long = True
                            is_hard_stop_long = True
                    elif pnl_pct_long >= 0.25 and dca['exit_count'] == 0:
                        take_profit_long_sig = True
                    elif pnl_pct_long >= 0.50 and dca['exit_count'] == 1:
                        take_profit_long_sig = True
                    elif pnl_pct_long >= 0.80 and dca['exit_count'] == 2:
                        take_profit_long_sig = True
                # ── [Winner Pyramiding] 추세 승자 롱 불타기 ──
                # [Walk-Forward OOS] 1차 트리거 0.40→0.15(마진 +15%), 2차 불타기 비활성(=FJ 스타일 단일 불타기)
                if self.PYRAMIDING_ENABLED and pnl_pct_long >= 0.15 and dca.get('pyramid_count', 0) == 0 and self._dca_ready(dca, t_curr) and not long_blocked:
                    if is_ema_trend_up and curr['st_d_loose'] == 1:  # [Fix] Series 비교 → 스칼라 비교
                        self.logger.info(f"🔥 [Winner Pyramiding 1차 불타기] 롱 {symbol} (PnL: +{pnl_pct_long*100:.1f}%)")
                        await self.execute_auto_entry(symbol, SideType.BUY, entry_type="pyramid")
                        dca['pyramid_count'] = 1
                        dca['last_entry_t'] = t_curr
                elif self.PYRAMIDING_ENABLED and pnl_pct_long >= 99.0 and dca.get('pyramid_count', 0) == 1 and self._dca_ready(dca, t_curr) and not long_blocked:
                    if is_ema_trend_up and is_long_momentum:
                        self.logger.info(f"🚀 [Winner Pyramiding 2차 불타기] 롱 {symbol} (PnL: +{pnl_pct_long*100:.1f}%)")
                        await self.execute_auto_entry(symbol, SideType.BUY, entry_type="pyramid")
                        dca['pyramid_count'] = 2
                        dca['last_entry_t'] = t_curr

                if dca.get('pyramid_count', 0) >= 1 and pnl_pct_long < 0.20:
                    force_close_long = True

                if dca['exit_count'] > 0 and px_now < avg_price_long:
                    # [개선안 #3] 최소 보유 캔들 수 체크 — 진입 직후 whipsaw 방지
                    candles_held = (t_curr - dca.get('first_entry_t', t_curr)) / (self.TIMEFRAME_MINUTES * 60 * 1000)
                    if candles_held >= self.MIN_HOLD_CANDLES:
                        force_close_long = True
            else:
                close_long_sig = False
                force_close_long = False

            take_profit_short_sig = False
            is_hard_stop_short = False  # [Fix] 미초기화 시 방어청산 경로에서 UnboundLocalError 발생
            if has_short and avg_price_short > 0:
                pnl_pct_short = ((avg_price_short - px_now) / avg_price_short) * leverage_short
                spot_pnl_pct_short = (avg_price_short - px_now) / avg_price_short  # [Fix] 레버리지 제외 가격 비율 — Breakeven 비교용
                if pnl_pct_short > dca['max_pnl_pct']:
                    dca['max_pnl_pct'] = pnl_pct_short
                
                is_profit = px_now < avg_price_short * (2.0 - self.PROFIT_THRESHOLD)
                st_v_short = curr['st_v_tight'] if is_profit else curr['st_v_loose']
                st_d_short = curr['st_d_tight'] if is_profit else curr['st_d_loose']
                close_short_sig = st_d_short == 1 or curr['c'] > st_v_short
                force_close_short = False

                # [Alpha ①] ATR Chandelier 트레일링 (숏): 저점 + k×ATR 상향 돌파 시 청산
                if self.ATR_TRAILING_ENABLED:
                    dca['lowest_px'] = min(dca.get('lowest_px') or avg_price_short, float(curr['l']), float(live['l']))
                    _chand_s = dca['lowest_px'] + self.ATR_TRAIL_K * float(curr['atr'])
                    if dca.get('max_pnl_pct', 0.0) >= self.ATR_TRAIL_ARM_PNL and px_now > _chand_s:
                        self.logger.info(
                            f"🎯 [Chandelier] 숏 트레일링 청산: {symbol} "
                            f"(저점 {dca['lowest_px']:.6g} + {self.ATR_TRAIL_K}×ATR, "
                            f"최고수익 {dca.get('max_pnl_pct',0)*100:.0f}%)"
                        )
                        force_close_short = True

                # [Fix] ATR 스탑 (숏): 진입가 × (1 + stop_pct) 상향 돌파 시 손절. HARD_STOP은 최후 방어선.
                stop_pct_short = dca.get('stop_pct') or self._stop_distance_pct(df)
                dca['stop_pct'] = stop_pct_short
                stop_px_short = avg_price_short * (1 + stop_pct_short)
                if pnl_pct_short <= self.HARD_STOP_LOSS_PCT:
                    force_close_short = True
                    is_hard_stop_short = True
                elif px_now >= stop_px_short:
                    self.logger.warning(
                        f"🚨 [ATR STOP] 숏 손절 {symbol} px={px_now:.6g} stop={stop_px_short:.6g} "
                        f"(ATR스탑 {stop_pct_short*100:.2f}%, ROE {pnl_pct_short*100:.1f}%)"
                    )
                    force_close_short = True
                    is_hard_stop_short = True  # 쿨다운 적용을 위해 hard stop 취급
                elif dca['max_pnl_pct'] >= 1.00 and pnl_pct_short <= dca['max_pnl_pct'] - 0.30:
                    force_close_short = True
                elif dca['max_pnl_pct'] >= 0.50 and pnl_pct_short <= dca['max_pnl_pct'] * 0.6:
                    force_close_short = True
                elif dca['max_pnl_pct'] >= 0.40 and pnl_pct_short <= 0.30:
                    force_close_short = True
                elif dca['max_pnl_pct'] >= 0.20 and pnl_pct_short <= 0.05:
                    force_close_short = True
                elif dca.get('exit_count', 0) >= 1 and spot_pnl_pct_short <= -0.005:
                    # [Fix] 레버리지 제외 가격 비율 비교: 진입가 대비 -0.5% 역행 시 본전 보호 청산
                    # (기존: ROE 0.5%로 비교 → 20x에서 가격 0.025% 역행에도 발동하는 버그)
                    self.logger.info(f"🛡️ [Breakeven Stop] 숏 1차 익절 후 본전 보호 전량 청산 (Spot: {spot_pnl_pct_short*100:+.2f}%, ROE: {pnl_pct_short*100:+.2f}%): {symbol}")
                    force_close_short = True
                # ── [수익성 개선] 평균회귀 모드 전용 익절/스탑 — 빠른 턴오버 (숏) ──
                if dca.get('mr_mode') and self.MEAN_REVERSION_ENABLED:
                    if pnl_pct_short >= self.MEAN_REVERSION_PROFIT_TARGET:
                        force_close_short = True
                        self.logger.info(f"🎯 [MR 익절] 숏 {symbol} (목표 도달: +{pnl_pct_short*100:.1f}%)")
                    elif dca.get('mr_entry_t'):
                        mr_hours = (t_curr - dca['mr_entry_t']) / (3600 * 1000)
                        if mr_hours >= self.MEAN_REVERSION_MAX_HOLD_HOURS:
                            force_close_short = True
                            self.logger.warning(f"⏰ [MR Time Stop] 숏 {symbol} (보유 {mr_hours:.1f}h ≥ {self.MEAN_REVERSION_MAX_HOLD_HOURS}h)")
                    elif dca['max_pnl_pct'] >= 0.05 and pnl_pct_short <= dca['max_pnl_pct'] - 0.05:
                        force_close_short = True
                        self.logger.info(f"🎯 [MR 트레일링] 숏 {symbol} (고점 대비 5%p回落)")
                elif not dca.get('mr_mode'):
                    # ── [FJ 개선] Time Stop: 장기 보유 손실 포지션 자동 청산 ──
                    if not force_close_short and self.TIME_STOP_ENABLED and dca.get('first_entry_t'):
                        hours_held = (t_curr - dca['first_entry_t']) / (3600 * 1000)
                        if hours_held >= self.TIME_STOP_HOURS and pnl_pct_short <= self.TIME_STOP_PROFIT_THRESHOLD:
                            self.logger.warning(
                                f"⏰ [Time Stop] 숏 수명 초과 강제 청산: {symbol} "
                                f"(보유 {hours_held:.1f}h ≥ {self.TIME_STOP_HOURS}h, PnL: {pnl_pct_short*100:+.2f}%)"
                            )
                            force_close_short = True
                            is_hard_stop_short = True
                    elif pnl_pct_short >= 0.25 and dca['exit_count'] == 0:
                        take_profit_short_sig = True
                    elif pnl_pct_short >= 0.50 and dca['exit_count'] == 1:
                        take_profit_short_sig = True
                    elif pnl_pct_short >= 0.80 and dca['exit_count'] == 2:
                        take_profit_short_sig = True
                # ── [Winner Pyramiding] 추세 승자 숏 불타기 ──
                # [Walk-Forward OOS] 1차 트리거 0.40→0.15(마진 +15%), 2차 불타기 비활성(=FJ 스타일 단일 불타기)
                if self.PYRAMIDING_ENABLED and pnl_pct_short >= 0.15 and dca.get('pyramid_count', 0) == 0 and self._dca_ready(dca, t_curr) and not short_blocked:
                    if is_ema_trend_down and curr['st_d_loose'] == -1:  # [Fix] Series 비교 → 스칼라 비교
                        self.logger.info(f"📉 [Winner Pyramiding 숏 1차 불타기] {symbol} (PnL: +{pnl_pct_short*100:.1f}%)")
                        await self.execute_auto_entry(symbol, SideType.SELL, entry_type="pyramid")
                        dca['pyramid_count'] = 1
                        dca['last_entry_t'] = t_curr
                elif self.PYRAMIDING_ENABLED and pnl_pct_short >= 99.0 and dca.get('pyramid_count', 0) == 1 and self._dca_ready(dca, t_curr) and not short_blocked:
                    if is_ema_trend_down and is_short_momentum:
                        self.logger.info(f"🚀 [Winner Pyramiding 숏 2차 불타기] {symbol} (PnL: +{pnl_pct_short*100:.1f}%)")
                        await self.execute_auto_entry(symbol, SideType.SELL, entry_type="pyramid")
                        dca['pyramid_count'] = 2
                        dca['last_entry_t'] = t_curr

                if dca.get('pyramid_count', 0) >= 1 and pnl_pct_short < 0.20:
                    force_close_short = True

                if dca['exit_count'] > 0 and px_now > avg_price_short:
                    # [개선안 #3] 최소 보유 캔들 수 체크
                    candles_held = (t_curr - dca.get('first_entry_t', t_curr)) / (self.TIMEFRAME_MINUTES * 60 * 1000)
                    if candles_held >= self.MIN_HOLD_CANDLES:
                        force_close_short = True
            else:
                close_short_sig = False
                force_close_short = False

            if has_long:
                if force_close_long and dca.get('last_exit_t') != t_curr:
                    if is_hard_stop_long:
                        self.logger.warning(f"🚨 [HARD STOP] 롱 전량 긴급 손절 (손실률: {pnl_pct_long*100:.2f}%): {symbol}")
                    else:
                        self.logger.info(f"💨 [Breakeven Stop] 롱 전량 방어 청산: {symbol}")
                    await self.send_webhook(SideType.CLOSE_LONG, symbol, 0)
                    # [Flip] 트레일링/방어 청산 시 즉시 숏 진입 (하드스탑 제외)
                    flipped = False
                    if self.FLIP_ON_TRAILING_CLOSE and not is_hard_stop_long and not long_blocked and not dca.get('mr_mode') and self.SHORTS_ENABLED:
                        self.logger.info(f"🔄 [FLIP] 롱 청산 → 숏 반대진입: {symbol} (최고수익: {dca['max_pnl_pct']*100:.0f}%)")
                        await self.execute_auto_entry(symbol, SideType.SELL, entry_type="flip")
                        flipped = True
                    dca['exit_count'] = self.SCALE_OUT_STEPS
                    dca['entry_count'] = 0
                    dca['pyramid_count'] = 0
                    dca['last_exit_t'] = t_curr
                    dca['max_pnl_pct'] = 0.0
                    dca['last_close_t'] = t_curr
                    dca['last_close_side'] = 'long'
                    if is_hard_stop_long:
                        dca['last_hard_stop_t'] = t_curr
                    if flipped:
                        # [Fix] 플립으로 열린 신규 숏 포지션 관리 가능하도록 상태 리셋
                        dca.update({
                            'entry_count': 1, 'exit_count': 0, 'pyramid_count': 0,
                            'max_pnl_pct': 0.0, 'first_entry_t': t_curr,
                            'last_entry_t': t_curr, 'last_close_t': 0, 'side': 'short',
                        })
                elif close_long_sig or take_profit_long_sig:
                    if dca['exit_count'] < self.SCALE_OUT_STEPS and dca.get('last_exit_t') != t_curr:
                        qty = self.auto_active_pos[(symbol, 'long')]['size']
                        if not self.SCALE_OUT_EXITS:
                            sell_qty = qty
                            dca['exit_count'] = self.SCALE_OUT_STEPS - 1
                        else:
                            # [Fix] 3단계 분할 (1/3 → 1/2 → 전량). 기존 1/8씩은 수익 실현이 너무 느림
                            sell_qty = qty / max(1, (self.SCALE_OUT_STEPS - dca['exit_count']))
                        m_info = self.exchange.markets.get(symbol)
                        if m_info:
                            min_amount = m_info.get('limits', {}).get('amount', {}).get('min', 0)
                            if min_amount and sell_qty < min_amount:
                                sell_qty = min_amount
                        if sell_qty >= qty:
                            sell_qty = 0
                        else:
                            sell_qty = float(self.exchange.amount_to_precision(symbol, sell_qty))
                        if sell_qty >= 0:
                            if take_profit_long_sig:
                                self.logger.info(f"💎 [Take Profit] 롱 목표가 달성 분할 익절 ({dca['exit_count']+1}/{self.SCALE_OUT_STEPS}): {symbol} (수량: {sell_qty})")
                            else:
                                self.logger.info(f"💨 [{self.STRATEGY_NAME} DCA] 롱 분할 청산 ({dca['exit_count']+1}/{self.SCALE_OUT_STEPS}): {symbol} (수량: {sell_qty if sell_qty > 0 else 'ALL'})")
                            await self.send_webhook(SideType.CLOSE_LONG, symbol, sell_qty)
                        dca['exit_count'] += 1
                        dca['last_exit_t'] = t_curr
                        if dca['exit_count'] >= self.SCALE_OUT_STEPS:
                            dca['entry_count'] = 0
                            dca['exit_count'] = 0
                            dca['max_pnl_pct'] = 0.0
                else:
                    if (self.MAX_DCA_ENTRIES > 0 and is_long_sig and not dca.get('mr_mode') and dca['entry_count'] < self.MAX_DCA_ENTRIES and self._dca_ready(dca, t_curr) and pnl_pct_long > 0.0 and not long_blocked):
                        self.logger.info(f"🔥 [{self.STRATEGY_NAME} DCA] 롱 분할 진입 ({dca['entry_count']+1}/{self.MAX_DCA_ENTRIES}): {symbol}")
                        await self.execute_auto_entry(symbol, SideType.BUY, entry_type="dca")
                        dca['entry_count'] += 1
                        dca['last_entry_t'] = t_curr

            if has_short:
                if force_close_short and dca.get('last_exit_t') != t_curr:
                    if is_hard_stop_short:
                        self.logger.warning(f"🚨 [HARD STOP] 숏 전량 긴급 손절 (손실률: {pnl_pct_short*100:.2f}%): {symbol}")
                    else:
                        self.logger.info(f"💨 [Breakeven Stop] 숏 전량 방어 청산: {symbol}")
                    await self.send_webhook(SideType.CLOSE_SHORT, symbol, 0)
                    # [Flip] 트레일링/방어 청산 시 즉시 롱 진입 (하드스탑 제외, Jev 활성화 시 레짐 바이패스)
                    flipped = False
                    if self.FLIP_ON_TRAILING_CLOSE and not is_hard_stop_short and not dca.get('mr_mode') and (self._long_regime_ok or jev_active) and not short_blocked and (dual_gate or jev_active):
                        self.logger.info(f"🔄 [FLIP] 숏 청산 → 롱 반대진입: {symbol} (최고수익: {dca['max_pnl_pct']*100:.0f}%)")
                        await self.execute_auto_entry(symbol, SideType.BUY, entry_type="flip")
                        flipped = True
                    dca['exit_count'] = self.SCALE_OUT_STEPS
                    dca['entry_count'] = 0
                    dca['pyramid_count'] = 0
                    dca['last_exit_t'] = t_curr
                    dca['max_pnl_pct'] = 0.0
                    dca['last_close_t'] = t_curr
                    dca['last_close_side'] = 'short'
                    if is_hard_stop_short:
                        dca['last_hard_stop_t'] = t_curr
                    if flipped:
                        # [Fix] 플립으로 열린 신규 롱 포지션 관리 가능하도록 상태 리셋
                        dca.update({
                            'entry_count': 1, 'exit_count': 0, 'pyramid_count': 0,
                            'max_pnl_pct': 0.0, 'first_entry_t': t_curr,
                            'last_entry_t': t_curr, 'last_close_t': 0, 'side': 'long',
                        })
                elif close_short_sig or take_profit_short_sig:
                    if dca['exit_count'] < self.SCALE_OUT_STEPS and dca.get('last_exit_t') != t_curr:
                        qty = self.auto_active_pos[(symbol, 'short')]['size']
                        if not self.SCALE_OUT_EXITS:
                            sell_qty = qty
                            dca['exit_count'] = self.SCALE_OUT_STEPS - 1
                        else:
                            # [Fix] 3단계 분할 (1/3 → 1/2 → 전량). 기존 1/8씩은 수익 실현이 너무 느림
                            sell_qty = qty / max(1, (self.SCALE_OUT_STEPS - dca['exit_count']))
                        m_info = self.exchange.markets.get(symbol)
                        if m_info:
                            min_amount = m_info.get('limits', {}).get('amount', {}).get('min', 0)
                            if min_amount and sell_qty < min_amount:
                                sell_qty = min_amount
                        if sell_qty >= qty:
                            sell_qty = 0
                        else:
                            sell_qty = float(self.exchange.amount_to_precision(symbol, sell_qty))
                        if sell_qty >= 0:
                            if take_profit_short_sig:
                                self.logger.info(f"💎 [Take Profit] 숏 목표가 달성 분할 익절 ({dca['exit_count']+1}/{self.SCALE_OUT_STEPS}): {symbol} (수량: {sell_qty})")
                            else:
                                self.logger.info(f"💨 [{self.STRATEGY_NAME} DCA] 숏 분할 청산 ({dca['exit_count']+1}/{self.SCALE_OUT_STEPS}): {symbol} (수량: {sell_qty if sell_qty > 0 else 'ALL'})")
                            await self.send_webhook(SideType.CLOSE_SHORT, symbol, sell_qty)
                        dca['exit_count'] += 1
                        dca['last_exit_t'] = t_curr
                        if dca['exit_count'] >= self.SCALE_OUT_STEPS:
                            dca['entry_count'] = 0
                            dca['exit_count'] = 0
                            dca['max_pnl_pct'] = 0.0
                elif is_short_pullback and not dca.get('mr_mode') and self.MAX_DCA_ENTRIES > 0 and pnl_pct_short > 0.0 and dca['entry_count'] < self.MAX_DCA_ENTRIES and self._dca_ready(dca, t_curr) and not short_blocked:
                    self.logger.info(f"📉 [Short Pullback 진입] {symbol} (DCA {dca['entry_count']+1}/{self.MAX_DCA_ENTRIES})")
                    await self.execute_auto_entry(symbol, SideType.SELL, entry_type="dca")
                    dca['entry_count'] += 1
                    dca['last_entry_t'] = t_curr

            # [CRITICAL BUG FIX #2] 기존 else: 는 has_short 가 아닐 때 롱을 들고 있어도 무조건 진입하여
            # 매 캔들마다 신규 진입 난사 및 entry_count=1 리셋 버그를 유발했음.
            # 롱/숏 포지션이 둘 다 없을 때만 신규 진입 블록 진입하도록 엄격 봉인.
            if not has_long and not has_short:
                active_symbols = set(sym for sym, _side in self.auto_active_pos.keys())
                total_count = len(active_symbols)
                new_listing_count = sum(1 for s in active_symbols if self._is_new_listing(s))
                regular_count = total_count - new_listing_count
                max_regular = self.MAX_OPEN_POSITIONS - self.NEW_LISTING_SLOTS
                
                is_new_listing = self._is_new_listing(symbol)
                
                if not is_new_listing:
                    if regular_count >= max_regular:
                        # 일반 종목 슬롯 포화 -> 예약된 신규 슬롯 보호를 위해 진입 차단
                        return
                
                if total_count >= self.MAX_OPEN_POSITIONS:
                    # 전체 슬롯 포화
                    return


                # [하드 스탑 쿨다운 체크]
                is_in_cooldown = False
                if 'last_hard_stop_t' in dca:
                    if (t_curr - dca['last_hard_stop_t']) < (self.HARD_STOP_COOLDOWN_HOURS * 3600 * 1000):
                        is_in_cooldown = True

                # [재진입] 전량 청산 후 같은 방향 추세 유지 시 쿨다운 후 재진입
                if self.REENTRY_ENABLED and not dca.get('mr_mode') and dca.get('last_close_t') and dca.get('last_entry_t') != t_curr and not is_in_cooldown and not (long_blocked if dca.get('side') == 'long' else short_blocked):
                    cooldown_ms = self.REENTRY_COOLDOWN_CANDLES * self.TIMEFRAME_MINUTES * 60 * 1000
                    if (t_curr - dca['last_close_t']) >= cooldown_ms:
                        side_closed = dca.get('last_close_side')
                        re_long = side_closed == 'long' and curr['st_d_loose'] == 1 and curr['c'] > curr['ema_target'] and (self._long_regime_ok or jev_active) and (dual_gate or jev_active)
                        re_short = side_closed == 'short' and curr['st_d_loose'] == -1 and curr['c'] < curr['ema_target'] and self.SHORTS_ENABLED
                        if re_long or re_short:
                            side = SideType.BUY if re_long else SideType.SELL
                            self.logger.info(f"♻️ [Re-entry] 추세 지속 재진입 ({side_closed}): {symbol}")
                            await self.execute_auto_entry(symbol, side, entry_type="reentry")
                            dca['entry_count'] = 1
                            dca['exit_count'] = 0
                            dca['last_entry_t'] = t_curr
                            dca['first_entry_t'] = t_curr
                            dca['last_close_t'] = 0
                            dca['side'] = 'long' if re_long else 'short'
                            dca['stop_pct'] = self._stop_distance_pct(df)  # [Fix] 진입 시점 ATR 스탑 고정
                            return

                if is_long_sig and dca.get('last_entry_t') != t_curr and not is_in_cooldown and not long_blocked:
                    self.logger.info(f"🟢 [Scoring System 신규 진입] {symbol} (Score: {long_score})")
                    entered = await self.execute_auto_entry(symbol, SideType.BUY, entry_type="new", base_score=long_score)
                    if entered:
                        dca['entry_count'] = 1
                        dca['pyramid_count'] = 0
                        dca['exit_count'] = 0
                        dca['last_entry_t'] = t_curr
                        dca['first_entry_t'] = t_curr
                        dca['side'] = 'long'
                        dca['stop_pct'] = self._stop_distance_pct(df)
                elif is_short_sig and dca.get('last_entry_t') != t_curr and not is_in_cooldown and not short_blocked:
                    # [Fix] HTF 추세 필터: Jev 비활성화 시에만 1h EMA50 체크 (Jev 활성화 시 Jev가 1번 판단권자)
                    htf = await self._check_htf_trend(symbol)
                    if not jev_active and htf['is_uptrend']:
                        self.logger.info(
                            f"🚫 [HTF Filter] 숏 진입 차단 — 1h EMA50 상승 중: {symbol} "
                            f"(기울기: {htf['ema_slope']*100:+.2f}%, Score: {short_score})"
                        )
                    else:
                        self.logger.info(f"🔴 [Scoring System 신규 진입] {symbol} (Score: {short_score}, HTF: {htf['ema_slope']*100:+.2f}%)")
                        entered = await self.execute_auto_entry(symbol, SideType.SELL, entry_type="new", base_score=short_score)
                        if entered:
                            dca['entry_count'] = 1
                            dca['pyramid_count'] = 0
                            dca['exit_count'] = 0
                            dca['last_entry_t'] = t_curr
                            dca['first_entry_t'] = t_curr
                            dca['side'] = 'short'
                            dca['stop_pct'] = self._stop_distance_pct(df)

                # ── [수익성 개선] 평균회귀(Mean-Reversion) 진입 — 레인지 시장용 ──
                # 횡보장(CHOP)에서 RSI 과매수/과매도 기반 반대편 진입
                if self.MEAN_REVERSION_ENABLED and is_mean_rev_long_sig and dca.get('last_entry_t') != t_curr and not is_in_cooldown and not mr_long_blocked:
                    self.logger.info(f"🔄 [Mean-Reversion LONG] {symbol} (RSI 과매도 — 레인지 반등)")
                    entered = await self.execute_auto_entry(symbol, SideType.BUY, entry_type="mean_rev", base_score=30, mr_mode=True)
                    if entered:
                        dca['entry_count'] = 1
                        dca['pyramid_count'] = 0
                        dca['exit_count'] = 0
                        dca['last_entry_t'] = t_curr
                        dca['first_entry_t'] = t_curr
                        dca['side'] = 'long'
                        dca['stop_pct'] = self.MEAN_REVERSION_ATR_K * float(curr['atr']) / float(curr['c']) if 'atr' in df.columns and float(curr['c']) > 0 else self._stop_distance_pct(df)
                        dca['mr_mode'] = True
                        dca['mr_entry_t'] = t_curr
                elif self.MEAN_REVERSION_ENABLED and is_mean_rev_short_sig and dca.get('last_entry_t') != t_curr and not is_in_cooldown and not mr_short_blocked:
                    self.logger.info(f"🔄 [Mean-Reversion SHORT] {symbol} (RSI 과매수 — 레인지 하락)")
                    entered = await self.execute_auto_entry(symbol, SideType.SELL, entry_type="mean_rev", base_score=30, mr_mode=True)
                    if entered:
                        dca['entry_count'] = 1
                        dca['pyramid_count'] = 0
                        dca['exit_count'] = 0
                        dca['last_entry_t'] = t_curr
                        dca['first_entry_t'] = t_curr
                        dca['side'] = 'short'
                        dca['stop_pct'] = self.MEAN_REVERSION_ATR_K * float(curr['atr']) / float(curr['c']) if 'atr' in df.columns and float(curr['c']) > 0 else self._stop_distance_pct(df)
                        dca['mr_mode'] = True
                        dca['mr_entry_t'] = t_curr

        except Exception as e:
            self.logger.error(f"⚠️ [{self.STRATEGY_NAME}] 로직 체크 실패 ({symbol}): {e}")

    async def execute_auto_entry(self, symbol: str, side: SideType, portion: float = 0.20, entry_type: str = "new", base_score: float = 70.0, mr_mode: bool = False):
        """
        [개선] equity 기반 균등 분할 포지션 사이징.

        기존: free_usdt × portion → 포지션 수 증가 시 사이즈가 0에 수렴
        개선: total_equity / MAX_OPEN_POSITIONS 기반 목표 마진 산출
              + MIN_POSITION_MARGIN 하한선으로 극소 포지션 원천 방지
              + [수익성] 컨빅션 사이징: 진입 점수에 비례해 포지션 크기 0.5x~2x 조절

        Args:
            symbol: 거래 심볼
            side: 매수/매도 방향
            portion: (하위 호환) 기존 portion 파라미터 (entry_type 우선)
            entry_type: "new", "dca", "pyramid", "flip", "reentry", "mean_rev"
            base_score: 진입 점수 (컨빅션 사이징에 사용, 기본 70)
            mr_mode: 평균회귀 모드 (사이즈 축소, 타이트한 스탑)
        """
        try:
            balance = await self.exchange.fetch_balance()
            free_usdt = balance.get('USDT', {}).get('free', 0)
            if not free_usdt:
                free_usdt = balance.get('free', {}).get('USDT', 0)
            total_usdt = balance.get('USDT', {}).get('total', 0)
            if not total_usdt:
                total_usdt = balance.get('total', {}).get('USDT', 0)
            # total이 0이면 free로 폴백 (API 응답 누락 대비)
            if not total_usdt:
                total_usdt = free_usdt

            ticker = await self.exchange.fetch_ticker(symbol)
            price = ticker.get('last')

            if not free_usdt or not price:
                return

            # [Fix] 같은 사이클에서 이미 예약된 마진 차감 (중복 진입으로 인한 51008 방지)
            effective_free = max(0.0, float(free_usdt) - self._reserved_margin)

            # [Bug Fix & Feature] 숏 방향일 경우 별도의 레버리지(5x) 적용
            if side == SideType.SELL:
                leverage = int(os.getenv("OKX_SHORT_LEVERAGE", "5"))
            else:
                default_leverage = int(os.getenv("OKX_LEVERAGE", "10"))
                leverage = getattr(self, 'STRATEGY_LEVERAGE', default_leverage)

            # ── [Fix] Equity 기반 균등 분할 사이징 ──
            target_margin = self._calc_target_margin(effective_free, total_usdt, entry_type)
            # [수익성] 평균회귀 모드: 사이즈 축소 (레인지 시장은 변동성 낮으므로 작은 포지션)
            if mr_mode and self.MEAN_REVERSION_ENABLED:
                target_margin *= self.MEAN_REVERSION_SIZE_MULT
            # [Fix] 트레이드당 리스크 예산: ATR 스탑 맞아도 손실 = 자산 × RISK_PER_TRADE.
            # 변동성 큰 종목은 스탑이 넓어지는 만큼 사이즈가 줄어 레버리지·종목 무관하게 손실 금액이 균일화된다.
            _df = self._df_cache.get(symbol)
            if mr_mode and self.MEAN_REVERSION_ENABLED:
                stop_pct = self.MEAN_REVERSION_ATR_K * float(_df['atr'].iloc[-2]) / float(_df['c'].iloc[-2]) if _df is not None and 'atr' in _df.columns and float(_df['c'].iloc[-2]) > 0 else self.ATR_STOP_MIN_PCT
            else:
                stop_pct = self._stop_distance_pct(_df) if _df is not None else self.ATR_STOP_MIN_PCT
            risk_budget = float(total_usdt) * self.RISK_PER_TRADE
            if stop_pct > 0 and leverage > 0:
                target_margin = min(target_margin, (risk_budget / stop_pct) / leverage)
            # [수익성] 컨빅션 사이징: 진입 점수에 비례 포지션 크기 조절 (0.5x~2x)
            if self.CONVICTION_SIZING_ENABLED and entry_type in ("new", "flip", "reentry"):
                conv_mult = max(self.CONVICTION_MIN_MULT, min(self.CONVICTION_MAX_MULT, base_score / 70.0))
                target_margin *= conv_mult
            # [단계적 배포] Jev 미사용 시에만 약추세 구간(ADX 소프트 경계~임계값) 축소 적용
            jev_active = bool(hasattr(self, 'jev_filter') and self.jev_filter and self.jev_filter.is_enabled)
            if not jev_active:
                deploy_scale = getattr(self, '_deploy_scale', 1.0)
                if deploy_scale < 1.0:
                    target_margin *= deploy_scale
                    if target_margin < self.MIN_POSITION_MARGIN:
                        return  # 축소해도 최소 마진 미달 시 스킵
            # [섹터별 사이즈] 신규상장 0.75x 등 섹터 배수 적용
            sec_size = self._sector_params(symbol).get('size_mult', 1.0)
            if sec_size < 1.0:
                target_margin *= sec_size
                if target_margin < self.MIN_POSITION_MARGIN:
                    return
            # [Alpha ⑤] 펀딩비 정렬: 극단 캐리에서 유리한 방향 확대 / 불리한 방향 축소
            if self.FUNDING_ADJUST_ENABLED:
                fr = await self._get_funding_rate(symbol)
                if fr is not None:
                    # [개선 #5] 극단 펀딩비 함정 진입 원천 차단 (8h ±0.30% 이상 초과열 종목)
                    if side == SideType.BUY and fr >= 0.0030:
                        self.logger.warning(f"🚫 [Funding Extreme] {symbol} 롱 진입 차단 — 펀딩비 과열 ({fr*100:+.3f}% ≥ +0.30%)")
                        return
                    elif side == SideType.SELL and fr <= -0.0030:
                        self.logger.warning(f"🚫 [Funding Extreme] {symbol} 숏 진입 차단 — 펀딩비 과열 ({fr*100:+.3f}% ≤ -0.30%)")
                        return

                    if abs(fr) >= min(abs(self.FUNDING_EXTREME_POS), abs(self.FUNDING_EXTREME_NEG)):
                        if side == SideType.BUY:
                            _fmult = 1.25 if fr <= self.FUNDING_EXTREME_NEG else (0.6 if fr >= self.FUNDING_EXTREME_POS else 1.0)
                        else:
                            _fmult = 1.25 if fr >= self.FUNDING_EXTREME_POS else (0.6 if fr <= self.FUNDING_EXTREME_NEG else 1.0)
                        if _fmult != 1.0:
                            self.logger.info(f"💸 [Funding] {symbol} 펀딩비 {fr*100:+.3f}% → 사이즈 ×{_fmult}")
                            target_margin *= _fmult
                            if target_margin < self.MIN_POSITION_MARGIN:
                                return
            if target_margin <= 0:
                if not self._margin_reject_logged:
                    self.logger.warning(
                        f"⚠️ [{self.STRATEGY_NAME}] 목표 마진 0 → 진입 거부 "
                        f"(free: {free_usdt:.2f}, 예약: {self._reserved_margin:.2f}, "
                        f"equity: {total_usdt:.2f}, type: {entry_type}) — 이번 사이클 추가 진입 스킵"
                    )
                    self._margin_reject_logged = True
                return

            # 목표 마진 → 레버리지 반영 노셔널 → 수량 계산
            notional = target_margin * leverage
            raw_amount = notional / price

            market_info = self.exchange.markets.get(symbol)
            contract_size = market_info.get('contractSize', 1) if market_info else 1
            raw_contracts = raw_amount / float(contract_size)

            amount = self.exchange.amount_to_precision(symbol, raw_contracts)
            amount = float(amount)

            if amount > 0:
                min_amount = market_info.get('limits', {}).get('amount', {}).get('min', 0) if market_info else 0
                max_amount = market_info.get('limits', {}).get('market', {}).get('max', 0) if market_info else 0

                if min_amount and amount < min_amount:
                    self.logger.warning(f"⚠️ 진입 수량({amount})이 최소 수량({min_amount}) 미만 (목표마진: {target_margin:.2f})")
                    return
                if max_amount and amount > max_amount:
                    amount = float(self.exchange.amount_to_precision(symbol, max_amount))

                required_margin = (amount * float(contract_size) * price) / leverage
                if required_margin > effective_free * 0.95:
                    self.logger.warning(f"⚠️ USDT 증거금 부족 - 필요: {required_margin:.2f}, 가용: {effective_free:.2f} (수량: {amount})")
                    return

                # [Fix] 신규 자본 투입(new/reentry/flip)은 진입 속도 예산 + 호가 깊이 가드 통과 필수
                is_fresh_capital = entry_type in ("new", "reentry", "flip", "mean_rev")
                if is_fresh_capital:
                    if not self._entry_budget_ok():
                        if not self._entry_budget_logged:
                            self.logger.info(
                                f"⏸️ [Entry Budget] 시간당 {self.MAX_ENTRIES_PER_HOUR}/일일 {self.MAX_ENTRIES_PER_DAY} 한도 또는 주말 차단 — {symbol} 진입 스킵 (이 사이클 추가 로그 생략)"
                            )
                            self._entry_budget_logged = True
                        return
                    if not await self._liquidity_ok(symbol, side, amount * float(contract_size) * price):
                        return

                order_type = "MARKET"
                target_price = None
                jev_score = None
                is_sim = False

                # [Jev AI Gate & Pricing] 초단타 호가창 예측 평가
                if hasattr(self, 'jev_filter') and self.jev_filter and is_fresh_capital:
                    try:
                        proposed_side_str = "BUY" if side == SideType.BUY else "SELL"
                        market_prec = market_info.get('precision', {}) if isinstance(market_info, dict) else {}
                        tick_size = float(market_prec.get('price', 0.1) or 0.1)
                        jev_res = await self.jev_filter.evaluate_signal(
                            symbol=symbol,
                            proposed_side=proposed_side_str,
                            tick_size=tick_size,
                            context_note=f"Strategy={self.STRATEGY_NAME}, Type={entry_type}, Lev={leverage}x"
                        )
                        if not jev_res.approved:
                            self.logger.info(
                                f"🚫 [Jev AI Gate] 진입 차단: {side.value} {symbol} "
                                f"(점수: {jev_res.jev_score:.3f}, 사유: {jev_res.reason})"
                            )
                            return

                        order_type = jev_res.order_type
                        target_price = jev_res.target_price
                        jev_score = jev_res.jev_score
                        is_sim = jev_res.is_simulation
                        self.logger.info(
                            f"🤖 [Jev AI 승인] {side.value} {symbol} | 점수: {jev_score:.3f} | "
                            f"방식: {order_type} @ {target_price} | 지연: {jev_res.latency_ms:.1f}ms | Sim: {is_sim}"
                        )
                    except Exception as ex_jev:
                        self.logger.warning(f"⚠️ Jev 평가 중 예외 (기준 로직으로 진행): {ex_jev}")

                self.logger.info(
                    f"🔥 [{self.STRATEGY_NAME}] 진입 시그널: {side.value} {symbol} "
                    f"(수량: {amount}, 목표마진: {target_margin:.1f} USDT, ATR스탑: {stop_pct*100:.2f}%, 유형: {entry_type})"
                )
                await self.send_webhook(
                    side, symbol, amount, leverage=leverage, stop_pct=stop_pct,
                    order_type=order_type, target_price=target_price,
                    jev_score=jev_score, is_simulation=is_sim
                )
                if is_fresh_capital and not is_sim:
                    self._entry_log.append(time.time())
                # [Fix] 사이클 내 후속 신호가 동일 마진을 중복 사용하지 않도록 예약 처리
                if not is_sim:
                    self._reserved_margin += target_margin
                return True
        except Exception as e:
            self.logger.error(f"⚠️ [{self.STRATEGY_NAME}] 진입 수량 계산 실패 ({symbol}): {e}")
            return False
        return False

    async def _update_regime(self):
        """
        [Fix] BTC 1h OHLCV 1회 fetch로 롱/숏 레짐 동시 갱신 (기존 2회 fetch → 1회).
        로깅은 레짐 전환 시에만 수행 (기존 매 사이클 중복 로그 스팸 제거).
        실패 시 기존 상태 유지(안전 측).
        """
        jev_active = bool(hasattr(self, 'jev_filter') and self.jev_filter and self.jev_filter.is_enabled)
        if jev_active:
            self._long_regime_ok = True
            self._short_regime_ok = True
            self._btc_above_ema50_1h = True
            if not getattr(self, '_jev_logged_supreme', False):
                self.logger.info("⚡ [Jev AI Supreme] 1번 판단권자 Jev 가동: BTC EMA50/200 및 ADX 게이트 100% 해제 (Jev AI 호가 수급 전권 행사)")
                self._jev_logged_supreme = True
            return
        try:
            ohlcv = await self.exchange.fetch_ohlcv('BTC/USDT:USDT', '1h', limit=220)
            if not ohlcv or len(ohlcv) < 200:
                return
            closes = pd.Series([c[4] for c in ohlcv], dtype='float64')
            ema200_series = closes.ewm(span=200, adjust=False).mean()
            ema50_series = closes.ewm(span=50, adjust=False).mean()
            ema200 = ema200_series.iloc[-1]
            long_ok = bool(closes.iloc[-1] >= ema200)
            # [K안] 시장 게이트: BTC > 1h EMA50
            self._btc_above_ema50_1h = bool(closes.iloc[-1] > ema50_series.iloc[-1])
            btc_bullish = bool(closes.iloc[-1] >= ema200 and ema50_series.iloc[-1] > ema50_series.iloc[-5])
            short_ok = not btc_bullish  # True = 숏 허용, False = 숏 억제

            if long_ok != self._long_regime_ok:
                self.logger.info(
                    f"🧭 [Regime 전환] 롱 {'허용' if long_ok else '억제'}: "
                    f"BTC {closes.iloc[-1]:.0f} vs EMA200 {ema200:.0f}"
                )
            if short_ok != self._short_regime_ok:
                self.logger.info(
                    f"🧭 [Regime 전환] 알트숏 {'허용' if short_ok else '억제'}: "
                    f"BTC {closes.iloc[-1]:.0f} vs EMA200 {ema200:.0f}, "
                    f"EMA50 {'상승' if btc_bullish else '횡보/하락'}"
                )
            self._long_regime_ok = long_ok
            self._short_regime_ok = short_ok
        except Exception as e:
            self.logger.warning(f"⚠️ [{self.STRATEGY_NAME}] BTC 레짐 체크 실패(기존 상태 유지): {e}")

    async def _update_chop_filter(self):
        """
        [항상 가동 + 사이즈 호흡] 철칙: 거래가 없으면 기회도 없다.
        ADX에 비례한 연속 배포 곡선 — 하드 차단 폐지.
          deploy_scale = clamp(ADX / 25, 0.25, 1.0)
        개별 트레이드 리스크는 포지션손실한도(-15%)가 통제 → 진입은 열어두고 청산이 지킨다.
        """
        jev_active = bool(hasattr(self, 'jev_filter') and self.jev_filter and self.jev_filter.is_enabled)
        if jev_active:
            self._deploy_scale = 1.0
            return

        if not self.CHOP_FILTER_ENABLED:
            self._deploy_scale = 1.0
            return
        try:
            ohlcv = await self.exchange.fetch_ohlcv('BTC/USDT:USDT', '1h', limit=100)
            if not ohlcv or len(ohlcv) < 50:
                return
            df = pd.DataFrame(ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            adx_series = calc_adx(df, 14)
            adx_now = float(adx_series.iloc[-2])  # 직전 확정 캔들 기준
            self._current_adx = adx_now  # 신호 보너스 게이트용

            # [8/24 교훈] 완만한 곡선은 저변동 심충보에서 과다 배포 → 급락 시 연쇄 손절.
            # FLOOR_ADX 아래에서는 바닥(15%)까지만 급감.
            raw_scale = (adx_now - self.CHOP_FLOOR_ADX) / max(self.CHOP_ADX_THRESHOLD - self.CHOP_FLOOR_ADX, 1.0)
            scale = max(self.CHOP_FLOOR, min(1.0, raw_scale))

            prev_state = getattr(self, '_deploy_state', None)
            # 로그 스팸 방지: 배율 밴드(0.25/0.50/0.75/1.00) 전환 시에만
            band_now = round(scale * 4) / 4
            band_prev = getattr(self, '_deploy_band', None)
            if band_now != band_prev or prev_state is None:
                pct = f"{band_now*100:.0f}%"
                if band_now >= 1.0:
                    self.logger.info(f"✅ [배포 {pct}] 강한 추세 (ADX {adx_now:.1f} ≥ {self.CHOP_ADX_THRESHOLD:.0f}) — 풀 사이즈")
                elif band_now <= self.CHOP_FLOOR + 0.01:
                    self.logger.info(f"🟡 [배포 {pct}] 극저변동 (ADX {adx_now:.1f}) — 최소 사이즈로 계속 거래")
                else:
                    self.logger.info(f"🟡 [배포 {pct}] 추세 강도 보통 (ADX {adx_now:.1f}) — 사이즈 비례 운용")
                self._deploy_band = band_now
            self._deploy_state = "always_on"
            # [백테스트 검증] 극저변동/횡보장(BTC 1h ADX < 임계값) 신규 진입만 차단, 청산은 계속 동작.
            block_threshold = getattr(self, 'CHOP_ADX_BLOCK_THRESHOLD', 20.0)
            was_blocked = getattr(self, '_chop_block', False)
            self._chop_block = adx_now < block_threshold
            if self._chop_block != was_blocked:
                if self._chop_block:
                    self.logger.info(f"⏸️ [{self.STRATEGY_NAME}] 횡보장 휩쏘 방어 가동 (BTC 1h ADX {adx_now:.1f} < {block_threshold:.0f}) — 신규 진입 차단 (청산/스탑 유지)")
                else:
                    self.logger.info(f"▶️ [{self.STRATEGY_NAME}] 추세 발생 확인 (BTC 1h ADX {adx_now:.1f} ≥ {block_threshold:.0f}) — 신규 진입 재개")
            self._deploy_scale = scale
        except Exception as e:
            self.logger.warning(f"⚠️ [{self.STRATEGY_NAME}] 배포 스케일 갱신 실패(기존 유지): {e}")

    # ── [Fix #2] 서킷 브레이커 상태 영속화 ──
    def _cb_state_path(self) -> str:
        return os.path.join(BASE_DIR, "state", f"circuit_breaker_{self.STRATEGY_NAME}.json")

    def _load_cb_state(self):
        try:
            with open(self._cb_state_path(), encoding="utf-8") as f:
                self._cb_state = json.load(f)
        except Exception:
            self._cb_state = {}
        self._circuit_open = bool(self._cb_state.get("tripped", False))

    def _save_cb_state(self):
        try:
            os.makedirs(os.path.dirname(self._cb_state_path()), exist_ok=True)
            tmp = self._cb_state_path() + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._cb_state, f)
            os.replace(tmp, self._cb_state_path())
        except Exception as e:
            self.logger.warning(f"⚠️ 서킷 브레이커 상태 저장 실패: {e}")

    async def _update_circuit_breaker(self):
        """
        [Fix #2] 현재 자산 vs 당일 시작 기준자산 비교 → 임계 이하 하락 시 진입 차단.
        [Fix 추가] 48시간 쿨다운 & 주간 MDD 락(-10%) 적용
        """
        if not self.CIRCUIT_BREAKER_ENABLED:
            self._circuit_open = False
            return
        try:
            balance = await self.exchange.fetch_balance()
            equity = float(balance.get('USDT', {}).get('total', 0) or balance.get('total', {}).get('USDT', 0) or 0)
            if equity <= 0:
                return
            
            # [EMERGENCY] 고점(HWM) 대비 절대 낙폭 킬스위치: 일/주 앵커 리셋과 무관
            hwm = float(self._cb_state.get("hwm", 0) or 0)
            if equity > hwm:
                self._cb_state["hwm"] = equity
                hwm = equity
                self._save_cb_state()
            dd_from_peak = (equity - hwm) / hwm * 100 if hwm > 0 else 0
            kill_pct = float(os.getenv("OKX_KILL_DD_FROM_PEAK", "-15"))
            if dd_from_peak <= kill_pct or self._cb_state.get("killed"):
                if not self._cb_state.get("killed"):
                    self._cb_state["killed"] = True
                    self._save_cb_state()
                    self.logger.critical(f"🛑 [KILL] HWM 대비 {dd_from_peak:.1f}% — 신규 진입 영구 차단 (state 파일 수동 해제 필요)")
                self._circuit_open = True
                self._short_regime_ok = False
                return

            # [Health CB] 전략 건강도: 최근 HEALTH_CB_LOOKBACK건 라운드트립 PF/승률 저조 → 24h 신규 진입 정지.
            # 자산 기준 CB와 독립. 재발동 판정은 직전 발동 이후 청산된 새 표본만 사용(영구 잠김 방지).
            # 08-30 밤에 발동했다면 08-31 하루 28건 -469 USDT를 막을 수 있었음.
            now_ts = time.time()
            if self.HEALTH_CB_ENABLED and now_ts >= float(self._cb_state.get("cooldown_until", 0) or 0):
                rt = self._recent_roundtrip_pnls(
                    self.HEALTH_CB_LOOKBACK, float(self._cb_state.get("health_last_trip_ts", 0) or 0)
                )
                if len(rt) >= self.HEALTH_CB_LOOKBACK:
                    gp = sum(x for x in rt if x > 0)
                    gl = -sum(x for x in rt if x <= 0)
                    pf = gp / gl if gl > 0 else 9.9
                    wr = sum(1 for x in rt if x > 0) / len(rt)
                    if pf < self.HEALTH_CB_MIN_PF or wr < self.HEALTH_CB_MIN_WR:
                        self._cb_state["tripped"] = True
                        self._cb_state["cooldown_until"] = now_ts + 86400
                        self._cb_state["health_last_trip_ts"] = now_ts
                        self._save_cb_state()
                        self._circuit_open = True
                        self.logger.warning(
                            f"🩺 [Health CB] 최근 {len(rt)}건 PF {pf:.2f} / 승률 {wr*100:.0f}% — 24h 신규 진입 정지"
                        )
                        try:
                            from utils_telegram import send_telegram_alert
                            send_telegram_alert(
                                f"🩺 [{self.STRATEGY_NAME}] Health CB 발동: 최근 {len(rt)}건 PF {pf:.2f}, 승률 {wr*100:.0f}% — 24h 진입 정지"
                            )
                        except Exception:
                            pass
                        return
            today = datetime.utcnow().strftime("%Y-%m-%d")
            # ISO 달력으로 주차(Year, Week) 구하기
            current_week = f"{datetime.utcnow().isocalendar()[0]}-W{datetime.utcnow().isocalendar()[1]}"
            
            # 주간 고점 트래킹 리셋 (주가 바뀌면 리셋)
            if self._cb_state.get("week") != current_week:
                self._cb_state["week"] = current_week
                self._cb_state["weekly_peak"] = equity
                # [Fix: Sticky Lock] 주간 락은 오직 새 주간 시작 시에만 해제
                if self._cb_state.get("weekly_locked", False):
                    self._cb_state["weekly_locked"] = False
                    self._save_cb_state()
                    self.logger.info("✅ [Global Lock] 새 주간 시작 — 주간 MDD 락 해제")
            else:
                self._cb_state["weekly_peak"] = max(self._cb_state.get("weekly_peak", equity), equity)
                
            weekly_peak = float(self._cb_state.get("weekly_peak", equity))
            weekly_mdd = (equity - weekly_peak) / weekly_peak * 100 if weekly_peak > 0 else 0
            
            # 주간 MDD 락 확인
            # [Fix: Sticky Lock] 한 번 발동한 주간 락은 등락으로 MDD가 -10% 위로 회복해도
            # 해제하지 않는다 (락/해제 플래핑 방지). 해제는 새 주간 시작 시에만 수행.
            global_mdd_limit = getattr(self, "GLOBAL_MDD_LIMIT", -10.0)
            if weekly_mdd <= global_mdd_limit or self._cb_state.get("weekly_locked", False):
                if not self._cb_state.get("weekly_locked", False):
                    self._cb_state["weekly_locked"] = True
                    self.logger.warning(f"🚨 [Global Lock] 주간 MDD {weekly_mdd:.2f}% ≤ 임계 {global_mdd_limit:.1f}% — 이번 주 전면 거래 정지 (해제: 새 주간)")
                    try:
                        from utils_telegram import send_telegram_alert
                        send_telegram_alert(f"🚨 [{self.STRATEGY_NAME}] 주간 누적 손실 {weekly_mdd:.2f}% 도달 — Global Lock 발동 (이번 주 거래 차단)")
                    except Exception:
                        pass
                self._circuit_open = True
                self._save_cb_state()
                return
            
            # 쿨다운 타이머 확인
            cooldown_until = self._cb_state.get("cooldown_until", 0)
            now_ts = time.time()
            if now_ts < cooldown_until:
                self._circuit_open = True
                if int(now_ts) % 3600 == 0:
                    rem_hours = (cooldown_until - now_ts) / 3600
                    self.logger.info(f"⏳ [Circuit Breaker] 쿨다운 적용 중 (남은 시간: {rem_hours:.1f}시간)")
                return

            if self._cb_state.get("date") != today:
                self._cb_state = {
                    "date": today, 
                    "anchor_equity": equity, 
                    "tripped": False,
                    "week": current_week,
                    "weekly_peak": self._cb_state.get("weekly_peak", equity),
                    "cooldown_until": 0,
                    # [Fix] 일일 리셋 시 HWM 킬스위치/Health CB 상태 유실 방지 (기존엔 hwm이 매일 현재 자산으로 초기화되어 킬스위치 무력화)
                    "hwm": self._cb_state.get("hwm", equity),
                    "killed": self._cb_state.get("killed", False),
                    "health_last_trip_ts": self._cb_state.get("health_last_trip_ts", 0),
                }
                self._circuit_open = False
                self._save_cb_state()
                self.logger.info(f"🔁 [Circuit Breaker] 새날 기준자산: {equity:.2f} USDT (임계 {self.CIRCUIT_BREAKER_ROE:.1f}%)")
            
            anchor = float(self._cb_state.get("anchor_equity", equity))
            if anchor <= 0:
                return
                
            chg_pct = (equity - anchor) / anchor * 100
            if not self._circuit_open and chg_pct <= self.CIRCUIT_BREAKER_ROE:
                self._circuit_open = True
                self._cb_state["tripped"] = True
                cooldown_hours = getattr(self, "CIRCUIT_BREAKER_COOLDOWN_HOURS", 48)
                self._cb_state["cooldown_until"] = now_ts + (cooldown_hours * 3600)
                self._save_cb_state()
                self.logger.warning(
                    f"🚨 [Circuit Breaker] 일손실 {chg_pct:.2f}% ≤ 임계 {self.CIRCUIT_BREAKER_ROE:.1f}% "
                    f"— {cooldown_hours}시간 진입 차단 (기준 {anchor:.0f} → 현재 {equity:.0f} USDT)"
                )
                try:
                    from utils_telegram import send_telegram_alert
                    send_telegram_alert(
                        f"🚨 [{self.STRATEGY_NAME}] 서킷 브레이커 발동: 일손실 {chg_pct:.2f}% — {cooldown_hours}시간 진입 차단"
                    )
                except Exception:
                    pass
            elif self._circuit_open and not self._cb_state.get("tripped", False):
                self._circuit_open = False
                
        except Exception as e:
            self.logger.warning(f"⚠️ [{self.STRATEGY_NAME}] 서킷 브레이커 체크 실패(기존 상태 유지): {e}")

    async def _check_htf_trend(self, symbol: str) -> dict:
        """
        [Fix] 상위 타임프레임(1h) EMA 추세 확인.
        Returns: {'ema_slope': float, 'is_uptrend': bool, 'is_downtrend': bool}
        5분 캐시 적용.
        """
        now = time.time()
        cached = self._htf_cache.get(symbol)
        if cached and (now - cached[0]) < 300:  # 5분 캐시
            return cached[1]

        result = {'ema_slope': 0.0, 'is_uptrend': False, 'is_downtrend': False, 'above_ema50': True}
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, self.HTF_TIMEFRAME, limit=60)
            if ohlcv and len(ohlcv) >= 50:
                closes = pd.Series([c[4] for c in ohlcv], dtype='float64')
                ema = closes.ewm(span=self.HTF_EMA_PERIOD, adjust=False).mean()
                # 최근 5캔들 기울기 (1h 기준 5시간)
                slope = (ema.iloc[-1] - ema.iloc[-5]) / ema.iloc[-5]
                result['ema_slope'] = slope
                result['is_uptrend'] = slope > self.HTF_SLOPE_THRESHOLD
                result['is_downtrend'] = slope < -self.HTF_SLOPE_THRESHOLD
                # [K안] 종목 게이트: 현재가 > 1h EMA50
                result['above_ema50'] = bool(closes.iloc[-1] > ema.iloc[-1])
        except Exception as e:
            self.logger.warning(f"⚠️ HTF 추세 체크 실패 ({symbol}): {e}")

        self._htf_cache[symbol] = (now, result)
        return result

    # ── [수정 3] 서바이벌 모드 ──
    # 연속 손실 시 자동으로 포지션 축소
    _survival_state = {'active': False, 'consecutive_losses': 0, 'activated_at': 0, 'cooldown_until': 0}

    def _check_survival_mode(self) -> bool:
        """최근 10건 라운드트립에서 연속 손실 체크.
        5연패 이상 → 서바이벌 모드 (마진 50% 축소)
        24h 무손실 → 해제"""
        now = time.time()
        if now < self._survival_state['cooldown_until']:
            return self._survival_state['active']

        try:
            path = os.path.join(BASE_DIR, "state", "trades.jsonl")
            if not os.path.exists(path):
                return False
            
            # 최근 10건의 라운드트립 PnL 추출
            pnls = []
            positions = {}
            with open(path, encoding='utf-8') as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = r.get('ts', 0)
                    if now - ts > 86400 * 7:  # 최근 7일만
                        continue
                    sym = r.get('symbol', '')
                    side = r.get('side', '')
                    px = float(r.get('price') or 0)
                    amt = float(r.get('amount') or 0)
                    
                    if side in ('BUY', 'SELL'):
                        if sym not in positions:
                            positions[sym] = {'qty': 0, 'avg': 0, 'side': 'long' if side == 'BUY' else 'short'}
                        p = positions[sym]
                        old_qty = p['qty']
                        p['avg'] = (p['avg'] * old_qty + px * amt) / (old_qty + amt) if (old_qty + amt) > 0 else px
                        p['qty'] = old_qty + amt
                        p['side'] = 'long' if side == 'BUY' else 'short'
                    elif side in ('CLOSE_LONG', 'CLOSE_SHORT'):
                        if sym in positions:
                            p = positions[sym]
                            cq = p['qty'] if amt == 0 else min(amt, p['qty'])
                            if cq > 0 and p['avg'] > 0:
                                sgn = 1 if p['side'] == 'long' else -1
                                pnl = (px - p['avg']) * cq * sgn
                                pnls.append(pnl)
                            p['qty'] = 0
            
            if len(pnls) < 3:
                return self._survival_state['active']
            
            # 최근 10건에서 연속 패배 카운트
            recent = pnls[-10:]
            consecutive_losses = 0
            for pnl in reversed(recent):
                if pnl <= 0:
                    consecutive_losses += 1
                else:
                    break
            
            self._survival_state['consecutive_losses'] = consecutive_losses
            
            # 연속 5패 이상 → 서바이벌 모드 활성화
            if consecutive_losses >= 5 and not self._survival_state['active']:
                self._survival_state['active'] = True
                self._survival_state['activated_at'] = now
                self._save_survival_state()
                self.logger.warning(f"🛡️ [서바이벌] 연속 {consecutive_losses}패 → 마진 50% 축소")
            
            # 서바이벌 중且 24h 무손실 → 해제
            if self._survival_state['active']:
                if any(p > 0 for p in recent[-5:]):
                    if now - self._survival_state['activated_at'] > 86400:  # 24h
                        self._survival_state['active'] = False
                        self._survival_state['cooldown_until'] = now + 86400
                        self._save_survival_state()
                        self.logger.info(f"✅ [서바이벌] 해제 — 정상 모드 복귀")
            
            return self._survival_state['active']
        except Exception:
            return False

    def _save_survival_state(self):
        try:
            path = os.path.join(BASE_DIR, "state", "survival_state.json")
            with open(path, 'w') as f:
                json.dump(self._survival_state, f)
        except Exception:
            pass

    def _load_survival_state(self):
        try:
            path = os.path.join(BASE_DIR, "state", "survival_state.json")
            if os.path.exists(path):
                with open(path) as f:
                    self._survival_state = json.load(f)
        except Exception:
            pass

    async def _detect_season_mode(self) -> str:
        """BTC 변동성과 추세로 시장 레짐 감지 → 시즌 모드 전환.
        normal → trend_up / trend_down / chop / crash"""
        if not self.SEASON_MODE_ENABLED:
            return self.SEASON_MODE_STATE
        now = time.time()
        if now - getattr(self, '_season_mode_ts', 0) < self.SEASON_MODE_COOLDOWN:
            return self.SEASON_MODE_STATE
        self._season_mode_ts = now
        try:
            ohlcv = await self.exchange.fetch_ohlcv('BTC/USDT:USDT', '1h', limit=48)
            if not ohlcv or len(ohlcv) < 24:
                return self.SEASON_MODE_STATE
            import statistics
            closes = [c[4] for c in ohlcv]
            returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
            vol = statistics.stdev(returns) * 100
            # 최근 24h 변동성 기준
            recent_24h = closes[-24:]
            trend = (closes[-1] - recent_24h[0]) / recent_24h[0] * 100
            # 이동평균 기울기로 추세 판단
            avg_24 = sum(recent_24h[-12:]) / 12
            avg_prev = sum(recent_24h[:12]) / 12
            slope = (avg_24 - avg_prev) / avg_prev * 100

            new_mode = self.SEASON_MODE_STATE
            if vol > 1.5 or abs(trend) > 5:
                new_mode = 'crash' if abs(trend) > 5 else ('trend_up' if trend > 0 else 'trend_down')
            elif vol > 0.8 or abs(slope) < 0.1:
                new_mode = 'chop'
            else:
                new_mode = 'trend_up' if slope > 0.1 else 'trend_down'

            if new_mode != self.SEASON_MODE_STATE:
                self.SEASON_MODE_STATE = new_mode
                self.logger.info(f"🌸 [시즌모드] {new_mode} 감증 (변동성 {vol:.2f}%, 추세 {trend:.1f}%)")
                # 모드에 따른 설정 변경
                if new_mode == 'trend_up':
                    self.SEASON_SIZE_MULT = self.SEASON_TREND_SIZE_MULT
                    self.SEASON_MAX_POS = self.SEASON_TREND_MAX_POS
                elif new_mode == 'trend_down':
                    self.SEASON_SIZE_MULT = self.SEASON_TREND_SIZE_MULT * 0.7
                    self.SEASON_MAX_POS = self.SEASON_TREND_MAX_POS
                elif new_mode == 'chop':
                    self.SEASON_SIZE_MULT = self.SEASON_CHOP_SIZE_MULT
                    self.SEASON_MAX_POS = self.SEASON_CHOP_MAX_POS
                elif new_mode == 'crash':
                    self.SEASON_SIZE_MULT = self.SEASON_CRASH_SIZE_MULT
                    self.SEASON_MAX_POS = self.SEASON_CRASH_MAX_POS
                # MAX_POSITIONS 동적 갱신
                self.MAX_OPEN_POSITIONS = self.SEASON_MAX_POS
        except Exception:
            pass
        return self.SEASON_MODE_STATE

    async def _detect_btc_direction(self) -> str:
        """BTC 1h EMA 기울기로 시장 방향 감지 → 동적 롱/숏 전환."""
        if not self.DYNAMIC_DIRECTION:
            return self._btc_direction
        now = time.time()
        if now - self._btc_direction_ts < 1800:  # 30분 캐시
            return self._btc_direction
        self._btc_direction_ts = now
        try:
            ohlcv = await self.exchange.fetch_ohlcv('BTC/USDT:USDT', '1h', limit=48)
            if not ohlcv or len(ohlcv) < 24:
                return self._btc_direction
            closes = [c[4] for c in ohlcv]
            import statistics
            ema = pd.Series(closes).ewm(span=50, adjust=False).mean().to_numpy()
            # 최근 24캔들 기울기
            recent = closes[-24:]
            recent_ema = ema[-24:]
            slope = (recent_ema[-1] - recent_ema[0]) / recent_ema[0] * 100
            self._btc_ema_slope = slope
            # ADX로 추세 강도 확인
            self._btc_adx = await self._calc_adx(ohlcv) if hasattr(self, '_calc_adx') else 20

            direction = self._btc_direction
            if abs(slope) > self.BTC_DIRECTION_THRESHOLD and self._btc_adx > 20:
                direction = 'long' if slope > 0 else 'short'
            elif abs(slope) < self.BTC_DIRECTION_THRESHOLD * 0.5 or self._btc_adx < 20:
                direction = 'chop'
            else:
                direction = 'long' if slope > 0 else 'short'

            if direction != self._btc_direction:
                self._btc_direction = direction
                self.logger.info(f"🎯 [승부사] 방향 전환: {direction} (EMA 기울기 {slope:.2f}%, ADX {self._btc_adx:.1f})")
        except Exception:
            pass
        return self._btc_direction

    async def _calc_adx(self, ohlcv, period=14):
        """간단 ADX 계산."""
        try:
            import pandas as pd
            import numpy as np
            df = pd.DataFrame({'high': [c[2] for c in ohlcv], 'low': [c[3] for c in ohlcv], 'close': [c[4] for c in ohlcv]})
            plus_dm = df['high'].diff()
            minus_dm = -df['low'].diff()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
            tr = np.maximum(df['high'] - df['low'], np.maximum(np.abs(df['high'] - df['close'].shift()), np.abs(df['low'] - df['close'].shift())))
            atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
            atr = atr.replace(0, np.nan).fillna(1)
            plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
            minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean()
            return float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 20.0
        except Exception:
            return 20.0

    async def run_auto_trade_loop(self):
        self.logger.info(f"🚀 [{self.STRATEGY_NAME}] 자동매매 엔진 시작 (Supertrend + StochRSI)")
        self._load_dca_state()
        self._load_cb_state()
        _config_check_counter = 0
        _debug_scores = {}  # [Fix #3] 디버그용 점수 수집
        while True:
            try:
                _config_check_counter += 1
                # [Fix] 사이클 단위 예약 마진/거부 로그 플래그 초기화
                self._reserved_margin = 0.0
                self._margin_reject_logged = False
                self._entry_budget_logged = False
                symbols = await self.get_target_symbols()
                if not symbols:
                    if _config_check_counter % 10 == 1:
                        self.logger.warning(
                            f"💓 [HEARTBEAT] 사이클 #{_config_check_counter} | "
                            f"대상 심볼 0개 — 심볼 로드 실패 또는 필터 조건 미충족"
                        )
                    await asyncio.sleep(60)
                    continue

                positions = await self.exchange.fetch_positions()
                self.auto_active_pos = {}
                for p in positions:
                    try:
                        contracts_str = p.get('contracts')
                        if contracts_str is not None and float(contracts_str) > 0:
                            sym = p.get('symbol')
                            s = p.get('side')
                            # [Fix] avgPrice=0 버그 수정: entryPrice → info.avgPx → avgPrice → price 폴백 체인
                            raw_avg = (
                                p.get('entryPrice')
                                or p.get('info', {}).get('avgPx')
                                or p.get('avgPrice')
                                or p.get('price')
                                or 0
                            )
                            avg_price = float(raw_avg) if raw_avg else 0.0
                            if avg_price <= 0:
                                self.logger.warning(
                                    f"⚠️ [{self.STRATEGY_NAME}] avgPrice=0 감지: {sym} ({s}) "
                                    f"— entryPrice={p.get('entryPrice')}, "
                                    f"avgPx={p.get('info', {}).get('avgPx')}, "
                                    f"avgPrice={p.get('avgPrice')}, price={p.get('price')}"
                                )
                            self.auto_active_pos[(sym, s)] = {
                                'size': float(contracts_str),
                                'avgPrice': avg_price,
                            }
                    except (ValueError, TypeError):
                        pass

                # 시장 레짐 필터 갱신 (사이클당 1회, 단일 fetch)
                await self._update_regime()
                # [Fix #1/#2] 횡보장 필터 & 서킷 브레이커 갱신 (사이클당 1회)
                await self._update_chop_filter()
                await self._update_circuit_breaker()
                # [왕복필터] 저승률 종목 신규 진입 제외 갱신 (내부 10분 캐시)
                self._refresh_churn_blacklist()
                # [Alpha ④] BTC 직전 확정 봉 변동률 추적 (베타 래그 보너스용)
                if self.BTC_BETA_LAG_ENABLED:
                    try:
                        _btc_bars = await self.exchange.fetch_ohlcv('BTC/USDT:USDT', self.TIMEFRAME, limit=3)
                        if _btc_bars and len(_btc_bars) >= 2:
                            _b = _btc_bars[-2]  # 직전 확정 봉
                            if _b['o']:
                                self._btc_move_15m = (_b['c'] - _b['o']) / _b['o'] * 100
                    except Exception:
                        pass

                for symbol in symbols:
                    await self.check_auto_logic(symbol)
                    await asyncio.sleep(0.1)

                if _config_check_counter % 10 == 0 and self.config:
                    self.config.refresh()
                self._save_dca_state()

                # [Fix #3] 10사이클마다 디버그 로그 출력 — 진입 안 되는 원인 진단용
                if _config_check_counter % 10 == 0:
                    pos_count = len(set(sym for sym, _ in self.auto_active_pos.keys()))
                    self.logger.info(
                        f"📊 [DEBUG] 사이클 #{_config_check_counter} | "
                        f"심볼 수: {len(symbols)} | 포지션: {pos_count}/{self.MAX_OPEN_POSITIONS} | "
                        f"상위 심볼: {symbols[:3]}"
                    )

            except Exception as e:
                self.logger.error(f"❌ [{self.STRATEGY_NAME}] 예외 발생: {e}")

            await asyncio.sleep(self.AUTO_TRADE_INTERVAL)

    async def run_all(self):
        await self.init_session()
        if hasattr(self, 'jev_filter') and self.jev_filter:
            try:
                target_symbols = await self.get_target_symbols()
                if target_symbols:
                    self.jev_filter.register_symbols(target_symbols)
                await self.jev_filter.initialize()
                self.logger.info(f"⚡ [Jev AI] LOB WebSocket 구독 및 피드 가동 완료 ({len(target_symbols) if target_symbols else 0}개 심볼)")
            except Exception as ex_jev_init:
                self.logger.warning(f"⚠️ JevFilter 시작 예외: {ex_jev_init}")
        try:
            await self.run_auto_trade_loop()
        finally:
            if hasattr(self, 'jev_filter') and self.jev_filter:
                try:
                    await self.jev_filter.close()
                except Exception:
                    pass
            await self.close_session()
