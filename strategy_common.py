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
    atr = (df['h'].combine(df['c'].shift(), max) - df['l'].combine(df['c'].shift(), min)).rolling(period).mean()

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
    MASTER_WEBHOOK_URL = "http://localhost:8009/webhook"
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
    HTF_SLOPE_THRESHOLD = 0.005  # HTF EMA 기울기 임계값 (0.5%)
    # [개선안 #1] Volume 확인 배수 — 서브클래스에서 오버라이드 가능
    # 추가 전략 파라미터 (AI 토너먼트 1등 Alpha_Trend 파라미터 적용)
    PROFIT_THRESHOLD = 1.03  # 3% 수익 구간부터 청산 고려
    VOL_CONFIRM_MULT = 1.0   # 거래량 급증 확인 배수 (저변동 장세 대응 1.2→1.0 완화)
    MIN_HOLD_CANDLES = 3     # 최소 보유 캔들 수
    EMA_PERIOD = 50          # 추세 필터 기간
    # [안전망 추가] 긴급 하드 스탑로스: 현물 기준 -5% (레버리지 10x 적용 시 PnL -50%) 
    HARD_STOP_LOSS_PCT = float(os.getenv("OKX_HARD_STOP_LOSS", "-0.30"))  # 손익비 개선: -50%→-30%
    SOFT_STOP_LOSS_PCT = float(os.getenv("OKX_SOFT_STOP_LOSS", "-0.12"))  # 조기 손절(가랑비 방어). -1 = 비활성 [8/26 백테스트]
    HARD_STOP_COOLDOWN_HOURS = 12  # 하드 스탑 시 12시간 쿨다운 (뇌동매매 방지)
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
    REENTRY_COOLDOWN_CANDLES = 4
    
    # [포트폴리오 리밸런싱] 자본 배분 가중치 및 불타기(Pyramid) 비율
    PORTFOLIO_WEIGHT = 1.0
    PYRAMID_RATIO = 0.35

    # [백테스트 검증] 트레일링 전량청산 시 즉시 반대방향 진입
    # Major 30m: PF 0.76→1.22 (흑자 전환) / Venture 15m: PF 1.23→1.13 (악화)
    # → 메이저처럼 박스권 성격 심볼에만 활성화
    FLIP_ON_TRAILING_CLOSE = False
    # [Fix] 분할익절 3단계 (기존 1/8씩 8단계는 수익 실현이 너무 느림)
    SCALE_OUT_STEPS = 3
    # [Fix] DCA 추가 진입 최소 간격 (캔들 수) — 매 캔들 물타기는 수수료 출혈
    DCA_MIN_CANDLES = 4
    # [검증] 물타기/피라미딩 백테스트(6~8월): 현행 조합 최악(-3.7K, MDD 83.7%) vs 둘다없음 최고(+18.1K, 48%)
    # DCA는 평균손실 -199→-274 확대, 피라미딩+20%가드는 승자 절단 → 기본 비활성 (env로 재활성 가능)
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
    ATR_TRAIL_K = float(os.getenv("OKX_ATR_TRAIL_K", "2.5"))
    ATR_TRAIL_ARM_PNL = float(os.getenv("OKX_ATR_TRAIL_ARM_PNL", "0.20"))  # 마진수익 20% 도달 후 작동
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
    # ── [Fix #2] 일손실 서킷 브레이커 ──
    # 당일 자산이 기준(일 시작 자산) 대비 임계값 이하로 하락하면 신규 진입 차단.
    # 회복(임계값의 절반 이상) 또는 다음 날(UTC) 자동 해제. 상태는 파일로 영속화(재시작 대비).
    CIRCUIT_BREAKER_ENABLED = os.getenv("OKX_CIRCUIT_BREAKER", "true").lower() == "true"
    CIRCUIT_BREAKER_ROE = float(os.getenv("OKX_CIRCUIT_BREAKER_ROE", "-6"))  # 일 시작 자산 대비 %
    # ── 포지션별 손실 한도 ──
    # 포지션 레벨 손실이 이 수준 이하로 떨어지면 추세/레짐 무관하게 무조건 청산 (빠른 차단).
    # 하드스탑(-30%)보다 빨리 발동해 횡보장 물타기 누적을 방지. 백테스트(F 변형) 검증 완료.
    POSITION_LOSS_LIMIT = float(os.getenv("OKX_POSITION_LOSS_LIMIT", "-0.15"))
    # ── [Fix #3] 노출 스케일 ──
    # 심볼당 목표 마진에 곱하는 전역 배율. 정점 대비 -27% 손실 후 총 노출 축소(마진 사용률 84% → ~50% 목표).
    EXPOSURE_SCALE = float(os.getenv("OKX_EXPOSURE_SCALE", "0.6"))
    # ── 수익성 향상: 컨빅션 사이징 ──
    # 진입 점수에 비례해 포지션 크기를 0.5x~2x로 조절. 고점수 셋업(강한 추세)에 몰빵 → 대수익 극대화.
    # 백테스트(08-18~23) 검증: 컨빅션+베어숏 조합 시 전체 +90 → +80,721.
    CONVICTION_SIZING_ENABLED = os.getenv("OKX_CONVICTION_SIZING", "true").lower() == "true"
    CONVICTION_MAX_MULT = float(os.getenv("OKX_CONVICTION_MAX_MULT", "2.0"))
    CONVICTION_MIN_MULT = float(os.getenv("OKX_CONVICTION_MIN_MULT", "0.5"))
    # ── 수익성 향상: 베어 숏 게이팅 ──
    # 불장(BTC>=EMA200)에서 숏 차단 → 불장 숏 손실 원천 방지. 베어장에서만 숏 허용.
    BEAR_SHORT_ENABLED = os.getenv("OKX_BEAR_SHORT", "true").lower() == "true"

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
        # Alpha Stack 상태
        self._funding_cache = {}          # sym -> (ts, rate) 30분 TTL
        self._btc_move_15m = 0.0          # 직전 확정 봉 BTC 변동률% (베타 래그용)
        self._btc_above_ema50_1h = True   # [K안] 시장 게이트 상태 (첫 갱신 전 fail-open)
        self._cb_state = {}

    def _is_trading_hour_allowed(self) -> bool:
        """DEPRECATED: Bots now run 24/7 relying purely on technical indicators."""
        return True

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
        if not self.LONG_DUAL_GATE:
            return True
        if not getattr(self, '_btc_above_ema50_1h', True):
            return False
        htf = await self._check_htf_trend(symbol)
        return bool(htf.get('above_ema50', True))

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
        }
        ratio = type_ratios.get(entry_type, 0.50)
        target_margin = base_margin * ratio

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

    async def send_webhook(self, side: SideType, symbol: str, qty: float):
        payload = WebhookPayload(
            action=ActionType.EXEC,
            side=side,
            symbol=symbol,
            qty=qty,
            signal_strength="STRONG",
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

            prev, curr = df.iloc[-2], df.iloc[-1]
            t_curr = curr['t']
            # [Fix #1/#2] 횡보장/서킷 브레이커 발동 시 자본 투입 전면 차단 (청산·스탑은 계속 동작)
            entries_blocked = self._chop_block or self._circuit_open
            # [K안] 롱 이중 게이트 (BTC 1h EMA50 + 심볼 1h EMA50, 5분 캐시)
            dual_gate = await self._long_dual_gate_ok(symbol)
            dca = self.dca_state.setdefault(symbol, {'entry_count': 0, 'exit_count': 0, 'last_entry_t': 0, 'last_exit_t': 0, 'first_entry_t': 0, 'max_pnl_pct': 0.0})

            pos_long = self.auto_active_pos.get((symbol, 'long'))
            has_long = pos_long is not None
            avg_price_long = pos_long['avgPrice'] if has_long else 0

            pos_short = self.auto_active_pos.get((symbol, 'short'))
            has_short = pos_short is not None
            avg_price_short = pos_short['avgPrice'] if has_short else 0

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

            # [버그 수정] curr['v']는 미완성 캔들이라 거래량이 턱없이 부족할 때가 많습니다. 
            # 따라서 직전 완성 캔들(prev['v'])에서 이미 거래량이 터졌거나, 
            # 현재 캔들에서 벌써 거래량 조건을 만족한 경우를 모두 인정합니다.
            vol_cond = (curr['v'] > prev['vol_ma'] * self.VOL_CONFIRM_MULT) or \
                       (prev['v'] > df.iloc[-3]['vol_ma'] * self.VOL_CONFIRM_MULT)

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
            if self.MOM_ROTATION_ENABLED and _adx_now >= self.MOM_MIN_ADX and len(df) > self.MOM_ROC_LOOKBACK + 1:
                roc_ref = float(df['c'].iloc[-1 - self.MOM_ROC_LOOKBACK])
                atr_pct = max(float(curr['atr']) / curr['c'], 1e-9)
                if roc_ref > 0 and atr_pct > 0:
                    ra_mom = ((curr['c'] - roc_ref) / roc_ref) / atr_pct
                    if ra_mom >= self.MOM_RISK_ADJ_THRESHOLD:
                        long_score += self.MOM_BONUS
                    elif ra_mom <= -self.MOM_RISK_ADJ_THRESHOLD:
                        short_score += self.MOM_BONUS

            # [Alpha ②] 스퀴즈 브레이크아웃: BB 폭 압축 해제 + 방향성 돌파 (ADX≥18 — 심충보에선 함정 신호)
            if self.SQUEEZE_SIGNAL_ENABLED and _adx_now >= self.SQUEEZE_MIN_ADX and 'bb_width' in df.columns and len(df) > 105:
                w_ref = df['bb_width'].iloc[-100:-3].min()
                prev_squeeze = float(df['bb_width'].iloc[-4]) <= w_ref * 1.1
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

            # [Fix] 비대칭 임계값: 롱 70, 숏 90 (숏은 구조적으로 위험하므로 엄격)
            # [섹터별 임계값] 종목 섹터에 따라 진입 점수 기준 차등 적용
            _sec_p = self._sector_params(symbol)
            ENTRY_THRESHOLD_LONG = _sec_p.get('thr_long', 70)
            ENTRY_THRESHOLD_SHORT = ENTRY_THRESHOLD_LONG + 30  # 숏 비대칭 강화 (기존 +20→+30): 숏 PF 0.95 적자 대응
            is_long_sig = (long_score >= ENTRY_THRESHOLD_LONG) and vol_cond and getattr(self, '_long_regime_ok', True) and dual_gate
            is_short_sig = (short_score >= ENTRY_THRESHOLD_SHORT) and vol_cond and getattr(self, '_short_regime_ok', True)
            # [수익성] 베어 숏 게이팅: 불장(BTC>=EMA200)에서 숏 차단 → 숏 손실 원천 방지
            if self.BEAR_SHORT_ENABLED and is_short_sig and getattr(self, '_long_regime_ok', True):
                is_short_sig = False
            
            default_leverage = int(os.getenv("OKX_LEVERAGE", "10"))
            leverage = getattr(self, 'STRATEGY_LEVERAGE', default_leverage)
            
            pnl_pct_long = 0
            is_hard_stop_long = False
            take_profit_long_sig = False
            if has_long and avg_price_long > 0:
                pnl_pct_long = ((curr['c'] - avg_price_long) / avg_price_long) * leverage
                if pnl_pct_long > dca['max_pnl_pct']:
                    dca['max_pnl_pct'] = pnl_pct_long
                
                is_profit = curr['c'] > avg_price_long * self.PROFIT_THRESHOLD
                st_v_long = curr['st_v_tight'] if is_profit else curr['st_v_loose']
                st_d_long = curr['st_d_tight'] if is_profit else curr['st_d_loose']
                close_long_sig = st_d_long == -1 or curr['c'] < st_v_long
                force_close_long = False

                # [Alpha ①] ATR Chandelier 트레일링: 수익 구간에서 고점 − k×ATR 이탈 시 청산
                # (철칙 2: 변동성 클수록 ATR이 커져 선이 넓어지는 대신, 가격 이탈 즉시 반응)
                if self.ATR_TRAILING_ENABLED:
                    dca['highest_px'] = max(dca.get('highest_px') or avg_price_long, float(curr['h']))
                    _chand_l = dca['highest_px'] - self.ATR_TRAIL_K * float(curr['atr'])
                    if dca.get('max_pnl_pct', 0.0) >= self.ATR_TRAIL_ARM_PNL and curr['c'] < _chand_l:
                        self.logger.info(
                            f"🎯 [Chandelier] 롱 트레일링 청산: {symbol} "
                            f"(고점 {dca['highest_px']:.6g} − {self.ATR_TRAIL_K}×ATR, "
                            f"최고수익 {dca.get('max_pnl_pct',0)*100:.0f}%)"
                        )
                        force_close_long = True

                if pnl_pct_long <= self.HARD_STOP_LOSS_PCT:
                    force_close_long = True
                    is_hard_stop_long = True
                elif pnl_pct_long <= self.POSITION_LOSS_LIMIT:
                    self.logger.warning(f"🚨 [POSITION LIMIT] 롱 손실 한도 초과 ({pnl_pct_long*100:.1f}% ≤ {self.POSITION_LOSS_LIMIT*100:.0f}%): {symbol}")
                    force_close_long = True
                    is_hard_stop_long = True
                elif pnl_pct_long <= self.SOFT_STOP_LOSS_PCT:
                    self.logger.warning(f"🚨 [SOFT STOP] 롱 무조건 조기 손절 (손실률: {pnl_pct_long*100:.2f}%, 가랑비 원천 차단): {symbol}")
                    force_close_long = True
                    is_hard_stop_long = True  # 조기 손절도 쿨다운 적용을 위해 hard stop 취급
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
                elif pnl_pct_long >= 0.15 and dca['exit_count'] == 0:
                    take_profit_long_sig = True
                elif pnl_pct_long >= 0.30 and dca['exit_count'] == 1:
                    take_profit_long_sig = True
                elif pnl_pct_long >= 0.50 and dca['exit_count'] == 2:
                    take_profit_long_sig = True
                # ── [Winner Pyramiding] 추세 승자 롱 불타기 ──
                if self.PYRAMIDING_ENABLED and pnl_pct_long >= 0.40 and dca.get('pyramid_count', 0) == 0 and self._dca_ready(dca, t_curr) and not entries_blocked:
                    if is_ema_trend_up and curr['st_d_loose'] == 1:  # [Fix] Series 비교 → 스칼라 비교
                        self.logger.info(f"🔥 [Winner Pyramiding 1차 불타기] 롱 {symbol} (PnL: +{pnl_pct_long*100:.1f}%)")
                        await self.execute_auto_entry(symbol, SideType.BUY, entry_type="pyramid")
                        dca['pyramid_count'] = 1
                        dca['last_entry_t'] = t_curr
                elif self.PYRAMIDING_ENABLED and pnl_pct_long >= 1.00 and dca.get('pyramid_count', 0) == 1 and self._dca_ready(dca, t_curr) and not entries_blocked:
                    if is_ema_trend_up and is_long_momentum:
                        self.logger.info(f"🚀 [Winner Pyramiding 2차 불타기] 롱 {symbol} (PnL: +{pnl_pct_long*100:.1f}%)")
                        await self.execute_auto_entry(symbol, SideType.BUY, entry_type="pyramid")
                        dca['pyramid_count'] = 2
                        dca['last_entry_t'] = t_curr

                if dca.get('pyramid_count', 0) >= 1 and pnl_pct_long < 0.20:
                    force_close_long = True

                if dca['exit_count'] > 0 and curr['c'] < avg_price_long:
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
                pnl_pct_short = ((avg_price_short - curr['c']) / avg_price_short) * leverage
                if pnl_pct_short > dca['max_pnl_pct']:
                    dca['max_pnl_pct'] = pnl_pct_short
                
                is_profit = curr['c'] < avg_price_short * (2.0 - self.PROFIT_THRESHOLD)
                st_v_short = curr['st_v_tight'] if is_profit else curr['st_v_loose']
                st_d_short = curr['st_d_tight'] if is_profit else curr['st_d_loose']
                close_short_sig = st_d_short == 1 or curr['c'] > st_v_short
                force_close_short = False

                # [Alpha ①] ATR Chandelier 트레일링 (숏): 저점 + k×ATR 상향 돌파 시 청산
                if self.ATR_TRAILING_ENABLED:
                    dca['lowest_px'] = min(dca.get('lowest_px') or avg_price_short, float(curr['l']))
                    _chand_s = dca['lowest_px'] + self.ATR_TRAIL_K * float(curr['atr'])
                    if dca.get('max_pnl_pct', 0.0) >= self.ATR_TRAIL_ARM_PNL and curr['c'] > _chand_s:
                        self.logger.info(
                            f"🎯 [Chandelier] 숏 트레일링 청산: {symbol} "
                            f"(저점 {dca['lowest_px']:.6g} + {self.ATR_TRAIL_K}×ATR, "
                            f"최고수익 {dca.get('max_pnl_pct',0)*100:.0f}%)"
                        )
                        force_close_short = True

                if pnl_pct_short <= self.HARD_STOP_LOSS_PCT:
                    force_close_short = True
                    is_hard_stop_short = True
                elif pnl_pct_short <= self.POSITION_LOSS_LIMIT:
                    self.logger.warning(f"🚨 [POSITION LIMIT] 숏 손실 한도 초과 ({pnl_pct_short*100:.1f}% ≤ {self.POSITION_LOSS_LIMIT*100:.0f}%): {symbol}")
                    force_close_short = True
                    is_hard_stop_short = True
                elif pnl_pct_short <= self.SOFT_STOP_LOSS_PCT:
                    self.logger.warning(f"🚨 [SOFT STOP] 숏 무조건 조기 손절 (손실률: {pnl_pct_short*100:.2f}%, 가랑비 원천 차단): {symbol}")
                    force_close_short = True
                    is_hard_stop_short = True  # 조기 손절도 쿨다운 적용을 위해 hard stop 취급
                elif dca['max_pnl_pct'] >= 1.00 and pnl_pct_short <= dca['max_pnl_pct'] - 0.30:
                    force_close_short = True
                elif dca['max_pnl_pct'] >= 0.50 and pnl_pct_short <= dca['max_pnl_pct'] * 0.6:
                    force_close_short = True
                elif dca['max_pnl_pct'] >= 0.40 and pnl_pct_short <= 0.30:
                    force_close_short = True
                elif dca['max_pnl_pct'] >= 0.20 and pnl_pct_short <= 0.05:
                    force_close_short = True
                elif pnl_pct_short >= 0.15 and dca['exit_count'] == 0:
                    take_profit_short_sig = True
                elif pnl_pct_short >= 0.30 and dca['exit_count'] == 1:
                    take_profit_short_sig = True
                elif pnl_pct_short >= 0.50 and dca['exit_count'] == 2:
                    take_profit_short_sig = True
                # ── [Winner Pyramiding] 추세 승자 숏 불타기 ──
                if self.PYRAMIDING_ENABLED and pnl_pct_short >= 0.40 and dca.get('pyramid_count', 0) == 0 and self._dca_ready(dca, t_curr) and not entries_blocked:
                    if is_ema_trend_down and curr['st_d_loose'] == -1:  # [Fix] Series 비교 → 스칼라 비교
                        self.logger.info(f"📉 [Winner Pyramiding 숏 1차 불타기] {symbol} (PnL: +{pnl_pct_short*100:.1f}%)")
                        await self.execute_auto_entry(symbol, SideType.SELL, entry_type="pyramid")
                        dca['pyramid_count'] = 1
                        dca['last_entry_t'] = t_curr
                elif self.PYRAMIDING_ENABLED and pnl_pct_short >= 1.00 and dca.get('pyramid_count', 0) == 1 and self._dca_ready(dca, t_curr) and not entries_blocked:
                    if is_ema_trend_down and is_short_momentum:
                        self.logger.info(f"🚀 [Winner Pyramiding 숏 2차 불타기] {symbol} (PnL: +{pnl_pct_short*100:.1f}%)")
                        await self.execute_auto_entry(symbol, SideType.SELL, entry_type="pyramid")
                        dca['pyramid_count'] = 2
                        dca['last_entry_t'] = t_curr

                if dca.get('pyramid_count', 0) >= 1 and pnl_pct_short < 0.20:
                    force_close_short = True

                if dca['exit_count'] > 0 and curr['c'] > avg_price_short:
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
                    if self.FLIP_ON_TRAILING_CLOSE and not is_hard_stop_long and not entries_blocked:
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
                    if dca['entry_count'] < self.MAX_DCA_ENTRIES and self._dca_ready(dca, t_curr) and pnl_pct_long > -0.10 and not entries_blocked:
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
                    # [Flip] 트레일링/방어 청산 시 즉시 롱 진입 (하드스탑 제외, 레짐 필터 적용)
                    flipped = False
                    if self.FLIP_ON_TRAILING_CLOSE and not is_hard_stop_short and self._long_regime_ok and not entries_blocked and dual_gate:
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
                elif is_short_pullback and dca['entry_count'] < self.MAX_DCA_ENTRIES and self._dca_ready(dca, t_curr) and not entries_blocked:
                    self.logger.info(f"📉 [Short Pullback 진입] {symbol} (DCA {dca['entry_count']+1}/{self.MAX_DCA_ENTRIES})")
                    await self.execute_auto_entry(symbol, SideType.SELL, entry_type="dca")
                    dca['entry_count'] += 1
                    dca['last_entry_t'] = t_curr

            else:
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
                if self.REENTRY_ENABLED and dca.get('last_close_t') and dca.get('last_entry_t') != t_curr and not is_in_cooldown and not entries_blocked:
                    cooldown_ms = self.REENTRY_COOLDOWN_CANDLES * self.TIMEFRAME_MINUTES * 60 * 1000
                    if (t_curr - dca['last_close_t']) >= cooldown_ms:
                        side_closed = dca.get('last_close_side')
                        re_long = side_closed == 'long' and curr['st_d_loose'] == 1 and curr['c'] > curr['ema_target'] and self._long_regime_ok and dual_gate
                        re_short = side_closed == 'short' and curr['st_d_loose'] == -1 and curr['c'] < curr['ema_target']
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
                            return

                if is_long_sig and dca.get('last_entry_t') != t_curr and not is_in_cooldown and not entries_blocked:
                    self.logger.info(f"🟢 [Scoring System 신규 진입] {symbol} (Score: {long_score})")
                    await self.execute_auto_entry(symbol, SideType.BUY, entry_type="new", base_score=long_score)
                    dca['entry_count'] = 1
                    dca['pyramid_count'] = 0
                    dca['exit_count'] = 0
                    dca['last_entry_t'] = t_curr
                    dca['first_entry_t'] = t_curr
                    dca['side'] = 'long'
                elif is_short_sig and dca.get('last_entry_t') != t_curr and not is_in_cooldown and not entries_blocked:
                    # [Fix] HTF 추세 필터: 1h EMA50 상승 중이면 숏 진입 차단
                    htf = await self._check_htf_trend(symbol)
                    if htf['is_uptrend']:
                        self.logger.info(
                            f"🚫 [HTF Filter] 숏 진입 차단 — 1h EMA50 상승 중: {symbol} "
                            f"(기울기: {htf['ema_slope']*100:+.2f}%, Score: {short_score})"
                        )
                    else:
                        self.logger.info(f"🔴 [Scoring System 신규 진입] {symbol} (Score: {short_score}, HTF: {htf['ema_slope']*100:+.2f}%)")
                        await self.execute_auto_entry(symbol, SideType.SELL, entry_type="new", base_score=short_score)
                        dca['entry_count'] = 1
                        dca['pyramid_count'] = 0
                        dca['exit_count'] = 0
                        dca['last_entry_t'] = t_curr
                        dca['first_entry_t'] = t_curr
                        dca['side'] = 'short'

        except Exception as e:
            self.logger.error(f"⚠️ [{self.STRATEGY_NAME}] 로직 체크 실패 ({symbol}): {e}")

    async def execute_auto_entry(self, symbol: str, side: SideType, portion: float = 0.20, entry_type: str = "new", base_score: float = 70.0):
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
            entry_type: "new", "dca", "pyramid", "flip", "reentry"
            base_score: 진입 점수 (컨빅션 사이징에 사용, 기본 70)
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

            # [Bug Fix] 서브클래스에서 STRATEGY_LEVERAGE를 강제 오버라이드한 경우 이를 우선 적용
            default_leverage = int(os.getenv("OKX_LEVERAGE", "10"))
            leverage = getattr(self, 'STRATEGY_LEVERAGE', default_leverage)

            # ── [Fix] Equity 기반 균등 분할 사이징 ──
            target_margin = self._calc_target_margin(effective_free, total_usdt, entry_type)
            # [수익성] 컨빅션 사이징: 진입 점수에 비례 포지션 크기 조절 (0.5x~2x)
            if self.CONVICTION_SIZING_ENABLED and entry_type in ("new", "flip", "reentry"):
                conv_mult = max(self.CONVICTION_MIN_MULT, min(self.CONVICTION_MAX_MULT, base_score / 70.0))
                target_margin *= conv_mult
            # [단계적 배포] 약추세 구간(ADX 소프트 경계~임계값)에서는 축소 사이즈로 참여
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
                if fr is not None and abs(fr) >= min(abs(self.FUNDING_EXTREME_POS), abs(self.FUNDING_EXTREME_NEG)):
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

                self.logger.info(
                    f"🔥 [{self.STRATEGY_NAME}] 진입 시그널: {side.value} {symbol} "
                    f"(수량: {amount}, 목표마진: {target_margin:.1f} USDT, 유형: {entry_type})"
                )
                await self.send_webhook(side, symbol, amount)
                # [Fix] 사이클 내 후속 신호가 동일 마진을 중복 사용하지 않도록 예약 처리
                self._reserved_margin += target_margin
        except Exception as e:
            self.logger.error(f"⚠️ [{self.STRATEGY_NAME}] 진입 수량 계산 실패 ({symbol}): {e}")

    async def _update_regime(self):
        """
        [Fix] BTC 1h OHLCV 1회 fetch로 롱/숏 레짐 동시 갱신 (기존 2회 fetch → 1회).
        로깅은 레짐 전환 시에만 수행 (기존 매 사이클 중복 로그 스팸 제거).
        실패 시 기존 상태 유지(안전 측).
        """
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
            # [8/26 백테스트 복원] 극저변동(BTC 1h ADX<15) 신규 진입만 차단, 청산은 계속.
            # 90일 분할검증: 전반기 손실 축소/후반기 수익 확대/MDD 19->15%. 청산·트레일링은 항상 동작.
            self._chop_block = adx_now < 15.0
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
        히스테리시스: 임계값의 절반 이상 회복 시 해제. 다음 날(UTC) 자동 리셋.
        """
        if not self.CIRCUIT_BREAKER_ENABLED:
            self._circuit_open = False
            return
        try:
            balance = await self.exchange.fetch_balance()
            equity = float(balance.get('USDT', {}).get('total', 0) or balance.get('total', {}).get('USDT', 0) or 0)
            if equity <= 0:
                return
            today = datetime.utcnow().strftime("%Y-%m-%d")
            if self._cb_state.get("date") != today:
                self._cb_state = {"date": today, "anchor_equity": equity, "tripped": False}
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
                self._save_cb_state()
                self.logger.warning(
                    f"🚨 [Circuit Breaker] 일손실 {chg_pct:.2f}% ≤ 임계 {self.CIRCUIT_BREAKER_ROE:.1f}% "
                    f"— 신규 진입 차단 (기준 {anchor:.0f} → 현재 {equity:.0f} USDT)"
                )
                try:
                    from utils_telegram import send_telegram_alert
                    send_telegram_alert(
                        f"🚨 [{self.STRATEGY_NAME}] 서킷 브레이커 발동: 일손실 {chg_pct:.2f}% — 신규 진입 차단"
                    )
                except Exception:
                    pass
            elif self._circuit_open and chg_pct > self.CIRCUIT_BREAKER_ROE / 2:
                self._circuit_open = False
                self._cb_state["tripped"] = False
                self._save_cb_state()
                self.logger.info(f"✅ [Circuit Breaker] 일손익 {chg_pct:.2f}% 회복 — 진입 재개")
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
        try:
            await self.run_auto_trade_loop()
        finally:
            await self.close_session()
