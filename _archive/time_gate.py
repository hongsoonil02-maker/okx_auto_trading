# -*- coding: utf-8 -*-
"""
time_gate.py — 시장별 진입 허용 구간 및 강제청산 트리거
Morning Sniper 전략의 시간 제어 게이트.
"""
from datetime import datetime, time
from zoneinfo import ZoneInfo
from enum import Enum

KST = ZoneInfo("Asia/Seoul")
EST = ZoneInfo("America/New_York")


class MarketSession(str, Enum):
    FORMING    = "FORMING"     # ORB 레인지 형성 중 (장 시작 ~ +15분)
    ENTRY_OK   = "ENTRY_OK"    # 진입 허용 구간
    NO_NEW_ENTRY = "NO_NEW_ENTRY"  # 신규 진입 금지 (청산만 허용)
    FORCE_EXIT = "FORCE_EXIT"  # 강제청산 시간
    CLOSED     = "CLOSED"      # 장 종료 / 비거래 시간


class TimeGate:
    """
    시장별 시간 제어 게이트.

    KR Stock  : 09:00~09:15 레인지 형성 / 09:15~10:20 진입 OK / 10:30 강제청산
    Upbit     : 09:00~09:15 레인지 형성 / 09:15~10:20 진입 OK / 10:30 강제청산
    Nasdaq    : 09:30~09:45 레인지 형성 / 09:45~10:50 진입 OK / 11:00 강제청산 (EST)
    OKX       : 제한 없음 (24시간 연속), 단 매 1시간 재스캔
    """

    # ── KR Stock ────────────────────────────────────────────────────────────
    KR_OPEN          = time(9,  0)
    KR_ORB_END       = time(9, 15)   # ORB 레인지 확정
    KR_ENTRY_CLOSE   = time(10, 20)  # 신규 진입 마감
    KR_FORCE_EXIT    = time(10, 30)  # 강제청산

    # ── Upbit (KST 기준 단타) ────────────────────────────────────────────────
    UPBIT_OPEN        = time(9,  0)
    UPBIT_ORB_END     = time(9, 15)
    UPBIT_ENTRY_CLOSE = time(10, 20)
    UPBIT_FORCE_EXIT  = time(10, 30)

    # ── Nasdaq (EST 기준) ────────────────────────────────────────────────────
    US_OPEN          = time(9, 30)
    US_ORB_END       = time(9, 45)
    US_ENTRY_CLOSE   = time(10, 50)
    US_FORCE_EXIT    = time(11,  0)

    @staticmethod
    def _now_kst() -> datetime:
        return datetime.now(tz=KST)

    @staticmethod
    def _now_est() -> datetime:
        return datetime.now(tz=EST)

    @classmethod
    def kr_stock_session(cls) -> MarketSession:
        """현재 KR Stock 세션 상태 반환"""
        t = cls._now_kst().time()
        # 주말 거래 없음
        if cls._now_kst().weekday() >= 5:
            return MarketSession.CLOSED
        if t < cls.KR_OPEN:
            return MarketSession.CLOSED
        if t < cls.KR_ORB_END:
            return MarketSession.FORMING
        if t < cls.KR_ENTRY_CLOSE:
            return MarketSession.ENTRY_OK
        if t < cls.KR_FORCE_EXIT:
            return MarketSession.NO_NEW_ENTRY
        return MarketSession.FORCE_EXIT

    @classmethod
    def upbit_session(cls) -> MarketSession:
        """현재 Upbit 단타 세션 상태 반환 (KST 09:00~10:30 집중)"""
        t = cls._now_kst().time()
        if cls._now_kst().weekday() >= 5:  # 업비트는 주말도 가능하지만 단타는 제한
            return MarketSession.CLOSED
        if t < cls.UPBIT_OPEN:
            return MarketSession.CLOSED
        if t < cls.UPBIT_ORB_END:
            return MarketSession.FORMING
        if t < cls.UPBIT_ENTRY_CLOSE:
            return MarketSession.ENTRY_OK
        if t < cls.UPBIT_FORCE_EXIT:
            return MarketSession.NO_NEW_ENTRY
        return MarketSession.FORCE_EXIT

    @classmethod
    def nasdaq_session(cls) -> MarketSession:
        """현재 Nasdaq 세션 상태 반환 (EST 기준)"""
        t = cls._now_est().time()
        if cls._now_est().weekday() >= 5:
            return MarketSession.CLOSED
        if t < cls.US_OPEN:
            return MarketSession.CLOSED
        if t < cls.US_ORB_END:
            return MarketSession.FORMING
        if t < cls.US_ENTRY_CLOSE:
            return MarketSession.ENTRY_OK
        if t < cls.US_FORCE_EXIT:
            return MarketSession.NO_NEW_ENTRY
        return MarketSession.FORCE_EXIT

    @classmethod
    def okx_session(cls) -> MarketSession:
        """OKX는 24시간 항상 진입 허용"""
        return MarketSession.ENTRY_OK

    @classmethod
    def is_entry_allowed(cls, market: str) -> bool:
        """진입 허용 여부 (True = 진입 가능)"""
        session_map = {
            "kr_stock": cls.kr_stock_session,
            "upbit":    cls.upbit_session,
            "nasdaq":   cls.nasdaq_session,
            "okx":      cls.okx_session,
        }
        fn = session_map.get(market.lower())
        if fn is None:
            return False
        return fn() == MarketSession.ENTRY_OK

    @classmethod
    def is_force_exit(cls, market: str) -> bool:
        """강제청산 시간 여부 (True = 즉시 전량청산)"""
        session_map = {
            "kr_stock": cls.kr_stock_session,
            "upbit":    cls.upbit_session,
            "nasdaq":   cls.nasdaq_session,
            "okx":      cls.okx_session,  # 항상 False
        }
        fn = session_map.get(market.lower())
        if fn is None:
            return False
        return fn() == MarketSession.FORCE_EXIT

    @classmethod
    def is_orb_forming(cls, market: str) -> bool:
        """ORB 레인지 형성 중 여부"""
        session_map = {
            "kr_stock": cls.kr_stock_session,
            "upbit":    cls.upbit_session,
            "nasdaq":   cls.nasdaq_session,
            "okx":      cls.okx_session,
        }
        fn = session_map.get(market.lower())
        if fn is None:
            return False
        return fn() == MarketSession.FORMING

    @classmethod
    def get_session_summary(cls) -> dict:
        """모든 시장의 현재 세션 상태 요약"""
        return {
            "KR_Stock":  cls.kr_stock_session().value,
            "Upbit":     cls.upbit_session().value,
            "Nasdaq":    cls.nasdaq_session().value,
            "OKX":       cls.okx_session().value,
            "KST":       cls._now_kst().strftime("%H:%M:%S"),
            "EST":       cls._now_est().strftime("%H:%M:%S"),
        }


if __name__ == "__main__":
    summary = TimeGate.get_session_summary()
    print("=" * 40)
    print("현재 시장 세션 상태")
    print("=" * 40)
    for k, v in summary.items():
        print(f"  {k:12s}: {v}")
    print("=" * 40)
    for market in ["kr_stock", "upbit", "nasdaq", "okx"]:
        entry = TimeGate.is_entry_allowed(market)
        force = TimeGate.is_force_exit(market)
        orb   = TimeGate.is_orb_forming(market)
        print(f"  {market:10s} | 진입={entry} | 강제청산={force} | ORB형성={orb}")
