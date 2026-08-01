# -*- coding: utf-8 -*-
"""
webhook_spec.py — 순수 명령 스키마 v2 (Brain/Body 완전 분리)
- 두 서버 공통 버전으로 통일
- WebhookValidator 추가 (하위 호환)
- signal_strength 필드 추가 (STRONG/WEAK)
"""
from dataclasses import dataclass, asdict, fields
from typing import Optional
from enum import Enum
import json
import uuid

class ActionType(str, Enum):
    EXEC = "EXEC"

class SideType(str, Enum):
    BUY         = "BUY"
    SELL        = "SELL"
    CLOSE_LONG  = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"

# 하위 호환성용 가짜 Enum
class MarketType(str, Enum):
    KR_STOCK = "KR_STOCK"
    NASDAQ = "NASDAQ"

class PositionMode(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"

@dataclass
class WebhookPayload:
    """
    체결 명령 스키마.
    API Key/비밀번호 등 민감정보는 Bot 내부 .env에서만 관리.
    """
    action:          ActionType
    side:            SideType
    symbol:          str
    qty:             float
    price:           Optional[float] = None   # 시장가의 경우 None
    flag:            Optional[str]   = None   # 보조 플래그
    signal_strength: Optional[str]  = None   # "STRONG" / "WEAK"
    signal_id:       str = ""  # 하위 호환성을 위해 유지

    def __post_init__(self):
        if not self.signal_id:
            self.signal_id = str(uuid.uuid4())

    @property
    def position_mode(self) -> PositionMode:
        if self.side in (SideType.BUY, SideType.CLOSE_SHORT):
            return PositionMode.LONG
        if self.side == SideType.SELL:
            return PositionMode.SHORT
        if self.side == SideType.CLOSE_LONG:
            return PositionMode.FLAT
        return PositionMode.FLAT

    @property
    def quantity(self) -> float:
        return self.qty

    @property
    def entry_price(self) -> Optional[float]:
        return self.price

    def to_json(self) -> str:
        data = asdict(self)
        data['action'] = self.action.value
        data['side']   = self.side.value
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'WebhookPayload':
        data = json.loads(json_str)
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        filtered_data['action'] = ActionType(filtered_data['action'])
        filtered_data['side']   = SideType(filtered_data['side'])
        return cls(**filtered_data)


class WebhookValidator:
    """하위 호환용 검증 클래스 (okx_execution.py 등에서 import)"""
    def validate_okx(self, payload: WebhookPayload):
        if not payload.symbol:
            return False, "symbol 누락"
        if payload.qty <= 0:
            return False, "qty는 0보다 커야 함"
        return True, "OK"

    def validate_upbit(self, payload: WebhookPayload):
        if payload.qty < 5000:
            return False, "최소 주문 5,000원 미만"
        return True, "OK"
        
    @staticmethod
    def validate_kr_stock(payload: WebhookPayload) -> tuple[bool, str]:
        if payload.qty <= 0:
            return False, "qty는 0보다 커야 함"
        return True, "OK"

    @staticmethod
    def validate_nasdaq(payload: WebhookPayload) -> tuple[bool, str]:
        if payload.qty <= 0:
            return False, "qty는 0보다 커야 함"
        return True, "OK"
