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
import os
import hmac
import hashlib

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
WEBHOOK_SIGNATURE_HEADER = "X-Signature"


def get_webhook_secret() -> str:
    """Dynamically read WEBHOOK_SECRET from environment at call time
    (avoids import-order issues when load_dotenv runs after this module loads)."""
    return os.getenv("WEBHOOK_SECRET", "")


def sign_payload(payload_str: str) -> str:
    """HMAC-SHA256 서명 생성 (발신자용)."""
    secret = get_webhook_secret()
    if not secret:
        return ""
    # Ensure consistent ordering, sort keys and use canonical JSON format
    import json
    try:
        parsed = json.loads(payload_str)
        # Canonical representation: sort keys and use compact separators
        canonical = json.dumps(parsed, separators=(',', ':'), sort_keys=True)
        return hmac.new(
            secret.encode(), canonical.encode(), hashlib.sha256
        ).hexdigest()
    except Exception:
        return hmac.new(
            secret.encode(), payload_str.encode(), hashlib.sha256
        ).hexdigest()


def verify_webhook_signature(payload_str: str, signature: str) -> bool:
    """HMAC-SHA256 서명 검증 (수신자용).
    WEBHOOK_SECRET이 비어있으면 검증을 건너뜀 (개발/테스트용, 프로덕션에서는 반드시 설정)."""
    secret = get_webhook_secret()
    if not secret:
        return True
    if not signature:
        return False
    expected = sign_payload(payload_str)
    return hmac.compare_digest(expected, signature)

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
    leverage:        Optional[int]   = None   # 개별 전략용 레버리지 오버라이드
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
        if "action" not in filtered_data:
            raise ValueError("WebhookPayload: 'action' 필드 누락")
        if "side" not in filtered_data:
            raise ValueError("WebhookPayload: 'side' 필드 누락")
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
