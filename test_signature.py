#!/usr/bin/env python3
import hmac
import hashlib
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/home/hongsoonil02/quant_system/.env')

WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET')

def sign_payload(payload: dict) -> str:
    """Generate HMAC-SHA256 signature for payload"""
    body = json.dumps(payload, separators=(',', ':'), sort_keys=True)
    signature = hmac.new(
        WEBHOOK_SECRET.encode('utf-8'),
        body.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature

# Test payload
test_payload = {
    "action": "open",
    "side": "buy", 
    "symbol": "BTC-USDT",
    "qty": 0.001,
    "leverage": 10
}

signature = sign_payload(test_payload)
print(f"Payload: {json.dumps(test_payload)}")
print(f"Signature: {signature}")
print(f"WEBHOOK_SECRET: {WEBHOOK_SECRET}")