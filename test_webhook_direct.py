#!/usr/bin/env python3
"""Direct test of webhook signature verification"""
import asyncio
import aiohttp
import json
import hmac
import hashlib
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/home/hongsoonil02/quant_system/.env')

WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET')
print(f"WEBHOOK_SECRET length: {len(WEBHOOK_SECRET) if WEBHOOK_SECRET else 0}")

# Create payload
payload = {
    "action": "open",
    "side": "buy",
    "symbol": "BTC-USDT",
    "qty": 0.001,
    "leverage": 10
}

# Serialize exactly as Bot E would
body = json.dumps(payload, separators=(',', ':'), sort_keys=True)
print(f"Body sent: {body}")

# Sign exactly as webhook_spec.sign_payload does
canonical = json.dumps(json.loads(body), separators=(',', ':'), sort_keys=True)
print(f"Canonical: {canonical}")
signature = hmac.new(
    WEBHOOK_SECRET.encode(), 
    canonical.encode(), 
    hashlib.sha256
).hexdigest()
print(f"Signature: {signature}")

# Now send request
async def send_test():
    async with aiohttp.ClientSession() as session:
        headers = {
            "Content-Type": "application/json",
            "X-Signature": signature
        }
        async with session.post("http://localhost:8009/webhook", data=body, headers=headers) as resp:
            text = await resp.text()
            print(f"Response status: {resp.status}")
            print(f"Response: {text}")

asyncio.run(send_test())