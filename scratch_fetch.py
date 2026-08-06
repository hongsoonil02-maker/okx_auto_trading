import sys
import json
sys.path.append('/home/hongsoonil02/quant_system')
from okx_copy_engine import get_copy_engine
engine = get_copy_engine()
res = engine._copy_request("GET", "/api/v5/copytrading/lead-traders", params={"instType": "SWAP"})
if not res.get("data"):
    res = engine._copy_request("GET", "/api/v5/copytrading/lead-trader-list", params={"instType": "SWAP"})
print(json.dumps(res, indent=2, ensure_ascii=False))
