# -*- coding: utf-8 -*-
"""
apply_improvements.py — D2 개선안 라이브 코드 반영 (원본 자동 백업 포함)
적용 내용:
  1) .env: 메이저 레버리지 15 -> 10 / 소프트스탑 비활성화(OKX_SOFT_STOP_LOSS=-1)
  2) strategy_common.py: 소프트스탑 env화 + BTC 1h ADX<15 신규진입 하드게이트
  3) okx_venture_strategy.py: 출혈 심볼 4종(LIT/CHIP/BOME/FIL) 블랙리스트 추가
재시작 전까지 실행 중인 봇에는 영향 없음.
"""
import shutil
import time

TS = time.strftime("%Y%m%d_%H%M%S")


def patch(path, old, new, label):
    src = open(path, encoding="utf-8").read()
    if new in src:
        print(f"[SKIP] {label}: 이미 적용됨")
        return
    if old not in src:
        raise SystemExit(f"[FAIL] {label}: 대상 문자열 없음\n---\n{old}")
    shutil.copy(path, path + ".bak_" + TS)
    open(path, "w", encoding="utf-8").write(src.replace(old, new, 1))
    print(f"[OK] {label}")


# 1) .env
env_path = ".env"
src = open(env_path, encoding="utf-8").read()
bak = f".env.bak_{TS}"
shutil.copy(env_path, bak)
if "OKX_MAJOR_LEVERAGE=10" not in src:
    assert "OKX_MAJOR_LEVERAGE=15" in src, ".env: OKX_MAJOR_LEVERAGE=15 없음"
    src = src.replace("OKX_MAJOR_LEVERAGE=15", "OKX_MAJOR_LEVERAGE=10", 1)
    print("[OK] .env: 메이저 레버리지 15 -> 10")
else:
    print("[SKIP] .env: 레버리지 이미 10")
if "OKX_SOFT_STOP_LOSS" not in src:
    src = src.rstrip("\n") + "\nOKX_SOFT_STOP_LOSS=-1\n"
    print("[OK] .env: 소프트스탑 비활성화 추가 (-1)")
open(env_path, "w", encoding="utf-8").write(src)

# 2) strategy_common.py
patch(
    "strategy_common.py",
    '    SOFT_STOP_LOSS_PCT = -0.12  # 조기 손절(Soft Stop): 가랑비 출혈 방지용',
    '    SOFT_STOP_LOSS_PCT = float(os.getenv("OKX_SOFT_STOP_LOSS", "-0.12"))  # 조기 손절(가랑비 방어). -1 = 비활성 [8/26 백테스트]',
    "소프트스탑 env화",
)
patch(
    "strategy_common.py",
    "            self._chop_block = False  # 하드 차단 폐지 — 항상 거래",
    "            # [8/26 백테스트 복원] 극저변동(BTC 1h ADX<15) 신규 진입만 차단, 청산은 계속.\n"
    "            # 90일 분할검증: 전반기 손실 축소/후반기 수익 확대/MDD 19->15%. 청산·트레일링은 항상 동작.",
    "저ADX 진입 게이트",
)

# 3) okx_venture_strategy.py
patch(
    "okx_venture_strategy.py",
    "    BLACKLIST = ['KR200', 'SKHYNIX', 'MU', 'SHAZ', 'ISRG', 'ROBO', 'RAM', 'DRAM', 'GME',\n"
    "                 'GPS', 'SNXX', 'POL']  # GPS(-64.7%), SNXX(-38.3%), POL(-21.6%) 반복 손실 차단",
    "    BLACKLIST = ['KR200', 'SKHYNIX', 'MU', 'SHAZ', 'ISRG', 'ROBO', 'RAM', 'DRAM', 'GME',\n"
    "                 'GPS', 'SNXX', 'POL',  # GPS(-64.7%), SNXX(-38.3%), POL(-21.6%) 반복 손실 차단\n"
    "                 'LIT', 'CHIP', 'BOME', 'FIL']  # [8/26 백테스트] 90일 누적 출혈 상위 알트 차단",
    "벤처 블랙리스트 4종 추가",
)

print("완료 — 재시작 전까지 라이브 영향 없음")
