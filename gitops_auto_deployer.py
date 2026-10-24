# -*- coding: utf-8 -*-
"""
gitops_auto_deployer.py — Automated GitOps Push-to-Deploy Pipeline
Checks origin/main on GitHub every 5 minutes.
If new commits are detected:
1. Pulls latest changes
2. Preflight compiles Python files
3. Restarts systemd trading services
4. Sends a Telegram deployment notification
"""

import os
import sys
import subprocess
import logging
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

from bot_utils import send_telegram

LOG_FILE = os.path.join(CURRENT_DIR, "gitops_auto_deployer.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8")
    ]
)
logger = logging.getLogger("GitOpsDeployer")

def run_cmd(cmd: str) -> tuple[int, str]:
    res = subprocess.run(cmd, shell=True, cwd=CURRENT_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = (res.stdout + "\n" + res.stderr).strip()
    return res.returncode, out

def check_and_deploy():
    # 1. Fetch remote origin
    code, out = run_cmd("git fetch origin main")
    if code != 0:
        logger.warning(f"Git fetch failed: {out}")
        return

    # 2. Check revs
    _, local_rev = run_cmd("git rev-parse HEAD")
    _, remote_rev = run_cmd("git rev-parse origin/main")

    local_rev = local_rev.strip()
    remote_rev = remote_rev.strip()

    if not local_rev or not remote_rev or local_rev == remote_rev:
        logger.debug("Local repo up to date with origin/main. No deployment needed.")
        return

    logger.info(f"🔄 New commits detected! Local: {local_rev[:7]} -> Remote: {remote_rev[:7]}")

    # 3. Pull latest changes
    code, pull_out = run_cmd("git pull origin main")
    if code != 0:
        logger.error(f"Git pull failed: {pull_out}")
        send_telegram(f"⚠️ [GitOps 배포 오류] git pull 실패: {pull_out[:200]}")
        return

    # 4. Preflight Python compile
    py_files = ["alpaca_golden_trader.py", "okx_jev_demo_trader.py", "daily_quant_auditor.py"]
    for pf in py_files:
        full_p = os.path.join(CURRENT_DIR, pf)
        if os.path.exists(full_p):
            code, comp_out = run_cmd(f"python3 -m py_compile {pf}")
            if code != 0:
                logger.error(f"Compile verification failed on {pf}: {comp_out}")
                # Rollback
                run_cmd(f"git reset --hard {local_rev}")
                send_telegram(f"🚨 [GitOps 배포 롤백] {pf} 문법 검증 실패! 이전 버전({local_rev[:7]})으로 자동 롤백되었습니다.")
                return

    # 5. Restart trading services gracefully
    code, restart_out = run_cmd("systemctl --user restart alpaca_golden_trader okx_jev_demo_trader")
    if code == 0:
        msg = (
            f"🚀 [100-Bagger GitOps 무인 배포 완료]\n"
            f"• 신규 커밋: {remote_rev[:7]}\n"
            f"• 배포 시각: {datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S')} KST\n"
            f"• 대상 서비스: alpaca_golden_trader, okx_jev_demo_trader\n"
            f"• 문법 사전 검증: 100% PASS ✅"
        )
        logger.info(msg)
        send_telegram(msg)
    else:
        logger.error(f"Service restart failed: {restart_out}")
        send_telegram(f"⚠️ [GitOps 재시작 오류] systemctl restart 실패: {restart_out[:200]}")

if __name__ == "__main__":
    check_and_deploy()
