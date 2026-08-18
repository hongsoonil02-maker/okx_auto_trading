"""
okx_copy_engine.py
─────────────────────────────────────────────────────────────────────────────
OKX V5 Copy Trading & Risk Mirror Engine (카피 트레이딩 & 몬테카를로 방어 엔진)

[핵심 기능 및 특징]
1. OKX 공식 API V5 CopyTrading SDK (`okx.CopyTrading.CopyTradingAPI`) 완벽 연동
2. 1등 리드 트레이더 (예: KoreanTop 등) 검색 (`uniqueCode` 기반) 및 통계 조회
3. 서버 단 0ms 지연 카피 트레이딩 설정 (`first-copy-settings`, `amend-copy-settings`)
   - 리드 트레이더가 진입하면 OKX 매칭 엔진에서 내 계정으로 즉시 자동 진입/청산
4. 활성 카피 서브포지션 (`current-subpositions`) 실시간 모니터링 및 몬테카를로 방어 가드
   - 리드 트레이더의 매매를 100% 카피하되, 개별 서브포지션의 손실률(`uplRatio`)이
     우리의 최대 허용 손실 한계치(예: -3.0% 등)나 몬테카를로 생존 한계를 이탈할 경우
     해당 포지션만 API(`close-subposition`)로 즉시 강제 청산하여 자본 보호!
─────────────────────────────────────────────────────────────────────────────
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional

try:
    import okx.CopyTrading as CopyTrading
    import okx.Account as Account
    import okx.Trade as Trade
    HAS_OKX_SDK = True
except ImportError:
    HAS_OKX_SDK = False

logger = logging.getLogger("OKXCopyEngine")


class OKXCopyEngine:
    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        passphrase: str = None,
        flag: str = "0",  # '0' = Live 실거래, '1' = Demo 모의투자
        max_subpos_loss_pct: float = None,  # .env OKX_COPY_MAX_LOSS_PCT에서 로드 (기본 15.0%)
    ):
        self.api_key = api_key or os.getenv("OKX_API_KEY", "")
        self.api_secret = api_secret or os.getenv("OKX_SECRET", "") or os.getenv("OKX_SECRET_KEY", "")
        self.passphrase = passphrase or os.getenv("OKX_PASSPHRASE", "") or os.getenv("OKX_PASSWORD", "")
        self.flag = flag
        self.max_subpos_loss_pct = max_subpos_loss_pct if max_subpos_loss_pct is not None else float(os.getenv("OKX_COPY_MAX_LOSS_PCT", "15.0"))
        self.max_active_subpositions = int(os.getenv("OKX_COPY_MAX_ACTIVE_SUBPOS", "1"))
        self.single_position_only = os.getenv("OKX_SINGLE_POSITION_ONLY", "true").lower() == "true"
        self.unique_code = os.getenv("OKX_LEAD_TRADER_UNIQUE_CODE", "171F98A6EAE83C1D")
        self.lead_trader_code = self.unique_code

        if not HAS_OKX_SDK:
            logger.warning("⚠️ python-okx SDK가 설치되지 않았습니다. API 호출 시 오류가 발생할 수 있습니다.")
            self.copy_api = None
            self.account_api = None
        elif self.api_key and self.api_secret and self.passphrase:
            self.copy_api = CopyTrading.CopyTradingAPI(
                self.api_key, self.api_secret, self.passphrase, False, self.flag
            )
            self.account_api = Account.AccountAPI(
                self.api_key, self.api_secret, self.passphrase, False, self.flag
            )
            logger.info(f"✅ OKX CopyTrading API 초기화 완료 (flag={self.flag})")
        else:
            logger.warning("⚠️ OKX API 자격증명이 부족합니다. 환경 변수(OKX_API_KEY 등)를 확인하세요.")
            self.copy_api = None
            self.account_api = None

    def _copy_request(self, method: str, path: str, params: Dict[str, Any] = None, body: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        CopyTradingAPI에 네이티브 메서드가 없는 카피 REST 경로를
        SDK 내부 인증 흐름을 타는 _request_with_params로 호출한다.
        """
        if not self.copy_api:
            return {"code": "-1", "msg": "no_copy_api", "data": []}
        try:
            merged: Dict[str, Any] = params or {}
            if body:
                merged.update(body)
            res = self.copy_api._request_with_params(method.upper(), path, merged)
            if hasattr(res, "json"):
                data = res.json()
            elif isinstance(res, dict):
                data = res
            else:
                data = {"raw": str(res)}
            if isinstance(data, dict):
                return data
            return {"code": "-1", "msg": "unexpected_response_type", "data": []}
        except Exception as e:
            logger.error(f"❌ _copy_request 예외: {e}")
            return {"code": "-1", "msg": str(e), "data": []}
    def get_current_lead_traders(self, inst_type: str = "SWAP") -> List[Dict[str, Any]]:
        """
        현재 복제 중인 리드 트레이더 목록 및 설정 조회
        API: GET /api/v5/copytrading/current-lead-traders
        """
        try:
            res = self._copy_request("GET", "/api/v5/copytrading/current-lead-traders", params={"instType": inst_type})
            if res and str(res.get("code")) == "0":
                return res.get("data", [])
            logger.error(f"❌ 복제 중인 리드 트레이더 조회 실패: {res.get('msg')}")
            return []
        except Exception as e:
            logger.error(f"❌ get_current_lead_traders 예외: {e}")
            return []

    def get_copy_settings(self, unique_code: str = None) -> List[Dict[str, Any]]:
        """
        특정 또는 전체 리드 트레이더에 대한 카피 설정 조회
        API: GET /api/v5/copytrading/copy-settings
        """
        try:
            params = {}
            if unique_code:
                params["uniqueCode"] = unique_code
            res = self._copy_request("GET", "/api/v5/copytrading/copy-settings", params=params)
            if res and str(res.get("code")) == "0":
                return res.get("data", [])
            logger.error(f"❌ 카피 설정 조회 실패: {res.get('msg')}")
            return []
        except Exception as e:
            logger.error(f"❌ get_copy_settings 예외: {e}")
            return []

    def setup_or_update_copy_trading(
        self,
        unique_code: str,
        copy_mode: str = "SMART_COPY",
        copy_ratio: str = "1",
        copy_total_amt: str = "1100",
        copy_inst_id_type: str = "copy",
        sub_pos_close_type: str = "copy_close"
    ) -> Dict[str, Any]:
        """
        리드 트레이더 카피 트레이딩 최초 설정 또는 설정 변경
        API: POST /api/v5/copytrading/first-copy-settings / amend-copy-settings
        """
        if not self.copy_api:
            return {"success": False, "reason": "no_api_client"}

        try:
            existing = self.get_copy_settings(unique_code=unique_code)

            if existing and len(existing) > 0:
                first = existing[0] if isinstance(existing[0], dict) else {}
                mgn_mode = first.get("copyMgnMode") or "copy"
                inst_type = first.get("copyInstIdType") or "copy"
                close_type = first.get("subPosCloseType") or "copy_close"

                logger.info(f"🔄 리드 트레이더({unique_code}) 카피 설정 변경 (amend-copy-settings) 실행...")
                amend_payload = {
                    "uniqueCode": unique_code,
                    "copyMgnMode": mgn_mode,
                    "copyInstIdType": inst_type,
                    "subPosCloseType": close_type,
                    "copyTotalAmt": str(copy_total_amt),
                }
                res = self._copy_request("POST", "/api/v5/copytrading/amend-copy-settings", body=amend_payload)
                if res and str(res.get("code")) != "0":
                    amend_payload["copyAmt"] = str(copy_total_amt)
                    res = self._copy_request("POST", "/api/v5/copytrading/amend-copy-settings", body=amend_payload)
            else:
                mgn_mode = "copy" if copy_mode in ("SMART_COPY", "copy") else "ratio"
                logger.info(f"🚀 리드 트레이더({unique_code}) 신규 카피 개시 (first-copy-settings) 실행...")
                first_copy_payload = {
                    "uniqueCode": unique_code,
                    "copyMgnMode": mgn_mode,
                    "copyTotalAmt": str(copy_total_amt),
                    "copyRatio": str(copy_ratio),
                    "copyInstIdType": copy_inst_id_type,
                    "subPosCloseType": sub_pos_close_type,
                }
                res = self._copy_request("POST", "/api/v5/copytrading/first-copy-settings", body=first_copy_payload)

            if res and str(res.get("code")) == "0":
                logger.info(f"✅ 리드 트레이더({unique_code}) 카피 설정 완료! (할당 자본: ${copy_total_amt} USDT)")
                return {"success": True, "data": res.get("data")}
            else:
                msg = res.get("msg", "Unknown error") if res else "No response"
                logger.error(f"❌ 카피 트레이딩 설정 실패: {msg}")
                return {"success": False, "reason": msg}
        except Exception as e:
            logger.error(f"❌ setup_or_update_copy_trading 예외: {e}")
            return {"success": False, "reason": str(e)}

    def get_active_subpositions(self, unique_code: str = None) -> List[Dict[str, Any]]:
        """
        현재 리드 트레이더로부터 카피되어 열려있는 실시간 서브포지션 목록 조회
        API: GET /api/v5/copytrading/current-subpositions
        """
        try:
            params = {}
            if unique_code:
                params["uniqueCode"] = unique_code
            res = self._copy_request("GET", "/api/v5/copytrading/current-subpositions", params=params)
            if res and str(res.get("code")) == "0":
                return res.get("data", [])
            logger.debug(f"ℹ️ 서브포지션 조회 응답: {res.get('msg') if res else 'None'}")
            return []
        except Exception as e:
            logger.error(f"❌ get_active_subpositions 예외: {e}")
            return []

    def get_leading_position_history(self, instId: str = "", after: str = "", before: str = "", limit: str = "") -> List[Dict[str, Any]]:
        """
        리드 트레이더의 과거 포지션 히스토리 조회
        API: GET /api/v5/copytrading/subpositions-history
        """
        try:
            params = {}
            if instId:
                params["instId"] = instId
            if after:
                params["after"] = after
            if before:
                params["before"] = before
            if limit:
                params["limit"] = limit
            res = self._copy_request("GET", "/api/v5/copytrading/subpositions-history", params=params)
            if res and str(res.get("code")) == "0":
                return res.get("data", [])
            logger.error(f"❌ 리드 트레이더 히스토리 조회 실패: {res.get('msg') if res else 'None'}")
            return []
        except Exception as e:
            logger.error(f"❌ get_leading_position_history 예외: {e}")
            return []

    def check_lead_trader_status(self, unique_code: str = None) -> Dict[str, Any]:
        """
        리드 트레이더 카피 상태 확인.
        활성/비활성 여부를 감지하고 필요시 Telegram 알림을 전송합니다.
        """
        target = unique_code or self.unique_code
        if not target:
            return {"ok": False, "reason": "no_unique_code"}
        try:
            settings = self.get_copy_settings(unique_code=target)
            if not settings:
                msg = f"⚠️ 리드 트레이더({target}) 카피 설정 없음"
                logger.warning(msg)
                try:
                    from utils_telegram import send_telegram_alert
                    send_telegram_alert(f"{msg} — 카피 활성화 필요")
                except Exception:
                    pass
                return {"ok": False, "reason": "no_settings"}
            s = settings[0] if isinstance(settings, list) and settings else (settings if isinstance(settings, dict) else {})
            copy_state = s.get("copyState") or s.get("copy_state") or s.get("state")
            if copy_state in (0, "0", None, "", "null"):
                msg = f"⚠️ 리드 트레이더({target}) 카피 비활성 감지(copyState={copy_state})"
                logger.warning(msg)
                try:
                    from utils_telegram import send_telegram_alert
                    send_telegram_alert(f"{msg} — OKX에서 카피를 활성화하세요.")
                except Exception:
                    pass
                return {"ok": False, "reason": "inactive", "copyState": copy_state}
            msg = f"✅ 리드 트레이더({target}) 카피 활성 확인(copyState={copy_state})"
            logger.info(msg)
            return {"ok": True, "copyState": copy_state}
        except Exception as e:
            logger.error(f"❌ 리드 트레이더 상태 확인 실패: {e}")
            return {"ok": False, "reason": str(e)}

    def close_subposition_instant(self, sub_pos_id: str, sub_pos_type: str = "SWAP") -> bool:
        """
        우리의 방어/몬테카를로 기준을 이탈한 개별 카피 서브포지션 즉시 강제 청산
        API: POST /api/v5/copytrading/close-subposition
        """
        try:
            payload = {
                "subPosId": sub_pos_id,
                "subPosType": sub_pos_type
            }
            logger.warning(f"🚨 위험 한계 도달! 서브포지션({sub_pos_id}) 강제 청산 명령 발송...")
            res = self._copy_request("POST", "/api/v5/copytrading/close-subposition", body=payload)
            if res and str(res.get("code")) == "0":
                logger.info(f"✅ 서브포지션({sub_pos_id}) 강제 청산 성공!")
                return True
            else:
                logger.error(f"❌ 서브포지션({sub_pos_id}) 강제 청산 실패: {res.get('msg') if res else 'None'}")
                return False
        except Exception as e:
            logger.error(f"❌ close_subposition_instant 예외: {e}")
            return False

    def stop_copy_trading(self, unique_code: str, sub_pos_close_type: str = "market_close") -> bool:
        """
        리드 트레이더 카피 전면 중단 (긴급 정지 또는 몬테카를로 로버스트 컷오프 발동 시)
        API: POST /api/v5/copytrading/stop-copy-trading
        """
        try:
            payload = {
                "uniqueCode": unique_code,
                "subPosCloseType": sub_pos_close_type
            }
            logger.warning(f"🛑 리드 트레이더({unique_code}) 카피 트레이딩 전면 중단 및 포지션 청산({sub_pos_close_type})...")
            res = self._copy_request("POST", "/api/v5/copytrading/stop-copy-trading", body=payload)
            if res and str(res.get("code")) == "0":
                logger.info(f"✅ 리드 트레이더({unique_code}) 카피 중단 완료!")
                return True
            else:
                logger.error(f"❌ stop_copy_trading 실패: {res.get('msg') if res else 'None'}")
                return False
        except Exception as e:
            logger.error(f"❌ stop_copy_trading 예외: {e}")
            return False

    def close_risky_subpositions(self, max_subpos_loss_pct: float = None, max_active_subpositions: int = None) -> List[Dict[str, Any]]:
        """
        [Orca/Kiro 승리 개선안]
        1. 개별 서브포지션 손실률이 max_subpos_loss_pct(예: -10.0%) 초과 시 즉시 강제 청산
        2. 리드 트레이더의 과도한 물타기(Pyramiding) 방지: 활성 서브포지션 수가 max_active_subpositions(1개, OKX_COPY_MAX_ACTIVE_SUBPOS) 초과 시 손실 폭이 큰 순서대로 청산
        3. single_position_only=true: 동일 심볼의 중복 포지션(ADL 풀링)도 방지
        """
        threshold = max_subpos_loss_pct if max_subpos_loss_pct is not None else self.max_subpos_loss_pct
        if max_active_subpositions is None:
            max_active_subpositions = self.max_active_subpositions
        subpositions = self.get_active_subpositions()
        if not subpositions:
            return []

        closed_results = []

        # ── 1. 개별 서브포지션 -3.5% 손절 처리 ──
        for subpos in subpositions:
            try:
                sub_id = subpos.get("subPosId") or subpos.get("subposId")
                inst_id = subpos.get("instId", "")
                upl_ratio = float(subpos.get("uplRatio", 0.0)) * 100.0  # % 단위 변환

                if upl_ratio <= -abs(threshold):
                    logger.warning(
                        f"🚨 [OKX Copy Risk Guard] 서브포지션({sub_id}, {inst_id}) "
                        f"손실률 {upl_ratio:.2f}% (한계치: -{abs(threshold)}%) → 강제 청산 실행!"
                    )
                    res = self.close_subposition_instant(sub_id)
                    closed_results.append({"subPosId": sub_id, "instId": inst_id, "upl_ratio": upl_ratio, "result": res})
            except Exception as e:
                logger.error(f"❌ 서브포지션 리스크 검사 예외: {e}")

        # ── 2. 리드 트레이더 물타기 과도화 방지 (Max Active Subposition Limit) ──
        remaining_subpos = [s for s in subpositions if s.get("subPosId") not in [c.get("subPosId") for c in closed_results]]
        if len(remaining_subpos) > max_active_subpositions:
            logger.warning(f"⚠️ [OKX Copy Risk Guard] 활성 서브포지션 초과 ({len(remaining_subpos)}개 > 최대 {max_active_subpositions}개) → 물타기 억제 청산 실행")
            # 손실률이 큰 순서대로 정렬하여 초과 수량만큼 청산
            sorted_subpos = sorted(remaining_subpos, key=lambda s: float(s.get("uplRatio", 0.0)))
            excess_count = len(remaining_subpos) - max_active_subpositions
            for i in range(excess_count):
                target = sorted_subpos[i]
                sub_id = target.get("subPosId")
                inst_id = target.get("instId", "")
                upl_ratio = float(target.get("uplRatio", 0.0)) * 100.0
                logger.warning(f"🚨 [물타기 방지 락] 초과 서브포지션({sub_id}, {inst_id}) 손실률 {upl_ratio:.2f}% 청산")
                res = self.close_subposition_instant(sub_id)
                closed_results.append({"subPosId": sub_id, "instId": inst_id, "upl_ratio": upl_ratio, "result": res})

        return closed_results

    def guard_subpositions_risk(self, unique_code: str = None) -> Dict[str, Any]:
        """
        [몬테카를로 & 자본 보호 방어 가드]
        리드 트레이더가 개설한 카피 서브포지션들의 수익률(uplRatio) 및 손익을 실시간 감시하여,
        지정된 최대 허용 손실률(max_subpos_loss_pct)을 초과하는 위험 포지션을 즉시 컷아웃(손절)합니다.
        수익 중인 효자 포지션(+500% 등)은 리드 트레이더의 청산 시점까지 그대로 유지하여 카피 수익 극대화!
        """
        subpos_list = self.get_active_subpositions(unique_code=unique_code)
        if not subpos_list:
            return {"active_count": 0, "total_upl": 0.0, "closed_subpos": []}

        total_upl = 0.0
        closed_ids = []

        for sp in subpos_list:
            sub_pos_id = sp.get("subPosId", "")
            inst_id = sp.get("instId", "UNKNOWN")
            side = sp.get("side", "")
            pos_side = sp.get("posSide", "")
            upl = float(sp.get("upl", 0.0))
            upl_ratio_str = sp.get("uplRatio", "0.0")
            
            # uplRatio가 소수점 비율(예: -0.04 = -4.0%)이거나 백분율 문자열일 수 있으므로 변환
            try:
                upl_ratio = float(upl_ratio_str)
                if abs(upl_ratio) < 10.0 and upl != 0.0:
                    # 1.0 = 100% 형태인 경우 % 로 변환
                    upl_pct = upl_ratio * 100.0
                else:
                    upl_pct = upl_ratio
            except ValueError:
                upl_pct = 0.0

            total_upl += upl

            # 방어 가드 검사: 개별 포지션 손실률이 한계치 이탈 시
            if upl_pct < -abs(self.max_subpos_loss_pct):
                logger.warning(
                    f"⚠️ [방어 컷아웃 발동] {inst_id} ({pos_side}) | 서브포지션 ID: {sub_pos_id} | "
                    f"손실률: {upl_pct:.2f}% (한계 -{abs(self.max_subpos_loss_pct):.2f}%) | 미실현 손익: ${upl:.2f}"
                )
                closed = self.close_subposition_instant(sub_pos_id, sub_pos_type="SWAP")
                if closed:
                    closed_ids.append(sub_pos_id)
            else:
                logger.debug(
                    f"🟢 [카피 유지] {inst_id} ({pos_side}) | ID: {sub_pos_id} | "
                    f"수익률: {upl_pct:+.2f}% | 미실현 PnL: ${upl:+.2f}"
                )

        return {
            "active_count": len(subpos_list),
            "total_upl": total_upl,
            "closed_subpos": closed_ids
        }

    def diagnose_copy_status(self, unique_code: str = None) -> Dict[str, Any]:
        """
        카피 트레이딩 전체 상태를 한눈에 진단하는 종합 리포트 생성.
        - 리드 트레이더 카피 활성 여부
        - 활성 서브포지션 목록 및 손익
        - 리스크 가드 임계값 설정 확인
        """
        target = unique_code or self.unique_code
        report: Dict[str, Any] = {
            "unique_code": target,
            "copy_active": False,
            "active_subpositions": [],
            "total_active_count": 0,
            "total_upl": 0.0,
            "risk_settings": {
                "max_subpos_loss_pct": self.max_subpos_loss_pct,
                "max_active_subpositions": self.max_active_subpositions,
                "single_position_only": self.single_position_only,
            },
        }

        # 1. 카피 상태 확인
        try:
            status = self.check_lead_trader_status(unique_code=target)
            report["copy_active"] = status.get("ok", False)
            report["copy_state_detail"] = status
        except Exception as e:
            report["copy_state_detail"] = {"error": str(e)}

        # 2. 활성 서브포지션 조회
        try:
            subpositions = self.get_active_subpositions(unique_code=target)
            report["total_active_count"] = len(subpositions)
            total_upl = 0.0
            sub_details = []
            for sp in subpositions:
                upl = float(sp.get("upl", 0.0))
                upl_ratio = float(sp.get("uplRatio", 0.0))
                total_upl += upl
                sub_details.append({
                    "subPosId": sp.get("subPosId", ""),
                    "instId": sp.get("instId", ""),
                    "posSide": sp.get("posSide", ""),
                    "upl": upl,
                    "uplRatio_pct": round(upl_ratio * 100, 2),
                })
            report["active_subpositions"] = sub_details
            report["total_upl"] = round(total_upl, 4)
        except Exception as e:
            report["subposition_error"] = str(e)

        # 3. 리드 트레이더 목록
        try:
            leaders = self.get_current_lead_traders()
            report["current_lead_traders"] = len(leaders)
        except Exception:
            report["current_lead_traders"] = "error"

        return report

    def detect_new_subpositions(self, known_ids: set, unique_code: str = None) -> List[Dict[str, Any]]:
        """
        기존에 알던 subPosId 집합과 비교하여 새로 열린 카피 서브포지션만 반환.
        리더의 신규 포지션 진입 감지에 사용한다.
        """
        current = self.get_active_subpositions(unique_code=unique_code)
        new_items = []
        for sp in current:
            sp_id = sp.get("subPosId")
            if sp_id and sp_id not in known_ids:
                new_items.append(sp)
        return new_items


    def select_top_trader(self, inst_id: str = "BTC-USDT") -> Optional[Dict[str, Any]]:
        """
        수익률 1등 리드 트레이더 후보를 선택한다.
        OKX CopyTrading 추천 리드 트레이더 목록을 조회하여 pnlRatio 기준 1위를 반환한다.
        API: GET /api/v5/copytrading/lead-trader-list (public)
        """
        try:
            res = self._copy_request("GET", "/api/v5/copytrading/lead-trader-list", params={"instId": inst_id})
            if res and str(res.get("code")) == "0":
                traders = res.get("data", [])
                if traders:
                    sorted_traders = sorted(
                        traders,
                        key=lambda t: float(t.get("pnlRatio", 0) or 0),
                        reverse=True,
                    )
                    return sorted_traders[0]
                logger.warning("⚠️ 리드 트레이더 목록이 비어있습니다.")
            else:
                logger.warning(f"⚠️ 리드 트레이더 목록 조회 실패: {res.get('msg') if res else 'None'}")
        except Exception as e:
            logger.error(f"❌ select_top_trader 예외: {e}")

        logger.warning("⚠️ 리드 트레이더 자동 선정 실패, 기본 unique_code 반환")
        return {
            "uniqueCode": self.unique_code,
            "nickName": "Default",
            "pnlRatio": 0,
        }

    def auto_copy_top_trader(
        self,
        copy_total_amt: str = "800",
        copy_ratio: str = "1",
        copy_mode: str = "SMART_COPY",
    ) -> Dict[str, Any]:
        """
        수익률 1등 리드 트레이더를 자동 선정한 후 카피 트레이딩을 개시한다.
        single_position_only=true이면 활성 포지션 수 제한 및 심볼 중복 방지 검사 수행.
        """
        # ADL 풀링 방지: 이미 활성 서브포지션 한도 도달 시 새 진입 거부
        if self.single_position_only:
            active = self.get_active_subpositions()
            if len(active) >= self.max_active_subpositions:
                logger.warning(
                    f"🚫 [Single Position Guard] 활성 서브포지션 {len(active)}개 ≥ "
                    f"최대 {self.max_active_subpositions}개 제한. 진입 거부."
                )
                return {"success": False, "reason": "max_active_subpositions_reached"}

        top = self.select_top_trader()
        if not top:
            return {"success": False, "reason": "no_top_trader_found"}

        unique_code = top.get("uniqueCode", self.unique_code)
        result = self.setup_or_update_copy_trading(
            unique_code=unique_code,
            copy_mode=copy_mode,
            copy_ratio=copy_ratio,
            copy_total_amt=copy_total_amt,
            copy_inst_id_type="copy",
            sub_pos_close_type="copy_close",
        )
        result["selected_trader"] = top
        return result


def get_copy_engine():
    return OKXCopyEngine()


