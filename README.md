# 🚀 OKX Auto Trading System (Quant System)

이 리포지토리는 OKX 거래소를 기반으로 동작하는 고도화된 자동매매 퀀트 시스템입니다. 메이저 코인, 벤처/알트 코인, 그리고 타겟 트레이더의 포지션을 카피하는 섀도우 미러링 봇까지 다양한 전략을 수행할 수 있도록 모듈화되어 있습니다. 팀원 누구나 이 가이드를 통해 서버를 세팅하고 봇을 독립적으로 구동할 수 있습니다.

---

## 🏗️ 1. 시스템 아키텍처 및 봇 라인업

현재 시스템은 여러 개의 독립된 봇으로 구성되어 있으며, 공통 라이브러리(`strategy_common.py`)를 기반으로 확장 가능하게 설계되었습니다.

*   **`okx_major_strategy.py` (메이저 전략 봇):** 비트코인(BTC) 및 이더리움(ETH) 등 대형 코인을 대상으로 안정적인 추세 추종 및 모멘텀 전략을 수행합니다.
*   **`okx_venture_strategy.py` (벤처 전략 봇):** 변동성이 큰 알트코인을 대상으로 거래량 급증(RVOL) 등을 포착해 단기 수익을 노리는 공격적 봇입니다.
*   **`okx_stock_venture_strategy.py` (스탁-벤처 하이브리드):** 주식형 차트 패턴이나 수급 데이터를 코인에 접목시켜 특정 시간대나 조건에서 매매하는 하이브리드 전략입니다.
*   **`bot_e_shadow_mirror.py` (섀도우 미러 봇):** 세계 상위권 카피 트레이더나 타 주력 봇의 매매를 은밀하게(Shadow) 거울처럼 따라가며 자체 리스크 관리를 적용하는 봇입니다.
*   **`strategy_common.py`:** 위 봇들이 공통적으로 사용하는 핵심 알고리즘, 지표 계산, API 연동 등을 담당하는 라이브러리입니다.

---

## ☁️ 2. 자동매매용 구글 클라우드(GCP) 서버 구축 가이드

24시간 무중단으로 자동매매를 돌리기 위해서는 클라우드 서버(가상 머신)가 필요합니다.

1.  **GCP 가입 및 프로젝트 생성:** [Google Cloud Platform](https://cloud.google.com/)에 가입 후 새 프로젝트를 생성합니다. (초기 가입 시 무료 크레딧 활용 가능)
2.  **VM 인스턴스 생성:**
    *   메뉴에서 **Compute Engine > VM 인스턴스**로 이동하여 `인스턴스 만들기`를 클릭합니다.
    *   **이름:** `quant-trading-server` (자유롭게 지정)
    *   **리전:** 아시아(예: `asia-northeast3` 서울 또는 `asia-northeast1` 도쿄)를 선택해 거래소와의 네트워크 지연(Ping)을 최소화합니다.
    *   **머신 구성:** 최소 `e2-micro` 이상 (여유로운 구동을 위해 `e2-medium` 추천).
    *   **부팅 디스크:** `Ubuntu 22.04 LTS` (또는 익숙한 리눅스 배포판). 디스크 크기는 `20GB` 이상 권장.
3.  **방화벽 설정:** 필요시 외부 접근을 위한 포트를 개방합니다. (보안을 위해 최소한의 포트만 엽니다)
4.  **만들기 버튼**을 눌러 서버 생성을 완료합니다.

---

## 💻 3. 콘솔(Console) 접속 및 기본 환경 세팅

생성된 서버에 접속하여 봇 구동 환경을 만듭니다.

1.  **SSH 접속:** GCP 대시보드의 VM 인스턴스 목록에서 `SSH` 버튼을 눌러 브라우저 콘솔을 띄우거나, 터미널 프로그램(MobaXterm, PuTTY, Mac Terminal 등)으로 외부 IP로 접속합니다.
2.  **시스템 업데이트 및 필수 패키지 설치:**
    ```bash
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y python3 python3-pip python3-venv git
    ```
3.  **리포지토리 클론 (소스코드 다운로드):**
    ```bash
    git clone https://github.com/hongsoonil02-maker/okx_auto_trading.git quant_system
    cd quant_system
    ```
4.  **파이썬 환경 설정 (가상환경 권장):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
5.  **환경 변수 설정:** 
    `.env` 파일을 생성하고 OKX API Key, Secret, Passphrase 등을 기입합니다. (보안상 절대 Github에 올리지 마세요!)

---

## 🤖 4. 안티그래비티(Antigravity) 설치 및 AI 봇 가동 방법

팀 내에서 봇 코드를 수정하거나, AI 에이전트를 이용해 퀀트 개발을 자동화하려면 안티그래비티(Antigravity) IDE를 세팅해야 합니다.

1.  **Antigravity 설치:**
    *   구글 서버 콘솔에 접속한 후, 안티그래비티 설치 스크립트나 바이너리를 다운로드하여 서버에 세팅합니다. (팀에서 제공받은 별도의 설치 명령어(curl 등)를 실행하세요)
    *   설치 완료 후 `antigravity` 명령어를 통해 서버 인스턴스에 백그라운드 서버를 가동시킵니다.
2.  **Antigravity로 봇 제어하기:**
    *   Antigravity 터미널 혹은 웹 UI에서 `quant_system` 워크스페이스를 엽니다.
    *   채팅 인터페이스를 통해 AI에게 "현재 백테스트 결과를 리포팅해줘" 혹은 "새로운 지표를 전략에 추가해줘"라고 명령하면, AI가 직접 코드를 수정하고 봇을 제어할 수 있습니다.

---

## ⚙️ 5. 봇 적재 및 가동 방법

봇을 수동으로, 혹은 시스템 데몬(무중단 서비스)으로 가동하는 두 가지 방법이 있습니다.

### 방법 A. 쉘 스크립트를 통한 간편 가동 (테스트용)
`start_coinbot.sh`를 활용해 프로세스를 백그라운드로 실행합니다.
```bash
chmod +x start_coinbot.sh
./start_coinbot.sh
```

### 방법 B. Systemd를 통한 안정적 무중단 가동 (운영 서버용 권장)
서버 재부팅 시에도 봇이 자동으로 시작되고, 에러 발생 시 자동 재시작되도록 시스템 데몬에 등록합니다. `systemd/` 디렉토리에 각 봇의 설정 파일이 들어있습니다.

1. **서비스 파일 복사:**
   ```bash
   sudo cp systemd/*.service /etc/systemd/system/
   ```
2. **데몬 리로드 및 서비스 시작:**
   ```bash
   sudo systemctl daemon-reload
   
   # 메이저 봇 시작 및 부팅 시 자동실행 등록
   sudo systemctl enable --now okx_major_strategy.service
   
   # 벤처 봇 시작
   sudo systemctl enable --now okx_venture_strategy.service
   
   # 섀도우 미러 봇 시작
   sudo systemctl enable --now bot_e_shadow_mirror.service
   ```
3. **가동 상태 확인 및 로그 보기:**
   ```bash
   sudo systemctl status okx_venture_strategy.service
   
   # 실시간 로그 확인 (ctrl+c 로 종료)
   journalctl -fu okx_venture_strategy.service
   ```

---

## 📈 6. 백테스트 및 유지보수

새로운 아이디어나 파라미터를 실전에 투입하기 전, 반드시 커밋되어 있는 백테스트 스크립트들을 사용해 검증하세요.
```bash
# 전체 전략 백테스트
python3 final_backtest_all.py

# 어제장 기준 비교 백테스트
python3 run_comparison_yesterday.py
```

기타 봇의 세부적인 수정 로직이나 업데이트 기록은 `_archive/obsidian_quant_dev_log.md`를 참고해 주세요. 화이팅! 🚀
