import asyncio
import json
import os
import sys

from tournament_common import (
    run_tournament,
    simulate,
    simulate_vibe,
    save_results_history,
    promote_strategy,
    load_promoted_config,
    PROMOTED_CONFIG_FILE,
    RESULTS_HISTORY_FILE,
)

SYMBOL_SETS = {
    "Major_Crypto": ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT'],
    "Venture_Alts": ['DOGE/USDT:USDT', 'PEPE/USDT:USDT', 'WIF/USDT:USDT'],
}

STRATEGIES = {
    "Supertrend_200": {
        "func": simulate,
        "params": {"ema_period": 200, "tight_mult": 2.5, "loose_mult": 4.0, "vol_mult": 1.2, "min_hold": 3, "max_dca": 8, "tp_thr": 1.02, "scale_out": True},
    },
    "Supertrend_50": {
        "func": simulate,
        "params": {"ema_period": 50, "tight_mult": 2.5, "loose_mult": 4.0, "vol_mult": 1.2, "min_hold": 3, "max_dca": 8, "tp_thr": 1.02, "scale_out": True},
    },
    "Geumgang_BB_RSI": {
        "func": simulate_vibe,
        "params": {},
    },
}


async def run_single_tournament(symbol_set_name, symbol_list, timeframe="15m"):
    print(f"\n{'='*60}")
    print(f" 📊 [{symbol_set_name}] 백테스트 ({timeframe})")
    print(f"{'='*60}")
    results = {}

    for agent_name, config in STRATEGIES.items():
        print(f"  🔬 [{agent_name}] 실행 중...", end=" ", flush=True)
        result = run_tournament(
            agent_name=f"{symbol_set_name}/{agent_name}",
            symbols=symbol_list,
            strategy_func=config["func"],
            strategy_params=config["params"],
            timeframe=timeframe,
        )
        if result:
            results[agent_name] = result
            print(f"✅ 수익률 {result['overall_ret']:.2f}% (승률 {result['overall_win_rate']:.1f}%, {result['total_trades']}회)")
        else:
            print("❌ 데이터 없음")

    return results


async def run_all_tournaments(timeframe="15m"):
    print("🚀 ORCA 토너먼트 엔진 - 전체 백테스트 시작")
    print(f"⏰ 시간프레임: {timeframe}")
    print(f"📅 시작 시간: {__import__('pandas').Timestamp.now().isoformat()}")

    all_results = {}

    for set_name, symbols in SYMBOL_SETS.items():
        results = await run_single_tournament(set_name, symbols, timeframe)
        all_results[set_name] = results

    save_results_history(all_results)

    print(f"\n{'='*60}")
    print(" 🏆 토너먼트 최종 결과")
    print(f"{'='*60}")

    best_overall = None
    best_agent = None
    best_set = None

    for set_name, results in all_results.items():
        print(f"\n[{set_name}]")
        for agent_name, result in sorted(results.items(), key=lambda x: x[1]['overall_ret'], reverse=True):
            print(f"  {agent_name:<25} | 수익률: {result['overall_ret']:>7.2f}% | 승률: {result['overall_win_rate']:>5.1f}% | 거래: {result['total_trades']:>3}회")
            if best_overall is None or result['overall_ret'] > best_overall:
                best_overall = result['overall_ret']
                best_agent = agent_name
                best_set = set_name
                best_result = result
                best_params = STRATEGIES[agent_name.split('/')[-1]]['params']

    if best_result:
        print(f"\n🏆 전체 최우승: {best_agent} (수익률: {best_overall:.2f}%)")
        promoted = promote_strategy(best_result, best_params)

        print(f"\n📌 프로모션된 전략이 promoted_strategy.json에 저장되었습니다.")
        print(f"   → 라이브 봇에서 promoted_strategy.json을 읽어 자동 적용 가능")

        promoted_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "promoted_strategy.json")
        with open(promoted_file, 'r') as f:
            config = json.load(f)
        print(f"\n   프로모션된 파라미터:")
        for k, v in config.get('promoted_params', {}).items():
            print(f"     {k}: {v}")

    return all_results


async def promote_to_live():
    config = load_promoted_config()
    if not config:
        print("❌ 프로모션된 전략이 없습니다. 먼저 토너먼트를 실행하세요.")
        return

    print(f"🔄 프로모션된 전략을 라이브 설정에 반영합니다...")
    print(f"   승리 전략: {config['winning_agent']}")
    print(f"   수익률: {config['winning_ret']:.2f}%")
    print(f"   승률: {config['winning_win_rate']:.1f}%")

    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    env_vars = {}
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    env_vars[key.strip()] = val.strip()

    params = config.get('promoted_params', {})
    if 'ema_period' in params:
        env_vars['OKX_EMA_PERIOD'] = str(params['ema_period'])
    if 'tight_mult' in params:
        env_vars['OKX_SUPERTREND_TIGHT'] = str(params['tight_mult'])
    if 'loose_mult' in params:
        env_vars['OKX_SUPERTREND_LOOSE'] = str(params['loose_mult'])
    if 'tp_thr' in params:
        env_vars['OKX_TP_THRESHOLD'] = str(params['tp_thr'])
    if 'max_dca' in params:
        env_vars['OKX_MAX_DCA_ENTRIES'] = str(params['max_dca'])

    with open(env_file, 'w') as f:
        for key, val in env_vars.items():
            f.write(f"{key}={val}\n")

    print(f"\n✅ .env 파일이 업데이트되었습니다. 라이브 봇 재시작 시 반영됩니다.")
    print(f"   변경된 파라미터:")
    for k, v in params.items():
        print(f"     {k} → {v}")


async def show_promoted():
    config = load_promoted_config()
    if not config:
        print("프로모션된 전략이 없습니다.")
        return
    print(json.dumps(config, indent=4, ensure_ascii=False))


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="ORCA Tournament Engine - Unified Backtest & Live Promotion")
    parser.add_argument("--run", action="store_true", help="Run all tournaments")
    parser.add_argument("--promote", action="store_true", help="Promote best strategy to live config")
    parser.add_argument("--show", action="store_true", help="Show promoted strategy config")
    parser.add_argument("--timeframe", default="15m", help="Timeframe for backtest (default: 15m)")
    parser.add_argument("--set", default=None, help="Run only specific symbol set (e.g., Major_Crypto)")
    args = parser.parse_args()

    if args.run:
        if args.set and args.set in SYMBOL_SETS:
            await run_single_tournament(args.set, SYMBOL_SETS[args.set], args.timeframe)
        else:
            await run_all_tournaments(args.timeframe)

    if args.promote:
        await promote_to_live()

    if args.show:
        await show_promoted()

    if not (args.run or args.promote or args.show):
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())