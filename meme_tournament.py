import asyncio
import json

from tournament_common import run_tournament, simulate, simulate_vibe

MEME_SYMBOLS = ['PEPE/USDT:USDT', 'DOGE/USDT:USDT', 'WIF/USDT:USDT', 'ORDI/USDT:USDT', 'SHIB/USDT:USDT', 'MEME/USDT:USDT']

STRATEGIES = {
    "Current_Meme_Base": {
        "func": "simulate",
        "params": {"ema_period": 200, "tight_mult": 2.0, "loose_mult": 4.0, "vol_mult": 1.0, "min_hold": 3, "max_dca": 3, "tp_thr": 1.025, "scale_out": False},
    },
    "Geumgang_Logic": {
        "func": "simulate_vibe",
        "params": {},
    },
}


async def main():
    print("📡 Meme/Alt Coin Tournament 시작...")
    results = {}

    for agent_name, config in STRATEGIES.items():
        print(f"\n🔬 [{agent_name}] 백테스트 실행 중...")
        result = run_tournament(
            agent_name=agent_name,
            symbols=MEME_SYMBOLS,
            strategy_func=globals()[config["func"]],
            strategy_params=config["params"],
            timeframe="15m",
        )
        if result:
            results[agent_name] = result
            print(f"  ✅ {agent_name}: 누적 수익률 {result['overall_ret']:.2f}% (승률 {result['overall_win_rate']:.1f}%)")

    if results:
        from tournament_common import save_results_history, promote_strategy
        save_results_history(results)

        best_agent = max(results.values(), key=lambda x: x['overall_ret'])
        print(f"\n🏆 최우승: {best_agent['agent_name']} (누적 수익률: {best_agent['overall_ret']:.2f}%)")

        promoted = promote_strategy(best_agent, STRATEGIES[best_agent['agent_name']]['params'])
        print(f"📌 프로모션된 전략 파라미터가 promoted_strategy.json에 저장되었습니다.")

    print(json.dumps(results, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())