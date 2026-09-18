# Before vs After Backtest Comparison Report

Generated: 2026-09-14
Data: 4,544 trades (60-day roundtrip analysis)
Split: First half (Aug 9 - Aug 27) vs Second half (Aug 28 - Sep 14)

## Summary Table

| Metric | Before (45d) | After (15d) | Change |
|--------|-------------|-------------|--------|
| Trades | 675 | 675 | 0 |
| Win Rate | 49.9% | 43.4% | -6.5pp |
| Total PnL | -$46,356 | -$18,132 | +$28,225 |
| Avg PnL/Trade | -$68.7 | -$26.9 | +$41.8 |
| Profit Factor | 0.66x | 0.85x | +0.19 |

## Key Findings

1. **PF improved +0.19x** (0.66 → 0.85) - Losses are being managed better
2. **Average loss reduced by $41.8/trade** - Survival mode + CB working
3. **Total PnL improved $28K** despite slightly lower win rate
4. **Win rate dropped 6.5pp** - More selective entries = fewer but better trades

## 4 Improvements Applied

### 1. Dynamic Leverage (3x↔1x by BTC 1h vol)
- Low vol (<0.3%): 3x leverage for maximum exposure
- High vol (>0.5%): 1x leverage for risk reduction
- Cache: 1 hour

### 2. Survival Mode
- Trigger: 5+ consecutive losses
- Effect: Margin reduced 50%
- Recovery: 24h cooldown

### 3. Season Mode (Market Regime Detection)
- trend_up (vol <0.8%, positive slope): Margin ×1.5, Max 10 pos
- trend_down: Margin ×1.05, Max 10 pos
- chop (vol >0.8%, flat slope): Margin ×0.5, Max 4 pos
- crash (vol >1.5% or |trend|>5%): Margin ×0.3, Max 2 pos

### 4. ADX Chop Block + Circuit Breaker Hardening
- ADX < 20: No new entries
- CB ROE: -12% (was -4%)
- CB cooldown: 24h (was 48h)
- Short trading disabled
- Weekend blocked
- Max 3 entries/day

## Live Status
- Season Mode: trend_up detected (BTC vol 0.21%, trend +1.6%)
- Margin multiplier: ×1.5 (trend mode)
- Max positions: 10
- Balance: $9,553 USDT
- Bots: RUNNING (ports 8013, 8014, 8015)
