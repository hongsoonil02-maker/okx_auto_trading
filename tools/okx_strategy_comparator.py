import os
import shutil
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==============================================================================
# OKX Momentum Strategy Visual Comparator (Before vs After)
# - Before: Gemini Original 15m Scalping Setup (High Whipsaw, Microstructure Noise)
# - After:  Quant Guardian 1H Trend Following Setup (Noise Filtered, Clamped Risk)
# ==============================================================================

SYMBOL = "BTC/USDT:USDT"
INITIAL_CAPITAL = 10_000.0
RISK_PER_TRADE = 0.01
RR_RATIO = 2.0
FEE_RATE = 0.0005  # 0.05% one-way maker/taker + slippage buffer
MAX_LEVERAGE = 3.0

def calc_hma(series, length=100):
    def wma(s, l):
        weights = np.arange(1, l + 1)
        return s.rolling(l).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    diff = 2 * wma(series, half_length) - wma(series, length)
    return wma(diff, sqrt_length)

def calc_macd(series, fast=12, slow=26, signal=9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calc_rsi(series, length=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def fetch_okx_candles(symbol, timeframe, limit=1500):
    exchange = ccxt.okx({'enableRateLimit': True})
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    now = exchange.milliseconds()
    since = now - (limit * tf_ms)
    
    all_ohlcv = []
    print(f"[{symbol} - {timeframe}] Fetching historical candles from OKX API...")
    while len(all_ohlcv) < limit:
        batch_limit = min(100, limit - len(all_ohlcv))
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=batch_limit)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + tf_ms
        if len(ohlcv) < batch_limit:
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def simulate_strategy(df, mode="BEFORE"):
    """
    mode 'BEFORE':
      - 15m, unconstrained sizing (risk / risk_dist, can exceed max leverage)
      - standard 10-bar swing low
    mode 'AFTER':
      - 1H, leverage clamped (max 3x leverage * 0.95 capital)
      - minimum 0.3% risk buffer to prevent immediate tick stop-out
    """
    df = df.copy()
    df['hma100'] = calc_hma(df['close'], 100)
    df['macd'], df['macd_signal'], _ = calc_macd(df['close'], 12, 26, 9)
    df['rsi'] = calc_rsi(df['close'], 14)
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['swing_low'] = df['low'].shift(1).rolling(10).min()

    capital = INITIAL_CAPITAL
    equity_curve = [{'datetime': df['datetime'].iloc[119], 'capital': capital}]
    trades = []

    in_pos = False
    entry_price = stop_loss = tp1_price = pos_size = 0.0
    tp1_hit = False
    entry_idx = 0

    for i in range(120, len(df)):
        c_open = df['open'].iloc[i]
        c_high = df['high'].iloc[i]
        c_low = df['low'].iloc[i]
        c_time = df['datetime'].iloc[i]

        if in_pos:
            # 1. Stop Loss check
            if c_low <= stop_loss:
                exit_price = stop_loss
                pnl = (exit_price - entry_price) * pos_size - (entry_price * pos_size + exit_price * pos_size) * FEE_RATE
                capital += pnl
                trades.append({
                    'type': 'SL' if not tp1_hit else 'BE',
                    'entry_time': df['datetime'].iloc[entry_idx],
                    'exit_time': c_time,
                    'pnl': pnl,
                    'return_pct': (exit_price / entry_price - 1) * 100,
                    'capital': capital
                })
                in_pos = False
                equity_curve.append({'datetime': c_time, 'capital': capital})
                continue

            # 2. Take Profit 1 (+2R: 50% scale out & Move SL to BE)
            if not tp1_hit and c_high >= tp1_price:
                half_size = pos_size / 2.0
                exit_price = tp1_price
                pnl = (exit_price - entry_price) * half_size - (entry_price * half_size + exit_price * half_size) * FEE_RATE
                capital += pnl
                trades.append({
                    'type': 'TP1',
                    'entry_time': df['datetime'].iloc[entry_idx],
                    'exit_time': c_time,
                    'pnl': pnl,
                    'return_pct': (exit_price / entry_price - 1) * 100,
                    'capital': capital
                })
                pos_size -= half_size
                tp1_hit = True
                stop_loss = entry_price  # Move stop loss to break-even
                equity_curve.append({'datetime': c_time, 'capital': capital})

            # 3. Technical Indicator Exit (MACD Dead Cross or RSI < 50)
            p1_macd, p1_sig = df['macd'].iloc[i-1], df['macd_signal'].iloc[i-1]
            p2_macd, p2_sig = df['macd'].iloc[i-2], df['macd_signal'].iloc[i-2]
            p1_rsi, p2_rsi = df['rsi'].iloc[i-1], df['rsi'].iloc[i-2]

            dead_cross = (p2_macd >= p2_sig) and (p1_macd < p1_sig)
            rsi_break = (p2_rsi >= 50) and (p1_rsi < 50)
            if dead_cross or rsi_break:
                exit_price = c_open
                pnl = (exit_price - entry_price) * pos_size - (entry_price * pos_size + exit_price * pos_size) * FEE_RATE
                capital += pnl
                trades.append({
                    'type': 'SIGNAL_EXIT',
                    'entry_time': df['datetime'].iloc[entry_idx],
                    'exit_time': c_time,
                    'pnl': pnl,
                    'return_pct': (exit_price / entry_price - 1) * 100,
                    'capital': capital
                })
                in_pos = False
                equity_curve.append({'datetime': c_time, 'capital': capital})
                continue

        else:
            # Entry condition evaluation
            p1_close, p1_hma, p2_hma = df['close'].iloc[i-1], df['hma100'].iloc[i-1], df['hma100'].iloc[i-2]
            hma_rising = (p1_hma > p2_hma) and (p1_close > p1_hma)

            p1_macd, p1_sig = df['macd'].iloc[i-1], df['macd_signal'].iloc[i-1]
            p2_macd, p2_sig = df['macd'].iloc[i-2], df['macd_signal'].iloc[i-2]
            p1_rsi, p2_rsi = df['rsi'].iloc[i-1], df['rsi'].iloc[i-2]
            p1_vol, p1_vol_ma = df['volume'].iloc[i-1], df['vol_ma20'].iloc[i-1]
            vol_ok = p1_vol > p1_vol_ma

            entry = False
            if hma_rising and vol_ok:
                if (p2_macd <= p2_sig) and (p1_macd > p1_sig) and (50 <= p1_rsi < 70):
                    entry = True
                elif (p1_macd > p1_sig) and (p2_rsi < 50 and p1_rsi >= 50):
                    entry = True

            if entry:
                entry_price = c_open
                raw_sl = df['swing_low'].iloc[i-1] * 0.999
                dist = entry_price - raw_sl

                # Guard in AFTER mode: ensure minimum distance
                if mode == "AFTER" and dist < entry_price * 0.003:
                    raw_sl = entry_price * 0.997
                    dist = entry_price - raw_sl

                if dist > 0:
                    raw_size = (capital * RISK_PER_TRADE) / dist
                    if mode == "AFTER":
                        # Clamp size to max leverage margin (3x)
                        max_allowed_size = (capital * MAX_LEVERAGE * 0.95) / entry_price
                        pos_size = min(raw_size, max_allowed_size)
                    else:
                        pos_size = raw_size

                    stop_loss = raw_sl
                    tp1_price = entry_price + (dist * RR_RATIO)
                    tp1_hit = False
                    in_pos = True
                    entry_idx = i

    eq_df = pd.DataFrame(equity_curve)
    eq_df.drop_duplicates(subset=['datetime'], keep='last', inplace=True)
    eq_df.sort_values('datetime', inplace=True)
    eq_df.reset_index(drop=True, inplace=True)
    
    trades_df = pd.DataFrame(trades)
    return trades_df, eq_df

def compute_metrics(trades_df, eq_df):
    if trades_df.empty or eq_df.empty:
        return {
            'roi': 0.0, 'trades': 0, 'win_rate': 0.0,
            'pf': 0.0, 'mdd': 0.0, 'final_cap': INITIAL_CAPITAL, 'types': {}
        }
    final_cap = eq_df['capital'].iloc[-1]
    roi = ((final_cap / INITIAL_CAPITAL) - 1) * 100
    n_trades = len(trades_df)
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    win_rate = (len(wins) / n_trades) * 100 if n_trades > 0 else 0.0
    profit = wins['pnl'].sum()
    loss = abs(losses['pnl'].sum())
    pf = (profit / loss) if loss > 0 else 99.9
    
    peak = eq_df['capital'].cummax()
    drawdown = (eq_df['capital'] - peak) / peak
    mdd = abs(drawdown.min()) * 100
    types = trades_df['type'].value_counts().to_dict()
    
    return {
        'roi': roi, 'trades': n_trades, 'win_rate': win_rate,
        'pf': pf, 'mdd': mdd, 'final_cap': final_cap, 'types': types
    }

def generate_comparison_chart(df_15m_eq, m_15m, df_1h_eq, m_1h, out_paths):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), gridspec_kw={'height_ratios': [2.5, 1]})
    fig.patch.set_facecolor('#0d1117')

    # Color palette
    col_red = '#ff5370'
    col_green = '#00e676'
    col_orange = '#ffb454'
    col_blue = '#82aaff'
    col_gray = '#8b949e'

    for row in axes:
        for ax in row:
            ax.set_facecolor('#161b22')
            ax.grid(True, linestyle='--', alpha=0.25, color='#30363d')
            ax.tick_params(colors='#c9d1d9', which='both', labelsize=10)
            for spine in ax.spines.values():
                spine.set_color('#30363d')

    # -------------------------------------------------------------
    # LEFT COLUMN: BEFORE (15m Gemini Scalp)
    # -------------------------------------------------------------
    ax_b_eq = axes[0, 0]
    ax_b_dd = axes[1, 0]

    # Equity
    df_15m_eq['peak'] = df_15m_eq['capital'].cummax()
    ax_b_eq.plot(df_15m_eq['datetime'], df_15m_eq['capital'], color=col_red, linewidth=2.2, label='15m Equity Curve ($)')
    ax_b_eq.plot(df_15m_eq['datetime'], df_15m_eq['peak'], color=col_gray, linestyle=':', alpha=0.6, label='Peak High')
    ax_b_eq.axhline(INITIAL_CAPITAL, color='#58a6ff', linestyle='--', alpha=0.5, label='Initial ($10,000)')
    ax_b_eq.set_title("[BEFORE] Gemini Original: 15m Momentum Scalp\n(Whipsaw Trap & Early Indicator Shakeout)",
                      color='#ff7b72', fontsize=13, fontweight='bold', pad=10)
    ax_b_eq.set_ylabel("Portfolio Value (USDT)", color='#c9d1d9', fontsize=11)
    ax_b_eq.legend(loc='lower left', facecolor='#21262d', edgecolor='none', labelcolor='#c9d1d9', fontsize=9)

    # Stat box BEFORE
    stat_b = (f"Period: 15m (~16 Days)\n"
              f"ROI: {m_15m['roi']:+.2f}%\n"
              f"Win Rate: {m_15m['win_rate']:.1f}%\n"
              f"Profit Factor: {m_15m['pf']:.2f}\n"
              f"MDD: {m_15m['mdd']:.2f}%\n"
              f"Trades: {m_15m['trades']} (SigExit: {m_15m['types'].get('SIGNAL_EXIT', 0)})")
    ax_b_eq.text(0.97, 0.95, stat_b, transform=ax_b_eq.transAxes,
                 fontsize=10, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', edgecolor=col_red, alpha=0.9),
                 color='#ffffff', family='monospace')

    # Drawdown
    df_15m_eq['dd'] = ((df_15m_eq['capital'] - df_15m_eq['peak']) / df_15m_eq['peak']) * 100
    ax_b_dd.plot(df_15m_eq['datetime'], df_15m_eq['dd'], color=col_red, linewidth=1.5)
    ax_b_dd.fill_between(df_15m_eq['datetime'], df_15m_eq['dd'], 0, color=col_red, alpha=0.25)
    ax_b_dd.set_title("Underwater Drawdown (%)", color='#c9d1d9', fontsize=11, pad=6)
    ax_b_dd.set_ylabel("Drawdown %", color='#c9d1d9', fontsize=10)
    ax_b_dd.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    # -------------------------------------------------------------
    # RIGHT COLUMN: AFTER (1H Guardian Trend)
    # -------------------------------------------------------------
    ax_a_eq = axes[0, 1]
    ax_a_dd = axes[1, 1]

    # Equity
    df_1h_eq['peak'] = df_1h_eq['capital'].cummax()
    ax_a_eq.plot(df_1h_eq['datetime'], df_1h_eq['capital'], color=col_green, linewidth=2.5, label='1H Equity Curve ($)')
    ax_a_eq.plot(df_1h_eq['datetime'], df_1h_eq['peak'], color=col_orange, linestyle=':', alpha=0.8, label='High Watermark')
    ax_a_eq.axhline(INITIAL_CAPITAL, color='#58a6ff', linestyle='--', alpha=0.5, label='Initial ($10,000)')
    ax_a_eq.set_title("[AFTER] Quant Guardian: 1H Trend Following\n(Macro Filter + Clamped Leverage + Free-Ride TP)",
                      color='#3fb950', fontsize=13, fontweight='bold', pad=10)
    ax_a_eq.set_ylabel("Portfolio Value (USDT)", color='#c9d1d9', fontsize=11)
    ax_a_eq.legend(loc='lower left', facecolor='#21262d', edgecolor='none', labelcolor='#c9d1d9', fontsize=9)

    # Stat box AFTER
    stat_a = (f"Period: 1H (~62 Days)\n"
              f"ROI: {m_1h['roi']:+.2f}%\n"
              f"Win Rate: {m_1h['win_rate']:.1f}%\n"
              f"Profit Factor: {m_1h['pf']:.2f}\n"
              f"MDD: {m_1h['mdd']:.2f}%\n"
              f"Trades: {m_1h['trades']} (TP1: {m_1h['types'].get('TP1', 0)})")
    ax_a_eq.text(0.97, 0.95, stat_a, transform=ax_a_eq.transAxes,
                 fontsize=10, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', edgecolor=col_green, alpha=0.9),
                 color='#ffffff', family='monospace')

    # Drawdown
    df_1h_eq['dd'] = ((df_1h_eq['capital'] - df_1h_eq['peak']) / df_1h_eq['peak']) * 100
    ax_a_dd.plot(df_1h_eq['datetime'], df_1h_eq['dd'], color='#f0883e', linewidth=1.5)
    ax_a_dd.fill_between(df_1h_eq['datetime'], df_1h_eq['dd'], 0, color='#f0883e', alpha=0.25)
    ax_a_dd.set_title("Underwater Drawdown (%)", color='#c9d1d9', fontsize=11, pad=6)
    ax_a_dd.set_ylabel("Drawdown %", color='#c9d1d9', fontsize=10)
    ax_a_dd.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    fig.autofmt_xdate()
    plt.tight_layout()

    for p in out_paths:
        plt.savefig(p, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
        print(f"Chart successfully saved to: {p}")
    plt.close()

def main():
    # 1. Fetch data
    df_15m = fetch_okx_candles(SYMBOL, "15m", limit=1500)
    df_1h = fetch_okx_candles(SYMBOL, "1h", limit=1500)

    # 2. Run simulation
    trades_15m, eq_15m = simulate_strategy(df_15m, mode="BEFORE")
    trades_1h, eq_1h = simulate_strategy(df_1h, mode="AFTER")

    # 3. Compute metrics
    m_15m = compute_metrics(trades_15m, eq_15m)
    m_1h = compute_metrics(trades_1h, eq_1h)

    print("\n" + "="*60)
    print("                BEFORE vs AFTER COMPARISON                ")
    print("="*60)
    print(f"[BEFORE - 15m Scalp] ROI: {m_15m['roi']:+.2f}% | WinRate: {m_15m['win_rate']:.1f}% | PF: {m_15m['pf']:.2f} | MDD: {m_15m['mdd']:.2f}% | Trades: {m_15m['trades']}")
    print(f"[AFTER  - 1H Trend ] ROI: {m_1h['roi']:+.2f}% | WinRate: {m_1h['win_rate']:.1f}% | PF: {m_1h['pf']:.2f} | MDD: {m_1h['mdd']:.2f}% | Trades: {m_1h['trades']}")
    print("="*60)

    # 4. Generate visual comparison chart
    artifact_dir = "/home/hongsoonil02/.gemini/antigravity-ide/brain/01fe45ad-88ec-482a-a506-49c25ca6dbce"
    out_paths = [
        "/home/hongsoonil02/quant_system/tools/okx_before_after_backtest.png",
        os.path.join(artifact_dir, "okx_before_after_backtest.png")
    ]
    generate_comparison_chart(eq_15m, m_15m, eq_1h, m_1h, out_paths)

if __name__ == '__main__':
    main()
