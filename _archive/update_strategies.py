import re
import os

files_to_update = {
    'okx_strategy.py': {'tgt': 1.015, 'tight': 2.5},
    'okx_stock_strategy.py': {'tgt': 1.015, 'tight': 2.5},
    'okx_venture_strategy.py': {'tgt': 1.025, 'tight': 2.0},
    'okx_stock_venture_strategy.py': {'tgt': 1.025, 'tight': 2.0},
}

for fname, params in files_to_update.items():
    if not os.path.exists(fname):
        print(f"File not found: {fname}")
        continue
    
    with open(fname, 'r', encoding='utf-8') as f:
        content = f.read()

    # We want to replace the body of check_auto_logic from calc_supertrend to the end of the pull_back conditions.
    pattern = re.compile(
        r"(df = pd\.DataFrame\(ohlcv, columns=\['t','o','h','l','c','v'\]\).*?)(st_d.*?)(if is_long_breakout or is_long_pullback:)", 
        re.DOTALL
    )
    
    match = pattern.search(content)
    if not match:
        print(f"Could not match pattern in {fname}")
        continue
    
    prefix = match.group(1)
    old_body = match.group(2)
    suffix = match.group(3)
    
    tgt = params['tgt']
    tight = params['tight']
    
    logger_name = "[Auto Trade]"
    if "Stock" in fname:
        logger_name = "[Stock Trade]"
    if "Venture" in fname or "venture" in fname:
        logger_name = "[Venture Trade]"
    
    if "롱 청산 시그널" in old_body:
        log_match = re.search(r'logger\.info\(f"💨 (\[.*?\]) 롱 청산 시그널', old_body)
        if log_match:
            logger_name = log_match.group(1)

    new_body = f"""
            # 진입용 (Loose) Supertrend (Multiplier 4.0)
            st_d_loose, st_v_loose = calc_supertrend(df, 10, 4.0)
            # 청산/방어용 (Tight) Supertrend (Multiplier {tight})
            st_d_tight, st_v_tight = calc_supertrend(df, 10, {tight})
            
            df['st_d_loose'] = st_d_loose
            df['st_v_loose'] = st_v_loose
            df['st_d_tight'] = st_d_tight
            df['st_v_tight'] = st_v_tight
            
            # Volume MA 20
            df['vol_ma'] = df['v'].rolling(20).mean()
            
            # StochRSI 적용 (K, D선)
            k, d = calc_stoch_rsi(df['c'], 14, 3, 3)
            df['stoch_k'] = k
            df['stoch_d'] = d
            
            prev, curr = df.iloc[-2], df.iloc[-1]
            
            pos_long = self.auto_active_pos.get((symbol, 'long'))
            has_long = pos_long is not None
            avg_price_long = pos_long['avgPrice'] if has_long else 0
            if has_long and avg_price_long > 0:
                self.max_price_state[(symbol, 'long')] = max(self.max_price_state.get((symbol, 'long'), avg_price_long), curr['c'])
                
            pos_short = self.auto_active_pos.get((symbol, 'short'))
            has_short = pos_short is not None
            avg_price_short = pos_short['avgPrice'] if has_short else 0
            if has_short and avg_price_short > 0:
                self.max_price_state[(symbol, 'short')] = min(self.max_price_state.get((symbol, 'short'), avg_price_short), curr['c'])

            vol_cond = curr['v'] > prev['vol_ma'] * 1.2

            # A. 추적 청산 (Trailing Stop 기반)
            if has_long and avg_price_long > 0:
                max_price = self.max_price_state.get((symbol, 'long'), avg_price_long)
                
                # 기본 청산 시그널 (Loose 기준 이탈)
                close_long_sig = curr['st_d_loose'] == -1 or curr['c'] < curr['st_v_loose']
                
                # 수익 {((tgt-1)*100):.1f}% 이상 도달 시 Tight 방어 로직 가동
                if max_price > avg_price_long * {tgt}:
                    if curr['st_d_tight'] == -1 or curr['c'] < curr['st_v_tight']:
                        close_long_sig = True
                # 본절가 이탈 방어 (Breakeven)
                elif max_price > avg_price_long * 1.01:
                    if curr['c'] < avg_price_long * 1.001:
                        close_long_sig = True
                        
                if close_long_sig:
                    logger.info(f"💨 {logger_name} 롱 청산 시그널 (ALL): {{symbol}}")
                    await self.send_webhook(SideType.CLOSE_LONG, symbol, 0)
            
            if has_short and avg_price_short > 0:
                min_price = self.max_price_state.get((symbol, 'short'), avg_price_short)
                
                close_short_sig = curr['st_d_loose'] == 1 or curr['c'] > curr['st_v_loose']
                
                # 수익 {((tgt-1)*100):.1f}% 이상 도달 시 Tight 방어 로직 가동
                if min_price < avg_price_short * (1 - {(tgt-1)}):
                    if curr['st_d_tight'] == 1 or curr['c'] > curr['st_v_tight']:
                        close_short_sig = True
                # 본절가 이탈 방어 (Breakeven)
                elif min_price < avg_price_short * 0.99:
                    if curr['c'] > avg_price_short * 0.999:
                        close_short_sig = True
                        
                if close_short_sig:
                    logger.info(f"💨 {logger_name} 숏 청산 시그널 (ALL): {{symbol}}")
                    await self.send_webhook(SideType.CLOSE_SHORT, symbol, 0)

            # B. 신규 진입
            is_long_breakout = prev['st_d_loose'] == -1 and curr['st_d_loose'] == 1 and vol_cond
            is_short_breakout = prev['st_d_loose'] == 1 and curr['st_d_loose'] == -1 and vol_cond

            is_long_pullback = curr['st_d_loose'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20 and vol_cond
            is_short_pullback = curr['st_d_loose'] == -1 and prev['stoch_k'] > 80 and curr['stoch_k'] <= 80 and vol_cond

            """
    
    new_content = content.replace(old_body, new_body)
    
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(new_content)
        
    print(f"Successfully updated {fname}")

print("Update complete.")
