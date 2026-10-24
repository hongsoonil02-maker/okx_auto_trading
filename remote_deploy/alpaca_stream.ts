// alpaca_stream.ts — Real-time Market Data WebSocket Streamer via Alpaca (IEX feed)
export interface QuoteEntry {
  symbol: string;
  bidPrice: number;
  bidSize: number;
  askPrice: number;
  askSize: number;
  midPrice: number;
  spread: number;
  spreadBps: number;
  imbalance: number; // -1.0 to +1.0
  timestamp: number; // epoch ms
}

export interface TradeEntry {
  symbol: string;
  price: number;
  size: number;
  timestamp: number;
}

export class AlpacaDataStreamer {
  private wsUrl: string;
  private apiKey: string;
  private secretKey: string;
  private symbols: Set<string>;
  private quotes: Map<string, QuoteEntry> = new Map();
  private lastTrades: Map<string, TradeEntry> = new Map();
  private ws: WebSocket | null = null;
  private isRunning: boolean = false;
  private reconnectTimer: any = null;

  constructor(apiKey: string, secretKey: string, initialSymbols: string[] = []) {
    this.apiKey = apiKey;
    this.secretKey = secretKey;
    this.wsUrl = process.env.ALPACA_WS_URL || "wss://stream.data.alpaca.markets/v2/iex";
    this.symbols = new Set(initialSymbols.map(s => s.toUpperCase()));
  }

  public addSymbols(symbols: string[]) {
    for (const s of symbols) {
      this.symbols.add(s.toUpperCase());
    }
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.sendSubscription();
    }
  }

  public getQuote(symbol: string): QuoteEntry | undefined {
    return this.quotes.get(symbol.toUpperCase());
  }

  public getAllQuotes(): Record<string, QuoteEntry> {
    const out: Record<string, QuoteEntry> = {};
    this.quotes.forEach((v, k) => { out[k] = v; });
    return out;
  }

  public formatForJev(symbol: string): string {
    const q = this.getQuote(symbol);
    const t = this.lastTrades.get(symbol.toUpperCase());
    const now = Date.now();

    if (!q) {
      return `[${symbol}] No real-time quote cached yet. Current timestamp: ${now}`;
    }

    const ageMs = now - q.timestamp;
    const imbStr = (q.imbalance >= 0 ? "+" : "") + q.imbalance.toFixed(3);
    let out = `[${symbol} Nasdaq L1/BBO Snapshot | Age: ${ageMs}ms]\n`;
    out += `Best Bid: $${q.bidPrice.toFixed(2)} (Size: ${q.bidSize}) | Best Ask: $${q.askPrice.toFixed(2)} (Size: ${q.askSize})\n`;
    out += `Spread: $${q.spread.toFixed(2)} (${q.spreadBps.toFixed(1)} bps) | Orderbook Imbalance: ${imbStr} (-1.0=Heavy Sell, +1.0=Heavy Buy)\n`;
    if (t) {
      out += `Last Trade: $${t.price.toFixed(2)} (Size: ${t.size}, ${now - t.timestamp}ms ago)\n`;
    }
    return out;
  }

  public async fetchInitialSnapshots() {
    const symList = Array.from(this.symbols);
    if (symList.length === 0) return;
    try {
      console.log(`[AlpacaREST] Fetching initial snapshots for ${symList.join(", ")}...`);
      const res = await fetch(`https://data.alpaca.markets/v2/stocks/snapshots?symbols=${symList.join(",")}&feed=iex`, {
        headers: {
          "APCA-API-KEY-ID": this.apiKey,
          "APCA-API-SECRET-KEY": this.secretKey,
        }
      });
      if (!res.ok) return;
      const data = await res.json();
      for (const [sym, item] of Object.entries<any>(data)) {
        const q = item.latestQuote;
        const t = item.latestTrade;
        if (q) {
          const bp = Number(q.bp) || 0;
          const bs = Number(q.bs) || 0;
          const ap = Number(q.ap) || 0;
          const as = Number(q.as) || 0;
          const mid = (bp + ap) / 2 || bp || ap;
          const spread = ap - bp;
          const spreadBps = mid > 0 ? (spread / mid) * 10000 : 0;
          const totalSize = bs + as;
          const imbalance = totalSize > 0 ? (bs - as) / totalSize : 0;
          const ts = q.t ? new Date(q.t).getTime() : Date.now();
          this.quotes.set(sym, {
            symbol: sym,
            bidPrice: bp,
            bidSize: bs,
            askPrice: ap,
            askSize: as,
            midPrice: mid,
            spread: spread,
            spreadBps: spreadBps,
            imbalance: imbalance,
            timestamp: ts,
          });
        }
        if (t) {
          this.lastTrades.set(sym, {
            symbol: sym,
            price: Number(t.p) || 0,
            size: Number(t.s) || 0,
            timestamp: t.t ? new Date(t.t).getTime() : Date.now(),
          });
        }
      }
      console.log(`[AlpacaREST] ✅ Loaded ${this.quotes.size} initial stock snapshots into cache.`);
    } catch (e) {
      console.warn("[AlpacaREST] Failed to fetch initial snapshots:", e);
    }
  }

  public start() {
    this.isRunning = true;
    this.fetchInitialSnapshots().catch(() => {});
    this.connect();
  }

  public stop() {
    this.isRunning = false;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  private connect() {
    if (!this.isRunning) return;
    console.log(`[AlpacaWS] Connecting to ${this.wsUrl}...`);
    try {
      this.ws = new WebSocket(this.wsUrl);

      this.ws.onopen = () => {
        console.log("[AlpacaWS] ✅ Connected to WebSocket stream.");
      };

      this.ws.onmessage = (event) => {
        try {
          const msgs = JSON.parse(event.data);
          for (const m of msgs) {
            this.handleMessage(m);
          }
        } catch (err) {
          console.error("[AlpacaWS] Parse error:", err);
        }
      };

      this.ws.onclose = (ev) => {
        console.warn(`[AlpacaWS] Disconnected (code=${ev.code}). Reconnecting in 3s...`);
        if (this.isRunning) {
          this.reconnectTimer = setTimeout(() => this.connect(), 3000);
        }
      };

      this.ws.onerror = (err) => {
        console.error("[AlpacaWS] Error:", err);
      };
    } catch (e) {
      console.error("[AlpacaWS] Connection failure:", e);
      if (this.isRunning) {
        this.reconnectTimer = setTimeout(() => this.connect(), 3000);
      }
    }
  }

  private handleMessage(msg: any) {
    if (msg.T === "success" && msg.msg === "connected") {
      console.log("[AlpacaWS] Authenticating with API Key...");
      this.ws?.send(JSON.stringify({
        action: "auth",
        key: this.apiKey,
        secret: this.secretKey,
      }));
    } else if (msg.T === "success" && msg.msg === "authenticated") {
      console.log("[AlpacaWS] 🔑 Authentication verified. Subscribing to symbols...");
      this.sendSubscription();
    } else if (msg.T === "q") {
      const sym = msg.S;
      const bp = Number(msg.bp) || 0;
      const bs = Number(msg.bs) || 0;
      const ap = Number(msg.ap) || 0;
      const as = Number(msg.as) || 0;
      const mid = (bp + ap) / 2 || bp || ap;
      const spread = ap - bp;
      const spreadBps = mid > 0 ? (spread / mid) * 10000 : 0;
      const totalSize = bs + as;
      const imbalance = totalSize > 0 ? (bs - as) / totalSize : 0;
      const ts = msg.t ? new Date(msg.t).getTime() : Date.now();

      this.quotes.set(sym, {
        symbol: sym,
        bidPrice: bp,
        bidSize: bs,
        askPrice: ap,
        askSize: as,
        midPrice: mid,
        spread: spread,
        spreadBps: spreadBps,
        imbalance: imbalance,
        timestamp: ts,
      });
    } else if (msg.T === "t") {
      const sym = msg.S;
      const p = Number(msg.p) || 0;
      const s = Number(msg.s) || 0;
      const ts = msg.t ? new Date(msg.t).getTime() : Date.now();
      this.lastTrades.set(sym, { symbol: sym, price: p, size: s, timestamp: ts });
    }
  }

  private sendSubscription() {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    const symList = Array.from(this.symbols);
    if (symList.length === 0) return;
    console.log(`[AlpacaWS] Subscribing to quotes for ${symList.length} symbols:`, symList.join(", "));
    this.ws.send(JSON.stringify({
      action: "subscribe",
      quotes: symList,
      trades: symList,
    }));
  }
}
