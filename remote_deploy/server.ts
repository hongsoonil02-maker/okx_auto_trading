// server.ts — Bun HTTP Sidecar Daemon for Jev Nasdaq Gating & Micro-Quoting
import { existsSync, readFileSync } from "fs";
import { resolve } from "path";
import { AlpacaDataStreamer } from "./alpaca_stream";
import { TypesafeJevClient, JevDecision } from "./jev_client";

// Explicitly load .env file regardless of how Bun was executed
const envPaths = [
  resolve(__dirname, "../.env"),
  "/home/hongsoonil02/jev_nasdaq/.env",
  resolve(process.cwd(), ".env"),
];
for (const p of envPaths) {
  if (existsSync(p)) {
    try {
      const content = readFileSync(p, "utf-8");
      for (const line of content.split("\n")) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith("#")) continue;
        const eqIdx = trimmed.indexOf("=");
        if (eqIdx > 0) {
          const k = trimmed.substring(0, eqIdx).trim();
          const v = trimmed.substring(eqIdx + 1).trim();
          if (!process.env[k]) {
            process.env[k] = v;
          }
        }
      }
      console.log(`[Jev Server] Loaded environment variables from: ${p}`);
      break;
    } catch (e) {
      console.warn(`[Jev Server] Warning reading ${p}:`, e);
    }
  }
}

const PORT = Number(process.env.SERVER_PORT || 8020);
const ALPACA_KEY = process.env.ALPACA_API_KEY || "PKVGU7IXDJMAF6JBWWVMP2I6R3";
const ALPACA_SECRET = process.env.ALPACA_SECRET_KEY || "CCnTinES4eWwkQqifr9879ASCcXSpm9kjuvbZnCKpHM3";
const OPENROUTER_KEY = process.env.OPENROUTER_API_KEY || "";
const OPENROUTER_MODEL = process.env.OPENROUTER_MODEL || "google/gemini-3.5-flash-lite";
const CONFIDENCE_THRESHOLD = Number(process.env.JEV_CONFIDENCE_THRESHOLD || 0.60);
const QUOTE_INSIDE_TICKS = Number(process.env.QUOTE_INSIDE_TICKS || 1);
const IS_SIMULATION = (process.env.JEV_SIMULATION_MODE || "false").toLowerCase() === "true";

const TRACKED_SYMBOLS = ["QQQ", "NVDA", "TSLA", "AAPL", "TQQQ", "SQQQ", "SOXL", "SOXS"];

const streamer = new AlpacaDataStreamer(ALPACA_KEY, ALPACA_SECRET, TRACKED_SYMBOLS);
const jevClient = new TypesafeJevClient(OPENROUTER_KEY, OPENROUTER_MODEL);

streamer.start();

const startTime = Date.now();

console.log(`🚀 [Jev Nasdaq] Starting Bun Sidecar Daemon on port ${PORT}...`);
console.log(`🔑 OpenRouter Key present: ${Boolean(OPENROUTER_KEY)}, Model: ${OPENROUTER_MODEL}, Simulation: ${IS_SIMULATION}`);

const server = Bun.serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);

    // 1. Health Check
    if (url.pathname === "/health") {
      const allQuotes = streamer.getAllQuotes();
      return Response.json({
        status: "ok",
        uptime_seconds: Math.floor((Date.now() - startTime) / 1000),
        simulation_mode: IS_SIMULATION,
        confidence_threshold: CONFIDENCE_THRESHOLD,
        openrouter_configured: Boolean(OPENROUTER_KEY),
        symbols_count: Object.keys(allQuotes).length,
        symbols: Object.keys(allQuotes),
      });
    }

    // 2. All Quotes
    if (url.pathname === "/quotes") {
      return Response.json({
        quotes: streamer.getAllQuotes(),
        timestamp: Date.now(),
      });
    }

    // 3. Predict Endpoint: POST /predict
    if (url.pathname === "/predict" && req.method === "POST") {
      try {
        const body = await req.json();
        const symbol = String(body.symbol || "QQQ").toUpperCase();
        const side = String(body.side || "buy").toLowerCase();
        const tickSize = Number(body.tick_size || 0.01);
        const contextNote = String(body.context || "");

        // 1) Get Cached Quote
        const quote = streamer.getQuote(symbol);
        const stateText = streamer.formatForJev(symbol);
        const imbalance = quote ? quote.imbalance : 0.0;

        // 2) Run Jev Inference
        const decision: JevDecision = await jevClient.predict(stateText, imbalance);

        // 3) Evaluate Gating
        let approved = false;
        if (side === "buy") {
          approved = decision.up_in_10 >= CONFIDENCE_THRESHOLD;
        } else if (side === "sell") {
          approved = decision.up_in_10 <= (1.0 - CONFIDENCE_THRESHOLD);
        }

        // 4) Compute Quote Inside Ticks Maker Price
        let targetPrice: number | null = null;
        let orderType = "MARKET";

        if (quote && quote.bidPrice > 0 && quote.askPrice > 0) {
          orderType = "POST_ONLY";
          if (side === "buy") {
            // Place limit 1 tick above best bid, capped below best ask
            targetPrice = Math.min(quote.bidPrice + QUOTE_INSIDE_TICKS * tickSize, quote.askPrice - tickSize);
            targetPrice = Number(targetPrice.toFixed(2));
          } else {
            // Place limit 1 tick below best ask, floored above best bid
            targetPrice = Math.max(quote.askPrice - QUOTE_INSIDE_TICKS * tickSize, quote.bidPrice + tickSize);
            targetPrice = Number(targetPrice.toFixed(2));
          }
        }

        const logTag = IS_SIMULATION ? "📝 [JEV SIMULATION]" : "🔥 [JEV LIVE]";
        console.log(
          `${logTag} ${symbol} ${side.toUpperCase()} | ` +
          `Score: ${decision.up_in_10.toFixed(3)} | Approved: ${approved} | ` +
          `Type: ${orderType} @ ${targetPrice} | Latency: ${decision.latency_ms.toFixed(1)}ms | ` +
          `Reason: ${decision.reason}`
        );

        return Response.json({
          symbol: symbol,
          side: side,
          approved: approved,
          score: decision.up_in_10,
          action: decision.action,
          confidence: decision.confidence,
          order_type: orderType,
          target_price: targetPrice,
          quote: quote || null,
          latency_ms: decision.latency_ms,
          is_simulation: IS_SIMULATION,
          is_fallback: decision.is_fallback,
          reason: decision.reason,
        });
      } catch (err: any) {
        console.error("❌ [Jev Server] Predict error:", err);
        return Response.json({
          approved: true, // fail-open to baseline
          score: 0.50,
          order_type: "MARKET",
          target_price: null,
          latency_ms: 0,
          is_simulation: IS_SIMULATION,
          is_fallback: true,
          reason: `SERVER_ERROR: ${err.message}`,
        }, { status: 500 });
      }
    }

    return new Response("Not Found", { status: 404 });
  },
});

console.log(`✅ [Jev Nasdaq] Listening on http://127.0.0.1:${PORT}`);
