// jev_client.ts — Sub-Second Jev Decision Engine for Nasdaq
export interface JevDecision {
  up_in_10: number; // 0.0 to 1.0
  action: "buy" | "sell" | "neutral";
  confidence: number;
  reason: string;
  latency_ms: number;
  is_fallback: boolean;
}

export class TypesafeJevClient {
  private apiKey: string;
  private model: string;
  private timeoutMs: number;

  constructor(
    apiKey?: string,
    model: string = "google/gemini-3.5-flash-lite",
    timeoutMs: number = 800
  ) {
    this.apiKey = apiKey || process.env.OPENROUTER_API_KEY || "";
    this.model = model || process.env.OPENROUTER_MODEL || "google/gemini-3.5-flash-lite";
    this.timeoutMs = timeoutMs;
  }

  public async predict(stateText: string, imbalanceFallback: number = 0.0): Promise<JevDecision> {
    const t0 = performance.now();

    if (!this.apiKey) {
      return this.heuristicFallback(imbalanceFallback, "NO_API_KEY", performance.now() - t0);
    }

    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);

      const prompt = `You are Jev AI (Typesafe AI System One Ultra-Low-Latency Orderbook Engine).
Evaluate the following real-time Nasdaq Orderbook state and predict 10-second micro-momentum direction:

${stateText}

Return ONLY a valid JSON object:
{
  "up_in_10": <float between 0.00 and 1.00 indicating probability the next trade moves up>,
  "action": <"buy" | "sell" | "neutral">,
  "confidence": <float between 0.00 and 1.00>,
  "reason": <concise rationale, max 8 words>
}`;

      const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${this.apiKey}`,
          "HTTP-Referer": "https://antigravity.nasdaq.quant",
          "X-Title": "Antigravity-Nasdaq-Jev",
        },
        body: JSON.stringify({
          model: this.model,
          messages: [
            { role: "system", content: "You are Jev AI. Respond strictly in valid JSON." },
            { role: "user", content: prompt }
          ],
          response_format: { type: "json_object" },
          temperature: 0.1,
          max_tokens: 100,
        }),
        signal: controller.signal,
      });

      clearTimeout(timer);
      const elapsed = performance.now() - t0;

      if (!res.ok) {
        return this.heuristicFallback(imbalanceFallback, `API_HTTP_${res.status}`, elapsed);
      }

      const body = await res.json();
      const content = body.choices?.[0]?.message?.content || "{}";
      const parsed = JSON.parse(content);

      const score = typeof parsed.up_in_10 === "number" ? parsed.up_in_10 : 0.50;
      const act = ["buy", "sell", "neutral"].includes(parsed.action) ? parsed.action : (score >= 0.65 ? "buy" : (score <= 0.35 ? "sell" : "neutral"));
      const conf = typeof parsed.confidence === "number" ? parsed.confidence : Math.abs(score - 0.5) * 2;
      const reason = parsed.reason || "JEV_INFERENCE_OK";

      return {
        up_in_10: score,
        action: act,
        confidence: conf,
        reason: reason,
        latency_ms: elapsed,
        is_fallback: false,
      };
    } catch (err: any) {
      const elapsed = performance.now() - t0;
      const isTimeout = err.name === "AbortError";
      return this.heuristicFallback(
        imbalanceFallback,
        isTimeout ? "TIMEOUT_FALLBACK" : "EXCEPTION_FALLBACK",
        elapsed
      );
    }
  }

  private heuristicFallback(imbalance: number, reason: string, elapsed: number): JevDecision {
    // Orderbook Imbalance is between -1.0 (heavy sell) and +1.0 (heavy buy)
    const probUp = Math.min(Math.max(0.50 + imbalance * 0.35, 0.10), 0.90);
    const action = probUp >= 0.60 ? "buy" : (probUp <= 0.40 ? "sell" : "neutral");

    return {
      up_in_10: probUp,
      action: action,
      confidence: Math.abs(imbalance),
      reason: `HEURISTIC_${reason}`,
      latency_ms: elapsed,
      is_fallback: true,
    };
  }
}
