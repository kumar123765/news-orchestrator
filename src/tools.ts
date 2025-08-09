// src/tools.ts
import fetch from "node-fetch";

const EDGE_URL = process.env.NEWSBOT_EDGE_URL!;
type EdgeOut = { ok?: boolean; result?: any; tool?: string; error?: string; text?: string };

async function callEdge(query: string, extra: Record<string, any> = {}, tries = 2): Promise<EdgeOut> {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), 15000); // 15s timeout

  try {
    const r = await fetch(EDGE_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, ...extra }),
      signal: controller.signal,
    });
    clearTimeout(t);

    let json: EdgeOut | null = null;
    try {
      json = (await r.json()) as EdgeOut;
    } catch {
      const txt = await r.text().catch(() => "");
      return { ok: false, error: `Edge non-JSON (${r.status})`, text: txt };
    }
    if (!json.ok) return { ok: false, error: json.error || `Edge returned ok=false` };
    return { ok: true, result: json.result ?? json };
  } catch (e: any) {
    clearTimeout(t);
    if (tries > 0) return callEdge(query, extra, tries - 1);
    return { ok: false, error: e?.name === "AbortError" ? "Edge timeout" : String(e) };
  }
}

export const tools = {
  async topHeadlines(opts?: { country?: string; lang?: string; max?: number }) {
    return callEdge("top headlines", opts || {});
  },
  async topicNews(topic: string, max = 10) {
    return callEdge(`latest on ${topic}`, { max_results: max });
  },
  async onThisDay(iso?: string) {
    return callEdge(iso ? `events on ${iso}` : "major events today");
  },
  async aroundYou(city?: string, max = 8, session_id?: string) {
    return callEdge(city ? `news in ${city}` : "news around me", { max_results: max, session_id });
  },
  async summarizeUrl(url: string) {
    return callEdge(`summarize ${url}`);
  },
};
