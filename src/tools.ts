import fetch from "node-fetch";

const EDGE_URL = process.env.NEWSBOT_EDGE_URL!;
type EdgeOut = { ok: boolean; result?: any; tool?: string; error?: string };

async function callEdge(query: string, extra: Record<string, any> = {}): Promise<EdgeOut> {
  const r = await fetch(EDGE_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, ...extra }),
  });
  return (await r.json()) as EdgeOut;
}

export const tools = {
  topHeadlines: (opts?: { country?: string; lang?: string; max?: number }) =>
    callEdge("top headlines", opts || {}),
  topicNews: (topic: string, max = 10) =>
    callEdge(`latest on ${topic}`, { max_results: max }),
  onThisDay: (iso?: string) =>
    callEdge(iso ? `events on ${iso}` : "major events today"),
  aroundYou: (city?: string, max = 8, session_id?: string) =>
    callEdge(city ? `news in ${city}` : "news around me", { max_results: max, session_id }),
  summarizeUrl: (url: string) =>
    callEdge(`summarize ${url}`),
};
