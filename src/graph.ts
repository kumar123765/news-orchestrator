import { START, END, StateGraph } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
import { tools } from "./tools.js";
import type { OrchestratorState } from "./state.js";

const llm = new ChatOpenAI({ model: process.env.OPENAI_MODEL || "gpt-4o-mini", temperature: 0 });

// Router: classify intent & extract slots
async function router(state: OrchestratorState): Promise<Partial<OrchestratorState>> {
  const schema = z.object({
    intent: z.enum(["HEADLINES","TOPIC","BRIEF","HISTORY","LOCAL","SUMMARIZE","UNKNOWN"]),
    topic: z.string().optional(),
    city: z.string().optional(),
    dateISO: z.string().optional(),
    url: z.string().optional()
  });
  const sys =
    "Classify the user's query for a news assistant in India. " +
    "INTENTS: HEADLINES, TOPIC, BRIEF, HISTORY, LOCAL, SUMMARIZE, UNKNOWN. Extract slots if present.";
  const prompt = [{ role: "system", content: sys }, { role: "user", content: state.input.query }];
  const out = await llm.withStructuredOutput(schema).invoke(prompt as any);
  return out;
}

// Actions
async function doHeadlines(_: OrchestratorState) {
  const out = await tools.topHeadlines();
  return { headlines: out.result?.articles, final: out.result };
}

async function doTopic(state: OrchestratorState) {
  if (!state.topic) return { followupQuestion: "Which topic?" };
  const out = await tools.topicNews(state.topic, 10);
  return { topicArticles: out.result?.articles, final: out.result };
}

async function doHistory(state: OrchestratorState) {
  const out = await tools.onThisDay(state.dateISO);
  return { historyEvents: out.result, final: out.result };
}

async function doLocal(state: OrchestratorState) {
  const out = await tools.aroundYou(state.city, 8, state.input.session_id);
  if (out.result?.need_followup) return { followupQuestion: out.result.question };
  return { final: out.result };
}

async function doSummarize(state: OrchestratorState) {
  if (!state.url) return { followupQuestion: "Please share the article link (URL)." };
  const out = await tools.summarizeUrl(state.url);
  return { final: out.result };
}

async function doBrief(state: OrchestratorState) {
  if (!state.topic) return { followupQuestion: "Brief on which topic?" };
  const news = await tools.topicNews(state.topic, 8);
  const arts = (news.result?.articles ?? []).slice(0, 3);

  const summaries: { url: string; summary: string }[] = [];
  for (const a of arts) {
    const url = a?.url;
    if (!url) continue;
    const s = await tools.summarizeUrl(url);
    summaries.push({ url, summary: s.result?.summary ?? "" });
  }

  const items = summaries.map(s => `- ${s.summary}`).join("\n").slice(0, 15000);
  const synth = await llm.invoke([
    { role: "system", content: "Create a concise 7–9 bullet executive brief for Indian readers. Keep dates & numbers exact. End with: 'What to watch'." },
    { role: "user", content: `Topic: ${state.topic}\nSummaries:\n${items}` }
  ] as any);

  return { summaries, brief: synth.content?.toString().trim(), final: { type: "multi_source_brief", topic: state.topic, sources: arts, brief: synth.content } };
}

export function buildGraph() {
  const g = new StateGraph<OrchestratorState>({
    channels: {
      input: "replace",
      intent: "replace",
      topic: "replace",
      city: "replace",
      dateISO: "replace",
      url: "replace",
      headlines: "replace",
      topicArticles: "replace",
      summaries: "replace",
      brief: "replace",
      historyEvents: "replace",
      followupQuestion: "replace",
      final: "replace",
    },
  });

  g.addNode("router", router);
  g.addNode("headlines", doHeadlines);
  g.addNode("topic", doTopic);
  g.addNode("history", doHistory);
  g.addNode("local", doLocal);
  g.addNode("summarize", doSummarize);
  g.addNode("brief", doBrief);

  g.addEdge(START, "router");
  g.addConditionalEdges("router", (s) => {
    switch (s.intent) {
      case "HEADLINES": return "headlines";
      case "TOPIC":     return "topic";
      case "HISTORY":   return "history";
      case "LOCAL":     return "local";
      case "SUMMARIZE": return "summarize";
      case "BRIEF":     return "brief";
      default:          return "headlines";
    }
  }, { headlines: "headlines", topic: "topic", history: "history", local: "local", summarize: "summarize", brief: "brief" });

  g.addEdge("headlines", END);
  g.addEdge("topic", END);
  g.addEdge("history", END);
  g.addEdge("local", END);
  g.addEdge("summarize", END);
  g.addEdge("brief", END);

  return g.compile();
}
