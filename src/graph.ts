// src/graph.ts
import { START, END, StateGraph, Annotation } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
import { tools } from "./tools.js";
import type { OrchestratorState } from "./state.js";

const llm = new ChatOpenAI({
  model: process.env.OPENAI_MODEL || "gpt-4o-mini",
  temperature: 0,
});

/**
 * LangGraph v0.4 state via Annotations
 * - Use Annotation<any>() generics
 * - Node IDs are prefixed with "node_" to avoid colliding with channel names
 */
const State = Annotation.Root({
  // request payload: { query: string; session_id?: string }
  input: Annotation<any>(),

  // routing slots
  intent: Annotation<OrchestratorState["intent"] | undefined>(),
  topic: Annotation<string | undefined>(),
  city: Annotation<string | undefined>(),
  dateISO: Annotation<string | undefined>(),
  url: Annotation<string | undefined>(),

  // outputs
  headlines: Annotation<any | undefined>(),
  topicArticles: Annotation<any | undefined>(),
  summaries: Annotation<any[] | undefined>(),
  brief: Annotation<string | undefined>(),
  historyEvents: Annotation<any | undefined>(),
  followupQuestion: Annotation<string | undefined>(),
  final: Annotation<any | undefined>(),
});

type GraphState = typeof State.State;

/** Router: classify intent & extract slots (Structured Outputs requires required+nullable) */
async function router(state: GraphState) {
  const schema = z.object({
    intent: z.enum([
      "HEADLINES",
      "TOPIC",
      "BRIEF",
      "HISTORY",
      "LOCAL",
      "SUMMARIZE",
      "UNKNOWN",
    ]),
    topic: z.string().nullable(),
    city: z.string().nullable(),
    dateISO: z.string().nullable(),
    url: z.string().nullable(),
  });

  const sys =
    "Classify the user's query for a news assistant (India-focused). " +
    "Return JSON with keys: intent, topic, city, dateISO, url. " +
    "intents: HEADLINES, TOPIC, BRIEF, HISTORY, LOCAL, SUMMARIZE, UNKNOWN.";

  const prompt = [
    { role: "system", content: sys },
    { role: "user", content: state.input?.query ?? "" },
  ];

  const out = await llm.withStructuredOutput(schema).invoke(prompt as any);

  // Normalize null -> undefined for downstream
  return {
    intent: out.intent ?? "HEADLINES",
    topic: out.topic ?? undefined,
    city: out.city ?? undefined,
    dateISO: out.dateISO ?? undefined,
    url: out.url ?? undefined,
  };
}

/** Actions */
async function doHeadlines(_: GraphState) {
  const resp = await tools.topHeadlines();
  return { headlines: resp.result?.articles, final: resp.result };
}

async function doTopic(state: GraphState) {
  if (!state.topic) return { followupQuestion: "Which topic?" };
  const resp = await tools.topicNews(state.topic, 10);
  return { topicArticles: resp.result?.articles, final: resp.result };
}

async function doHistory(state: GraphState) {
  const resp = await tools.onThisDay(state.dateISO);
  return { historyEvents: resp.result, final: resp.result };
}

async function doLocal(state: GraphState) {
  const resp = await tools.aroundYou(state.city, 8, state.input?.session_id);
  if (resp.result?.need_followup) return { followupQuestion: resp.result.question };
  return { final: resp.result };
}

async function doSummarize(state: GraphState) {
  if (!state.url) return { followupQuestion: "Please share the article link (URL)." };
  const resp = await tools.summarizeUrl(state.url);
  return { final: resp.result };
}

async function doBrief(state: GraphState) {
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

  const items = summaries.map((s) => `- ${s.summary}`).join("\n").slice(0, 15000);
  const synth = await llm.invoke([
    {
      role: "system",
      content:
        "Create a concise 7–9 bullet executive brief for Indian readers. Keep dates & numbers exact. End with: 'What to watch'.",
    },
    { role: "user", content: `Topic: ${state.topic}\nSummaries:\n${items}` },
  ] as any);

  return {
    summaries,
    brief: synth.content?.toString().trim(),
    final: {
      type: "multi_source_brief",
      topic: state.topic,
      sources: arts,
      brief: synth.content,
    },
  };
}

/** Build & compile graph */
export function buildGraph() {
  const g = new StateGraph(State);

  // Node IDs must be different from channel names
  g.addNode("node_router", router);
  g.addNode("node_headlines", doHeadlines);
  g.addNode("node_topic", doTopic);
  g.addNode("node_history", doHistory);
  g.addNode("node_local", doLocal);
  g.addNode("node_summarize", doSummarize);
  g.addNode("node_brief", doBrief);

  g.addEdge(START as any, "node_router" as any);

  g.addConditionalEdges(
    "node_router" as any,
    (s: GraphState) => {
      switch (s.intent) {
        case "HEADLINES": return "node_headlines";
        case "TOPIC":     return "node_topic";
        case "HISTORY":   return "node_history";
        case "LOCAL":     return "node_local";
        case "SUMMARIZE": return "node_summarize";
        case "BRIEF":     return "node_brief";
        default:          return "node_headlines";
      }
    },
    {
      node_headlines: "node_headlines",
      node_topic: "node_topic",
      node_history: "node_history",
      node_local: "node_local",
      node_summarize: "node_summarize",
      node_brief: "node_brief",
    } as any
  );

  g.addEdge("node_headlines" as any, END as any);
  g.addEdge("node_topic" as any, END as any);
  g.addEdge("node_history" as any, END as any);
  g.addEdge("node_local" as any, END as any);
  g.addEdge("node_summarize" as any, END as any);
  g.addEdge("node_brief" as any, END as any);

  return g.compile();
}
