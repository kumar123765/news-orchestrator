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
 * - Use generics like Annotation<any>() (not Annotation.Any()).
 * - We keep types readable and tolerant for CI builds.
 */
const State = Annotation.Root({
  // incoming payload: { query: string; session_id?: string }
  input: Annotation<any>(),

  // routing fields
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

/** Router: classify intent & extract slots */
async function router(state: GraphState) {
  const schema = z.object({
    intent: z
      .enum(["HEADLINES", "TOPIC", "BRIEF", "HISTORY", "LOCAL", "SUMMARIZE", "UNKNOWN"])
      .optional(),
    topic: z.string().optional(),
    city: z.string().optional(),
    dateISO: z.string().optional(),
    url: z.string().optional(),
  });

  const sys =
    "Classify the user's query for a news assistant in India. " +
    "INTENTS: HEADLINES, TOPIC, BRIEF, HISTORY, LOCAL, SUMMARIZE, UNKNOWN. Extract slots if present.";

  const prompt = [
    { role: "system", content: sys },
    { role: "user", content: state.input?.query ?? "" },
  ];

  const out = await llm.withStructuredOutput(schema).invoke(prompt as any);
  return {
    intent: out.intent ?? "HEADLINES",
    topic: out.topic,
    city: out.city,
    dateISO: out.dateISO,
    url: out.url,
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
    final: { type: "multi_source_brief", topic: state.topic, sources: arts, brief: synth.content },
  };
}

/** Build & compile graph */
export function buildGraph() {
  const g = new StateGraph(State);

  g.addNode("router", router);
  g.addNode("headlines", doHeadlines);
  g.addNode("topic", doTopic);
  g.addNode("history", doHistory);
  g.addNode("local", doLocal);
  g.addNode("summarize", doSummarize);
  g.addNode("brief", doBrief);

  // Cast node IDs to 'any' to avoid strict generic constraints in v0.4
  g.addEdge(START as any, "router" as any);

  g.addConditionalEdges(
    "router" as any,
    (s: GraphState) => {
      switch (s.intent) {
        case "HEADLINES": return "headlines";
        case "TOPIC":     return "topic";
        case "HISTORY":   return "history";
        case "LOCAL":     return "local";
        case "SUMMARIZE": return "summarize";
        case "BRIEF":     return "brief";
        default:          return "headlines";
      }
    },
    {
      headlines: "headlines",
      topic: "topic",
      history: "history",
      local: "local",
      summarize: "summarize",
      brief: "brief",
    } as any
  );

  g.addEdge("headlines" as any, END as any);
  g.addEdge("topic" as any, END as any);
  g.addEdge("history" as any, END as any);
  g.addEdge("local" as any, END as any);
  g.addEdge("summarize" as any, END as any);
  g.addEdge("brief" as any, END as any);

  return g.compile();
}
