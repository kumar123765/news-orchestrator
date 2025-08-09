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

const State = Annotation.Root({
  // request payload
  input: Annotation<any>(), // { query: string; session_id?: string }

  // routing slots
  intent: Annotation<OrchestratorState["intent"] | undefined>(),
  topic: Annotation<string | undefined>(),
  city: Annotation<string | undefined>(),
  dateISO: Annotation<string | undefined>(),
  url: Annotation<string | undefined>(),
  max: Annotation<number | undefined>(),

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

/** Router: Structured Outputs (required+nullable) + max extraction */
async function router(state: GraphState) {
  const schema = z.object({
    intent: z.enum(["HEADLINES", "TOPIC", "BRIEF", "HISTORY", "LOCAL", "SUMMARIZE", "UNKNOWN"]),
    topic: z.string().nullable(),
    city: z.string().nullable(),
    dateISO: z.string().nullable(),
    url: z.string().nullable(),
    max: z.number().nullable(),
  });

  const sys =
    "Classify the user's query for a news assistant (India-focused). " +
    "Return JSON with keys: intent, topic, city, dateISO, url, max. " +
    "If the user asks for N headlines/results (e.g., 'top 8 headlines', 'show 5 stories'), set max to that integer.";

  const prompt = [
    { role: "system", content: sys },
    { role: "user", content: state.input?.query ?? "" },
  ];

  const out = await llm.withStructuredOutput(schema).invoke(prompt as any);
  return {
    intent: out.intent ?? "HEADLINES",
    topic: out.topic ?? undefined,
    city: out.city ?? undefined,
    dateISO: out.dateISO ?? undefined,
    url: out.url ?? undefined,
    max: out.max ?? undefined,
  };
}

/** Actions */
async function doHeadlines(state: GraphState) {
  const resp = await tools.topHeadlines({ max: state.max ?? undefined });
  if (!resp.ok) return { final: { error: resp.error } };
  return { headlines: resp.result?.articles, final: resp.result };
}

async function doTopic(state: GraphState) {
  if (!state.topic) return { followupQuestion: "Which topic?" };
  const resp = await tools.topicNews(state.topic, state.max ?? 10);
  if (!resp.ok) return { final: { error: resp.error } };
  return { topicArticles: resp.result?.articles, final: resp.result };
}

async function doHistory(state: GraphState) {
  const resp = await tools.onThisDay(state.dateISO);
  if (!resp.ok) return { final: { error: resp.error } };
  return { historyEvents: resp.result, final: resp.result };
}

async function doLocal(state: GraphState) {
  const resp = await tools.aroundYou(state.city, state.max ?? 8, state.input?.session_id);
  if (!resp.ok) return { final: { error: resp.error } };
  if (resp.result?.need_followup) return { followupQuestion: resp.result.question };
  return { final: resp.result };
}

async function doSummarize(state: GraphState) {
  if (!state.url) return { followupQuestion: "Please share the article link (URL)." };
  const resp = await tools.summarizeUrl(state.url);
  if (!resp.ok) return { final: { error: resp.error } };
  return { final: resp.result };
}

async function doBrief(state: GraphState) {
  if (!state.topic) return { followupQuestion: "Brief on which topic?" };

  const news = await tools.topicNews(state.topic, state.max ?? 8);
  if (!news.ok) return { final: { error: news.error } };

  const arts = (news.result?.articles ?? []).slice(0, 3);

  // Parallelize summarization for speed
  const summaries = await Promise.all(
    arts.map(async (a: any) => {
      if (!a?.url) return null;
      const s = await tools.summarizeUrl(a.url);
      if (!s.ok) return null;
      return { url: a.url, summary: s.result?.summary ?? "" };
    })
  ).then((x) => x.filter(Boolean) as { url: string; summary: string }[]);

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

  // node IDs (distinct from channel names)
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
