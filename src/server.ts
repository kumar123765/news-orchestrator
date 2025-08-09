// src/server.ts
import "dotenv/config";
import express, { Request, Response, NextFunction } from "express";
import { buildGraph } from "./graph.js";

const app = express();
app.use(express.json());

// Lightweight CORS without extra deps
app.use((req: Request, res: Response, next: NextFunction) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Headers", "authorization, x-client-info, apikey, content-type");
  res.header("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  if (req.method === "OPTIONS") return res.status(200).send("ok");
  next();
});

// Build the LangGraph once at startup
const graph = buildGraph();

// Health check
app.get("/health", (_req, res) => {
  res.json({ ok: true, status: "healthy" });
});

// Orchestrate endpoint
app.post("/orchestrate", async (req: Request, res: Response) => {
  try {
    const { query, session_id } = req.body || {};
    if (!query || typeof query !== "string") {
      return res.status(400).json({ ok: false, error: "Body must include { query: string }" });
    }

    // Invoke the compiled graph. We relax TS generics here for simplicity.
    const result = await graph.invoke({
      input: { query, session_id },
    } as any);

    if (result?.followupQuestion) {
      return res.json({ ok: true, need_followup: true, question: result.followupQuestion });
    }

    return res.json({
      ok: true,
      tool: result?.intent ?? "HEADLINES",
      result: result?.final ?? null,
    });
  } catch (e: any) {
    console.error("orchestrate error:", e);
    return res.status(500).json({ ok: false, error: e?.message || String(e) });
  }
});

// Root
app.get("/", (_req, res) => {
  res.send("News Orchestrator is running. POST /orchestrate");
});

// Render provides PORT; default to 8081 locally
const port = process.env.PORT ? Number(process.env.PORT) : 8081;
app.listen(port, () => {
  console.log(`LangGraph Orchestrator on :${port}`);
});
