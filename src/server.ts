import "dotenv/config";
import express from "express";
import bodyParser from "body-parser";
import { buildGraph } from "./graph.js";
import type { OrchestratorState } from "./state.js";

const app = express();
app.use(bodyParser.json());
const graph = buildGraph();

app.post("/orchestrate", async (req, res) => {
  try {
    const { query, session_id } = req.body || {};
    if (!query) return res.status(400).json({ ok: false, error: "query is required" });

    const result = await graph.invoke({ input: { query, session_id } } as OrchestratorState);

    if (result.followupQuestion) {
      return res.json({ ok: true, need_followup: true, question: result.followupQuestion });
    }
    return res.json({ ok: true, tool: result.intent, result: result.final });
  } catch (e: any) {
    return res.status(500).json({ ok: false, error: e.message || String(e) });
  }
});

const port = process.env.PORT ? Number(process.env.PORT) : 8081;
app.listen(port, () => console.log(`LangGraph Orchestrator on :${port}`));
