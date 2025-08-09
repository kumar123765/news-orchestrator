export type Article = { title?: string; url?: string; source?: string; description?: string; publishedAt?: string };

export type OrchestratorInput = {
  query: string;
  session_id?: string;
};

export type OrchestratorState = {
  input: OrchestratorInput;
  intent?: "HEADLINES" | "TOPIC" | "BRIEF" | "HISTORY" | "LOCAL" | "SUMMARIZE" | "UNKNOWN";
  topic?: string;
  city?: string;
  dateISO?: string;
  url?: string;

  headlines?: Article[];
  topicArticles?: Article[];
  summaries?: { url: string; summary: string }[];
  brief?: string;
  historyEvents?: any;

  followupQuestion?: string;
  final?: any;
};
