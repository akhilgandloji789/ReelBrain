import numpy as np
from typing import Any
from modules.storage import ReelDatabase
from modules.analyzer import ReelAnalyzer


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


class SearchEngine:
    def __init__(self, db: ReelDatabase, analyzer: ReelAnalyzer):
        self.db = db
        self.analyzer = analyzer

    def hybrid_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        fts_matches = self.db.search_fts(query, limit=top_k)
        seen_ids = {m["id"] for m in fts_matches}
        ranked_results = list(fts_matches)

        try:
            query_embedding = self.analyzer.generate_embedding(query)
            candidates = self.db.get_all_embeddings()
            scored: list[tuple[float, dict[str, Any]]] = []
            for c in candidates:
                sim = cosine_similarity(query_embedding, c["embedding"])
                scored.append((sim, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            
            for sim, candidate in scored[:top_k]:
                if candidate["id"] not in seen_ids:
                    seen_ids.add(candidate["id"])
                    ranked_results.append(candidate)
        except Exception:
            pass

        return ranked_results[:top_k]

    def answer_conversational_query(self, query: str) -> str:
        matches = self.hybrid_search(query, top_k=3)
        if not matches:
            return "🔍 I couldn't find any saved reels matching your question."

        context_lines = []
        for idx, m in enumerate(matches, 1):
            ents = self.db.get_entities(m["id"])
            ent_details = []
            for e in ents[:5]:
                start_sec = int(e.get("start_ts", 0.0))
                mins = start_sec // 60
                secs = start_sec % 60
                ts_str = f"{mins:02d}:{secs:02d}"
                ent_details.append(f"{e['text']} (⏱️ {ts_str})")

            ent_summary = "; ".join(ent_details)
            intent_str = f"Why saved: {m['user_intent']}\n" if m.get("user_intent") else ""
            context_lines.append(
                f"[{idx}] Title: {m.get('title')}\n"
                f"{intent_str}"
                f"Category: {m.get('category')}\n"
                f"Evidence: {ent_summary}\n"
                f"Source: {m.get('url')}"
            )
        context_str = "\n\n".join(context_lines)

        prompt = f"""You are ReelMind, a personal AI second brain assistant.
The user is asking: "{query}"

Here is the retrieved grounded evidence from their saved reels:
{context_str}

Answer the user's question directly and concisely in Telegram HTML format.
Rules:
1. Ground your answer ONLY in the evidence provided above.
2. For every fact, recipe step, or code tip, cite the video timestamp (e.g. ⏱️ 00:31) so the user can scrub directly to that exact second in the video.
3. Always include the source Reel URL at the end of your answer.
4. If the retrieved evidence does not contain the answer, state clearly that it was not found in their saved notes.
"""
        response = self.analyzer.client.models.generate_content(
            model=self.analyzer.model,
            contents=prompt
        )
        return response.text
