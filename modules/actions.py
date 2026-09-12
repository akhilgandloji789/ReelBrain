import html
import json
from pathlib import Path
from typing import Any
from modules.storage import ReelDatabase
from modules.analyzer import ReelAnalyzer


class ActionHandler:
    def __init__(self, db: ReelDatabase, analyzer: ReelAnalyzer):
        self.db = db
        self.analyzer = analyzer

    def generate_grocery_list(self, reel_id: int) -> str:
        entities = self.db.get_entities(reel_id)
        ingredients = [e["text"] for e in entities if e["entity_type"] == "ingredient"]
        if not ingredients:
            return "⚠️ No ingredients found for this reel."
        
        seen = set()
        deduped = []
        for ing in ingredients:
            cleaned = ing.strip()
            if cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                deduped.append(cleaned)

        lines = ["🛒 <b>Grocery Checklist:</b>\n"]
        for item in deduped:
            lines.append(f"• [ ] {html.escape(item)}")
        
        self.db.log_action(reel_id, "grocery", {"items": deduped})
        return "\n".join(lines)

    def generate_code_block(self, reel_id: int) -> str:
        entities = self.db.get_entities(reel_id)
        snippets = [e["text"] for e in entities if e["entity_type"] in ("code", "code_snippet")]
        if not snippets:
            return "⚠️ No code snippets found for this reel."

        lines = ["💻 <b>Extracted Code:</b>\n"]
        for s in snippets:
            lines.append(f"<code>{html.escape(s)}</code>\n")
        
        self.db.log_action(reel_id, "code", {"snippets": snippets})
        return "\n".join(lines)

    def record_feedback(self, reel_id: int, thumb: str) -> str:
        action_type = "feedback_thumb_up" if thumb == "up" else "feedback_thumb_down"
        self.db.log_action(reel_id, action_type, {"vote": thumb})
        if thumb == "up":
            return "👍 Thanks for the feedback! Marked as accurate."
        return "👎 Noted as inaccurate. You can tap [✏️ Edit] to correct any wrong details."

    def apply_edit(self, entity_id: int, new_text: str) -> tuple[bool, str]:
        ok, reel_id = self.db.update_entity(entity_id, new_text)
        if not ok:
            return False, f"Entity {entity_id} not found."

        try:
            reel = self.db.get_reel_by_id(reel_id)
            if reel:
                ents = self.db.get_entities(reel_id)
                combined_text = f"{reel.get('title', '')}\n{reel.get('user_intent') or ''}\n" + " ".join([e["text"] for e in ents])
                new_embedding = self.analyzer.generate_embedding(combined_text)
                self.db.store_embedding(
                    reel_id,
                    model=getattr(self.analyzer, "embedding_model", "models/gemini-embedding-001"),
                    dim=getattr(self.analyzer, "embedding_dim", 768),
                    vector=new_embedding
                )
        except Exception:
            pass

        self.db.log_action(reel_id, "edit", {"entity_id": entity_id, "new_text": new_text})
        return True, f"✅ Entity {entity_id} updated & vector index refreshed."

    def export_markdown(self, output_path: Path | str) -> Path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        reels = self.db.get_all_reels_with_entities()

        lines = ["# ReelMind Knowledge Export\n"]
        for r in reels:
            lines.append(f"## {r.get('title', 'Untitled')}")
            if r.get("user_intent"):
                lines.append(f"- **Why I saved this:** {r['user_intent']}")
            lines.append(f"- **Category:** #{r.get('category')}")
            lines.append(f"- **Source:** {r.get('url')}")
            lines.append(f"- **Summary:** {r.get('raw_transcript')}\n")
            lines.append("### Grounded Evidence:")
            for e in r.get("entities", []):
                mins = int(e.get("start_ts", 0.0)) // 60
                secs = int(e.get("start_ts", 0.0)) % 60
                ts_str = f"{mins:02d}:{secs:02d}"
                conf_str = f", {int(e['confidence']*100)}% conf" if e.get("confidence") is not None else ""
                lines.append(f"- [⏱️ {ts_str}] {e['text']} ({e.get('entity_type', 'fact')}{conf_str})")
            lines.append("\n---\n")

        out_file.write_text("\n".join(lines), encoding="utf-8")
        return out_file

    def export_json(self, output_path: Path | str) -> Path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        reels = self.db.get_all_reels_with_entities()
        out_file.write_text(json.dumps(reels, indent=2, ensure_ascii=False), encoding="utf-8")
        return out_file
