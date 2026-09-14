import json
import time
from pathlib import Path
from typing import Literal
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

CategoryType = Literal["recipe", "tech", "workout", "idea", "travel", "finance", "other"]


class EntityItem(BaseModel):
    entity_type: str = Field(description="Type: ingredient, instruction, code_snippet, tech_fact, exercise, idea_point")
    text: str = Field(description="Exact fact, instruction, measurement or code")
    start_ts: float = Field(description="Start time in seconds where this appears or is spoken")
    end_ts: float | None = Field(default=None, description="End time in seconds")
    confidence: float | None = Field(default=None, description="Model-reported confidence (nullable, no fake default)")


class ReelAnalysisOutput(BaseModel):
    title: str = Field(description="Short, crisp, descriptive title")
    category: CategoryType = Field(description="Category of the reel content")
    tldr: str = Field(description="1-2 sentence executive summary")
    entities: list[EntityItem] = Field(description="Extracted grounded claims, ingredients, instructions or code")
    keyframe_timestamps: list[float] = Field(description="2-4 timestamps for key visual evidence snapshots")


ANALYSIS_PROMPT = """
You are ReelBrain (ReelMind), an elite multimodal AI knowledge extraction agent.
Your mission is to perform meticulous, hyper-accurate video and audio analysis of Instagram Reels to build an authoritative personal second brain.

1. AUDIO & SPOKEN CONTENT EXTRACTION:
- Transcribe and listen intently to all spoken speech, narration, voiceovers, dialogue, and audio cues.
- Capture exact verbal statements: specific numbers, temperatures, times, ingredient quantities, book titles, author names, coding libraries, tool names, exercise form warnings.
- Do not gloss over or skip spoken details even if they are not shown as on-screen text.

2. VISUAL OCR & CONTEXT:
- Read all on-screen text, subtitles, code editors, command terminals, nutrition facts, step-by-step checklists, charts, and slide bullet points.
- Correlate on-screen visual demonstrations with spoken instructions.

3. STRICT DOMAIN CATEGORIZATION:
Classify the reel into exactly ONE of the following categories:
- 'recipe': Food recipes, cooking methods, baking, drinks/cocktails, meal prep, restaurant dishes, grocery lists.
- 'tech': Software development, coding, AI models/agents, dev tools, hardware, Linux/CLI, apps, tech architecture.
- 'workout': Fitness routines, gym exercises, form cues, reps/sets, mobility, stretching, bodybuilding, athletic training.
- 'idea': Books, philosophy, psychology, mental models, productivity, study habits, mindset, career/life wisdom.
- 'travel': Destinations, itineraries, travel hacks, flights, packing, city/hotel recommendations.
- 'finance': Stock market, investing, budgeting, real estate, taxes, personal finance, business models.
- 'other': Humor, entertainment, fashion, art, music, gaming, or content not fitting the above.

4. ACCURACY & EVIDENCE GROUNDING:
- Title: Highly specific and informative (e.g., "Authentic Roman Carbonara with Guanciale" or "FastAPI Microservices with Docker & Redis"). NEVER use generic titles like "Cooking Video" or "Tech Tips".
- TL;DR: 2 to 3 concise, information-dense sentences summarizing the core takeaways, tools/ingredients used, and main outcome.
- Entities: Extract 4 to 15 granular evidence points (ingredients, instructions, code snippets, exercise steps, or key insights). Link EVERY entity to its exact start_ts in seconds where it is spoken or shown.
- Keyframe Timestamps: Select 2 to 4 distinct timestamps representing the most informative visual evidence (e.g., finished dish, code snippet, form demonstration, summary slide).
- Confidence: Provide a truthful score (e.g. 0.95) if verified in video/audio; otherwise leave as null. Never hallucinate.

Return strict JSON matching the schema.
"""


class ReelAnalyzer:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.6-flash",
        embedding_model: str = "models/gemini-embedding-001",
        embedding_dim: int = 768
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5), reraise=True)
    def analyze_video(self, video_path: Path | str) -> ReelAnalysisOutput:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        uploaded = self.client.files.upload(file=video_path)
        while getattr(uploaded, "state", None) == "PROCESSING":
            time.sleep(1)
            uploaded = self.client.files.get(name=uploaded.name)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[uploaded, "Index and extract structured knowledge with exact evidence timestamps."],
                config=types.GenerateContentConfig(
                    system_instruction=ANALYSIS_PROMPT,
                    response_mime_type="application/json",
                    response_schema=ReelAnalysisOutput,
                    temperature=0.2,
                )
            )
            data = json.loads(response.text)
            return ReelAnalysisOutput.model_validate(data)
        finally:
            try:
                self.client.files.delete(name=uploaded.name)
            except Exception:
                pass

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5), reraise=True)
    def generate_embedding(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model=self.embedding_model,
            contents=text
        )
        if getattr(response, "embeddings", None):
            values = response.embeddings[0].values
        elif getattr(response, "embedding", None):
            values = response.embedding.values
        else:
            values = getattr(response, "values", [])
        return list(values[:self.embedding_dim])
