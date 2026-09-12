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
You are ReelMind, an expert multimodal knowledge extraction agent.
Analyze the video and audio of this Reel thoroughly.
Extract grounded knowledge where every fact, measurement, or instruction is linked to its exact video timestamp.
Do NOT fabricate confidence scores. If confident, provide a score (e.g. 0.95); otherwise leave confidence as null.
Classify the category (recipe, tech, workout, idea, travel, finance, other).
Identify 2 to 4 keyframe timestamps for high-res photo snapshots.
Return strict JSON matching the schema.
"""


class ReelAnalyzer:
    def __init__(self, api_key: str, embedding_model: str = "models/gemini-embedding-001", embedding_dim: int = 768):
        self.client = genai.Client(api_key=api_key)
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
                model="gemini-2.0-flash",
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
        return list(response.embedding.values[:self.embedding_dim])
