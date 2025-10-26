from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

import os
from llama_index.core.settings import Settings
from llama_index.llms.anthropic import Anthropic as LIAnthropic
from llama_index.embeddings.openai import OpenAIEmbedding

# Claude for LlamaIndex synthesis
Settings.llm = LIAnthropic(
    model="claude-sonnet-4-5-20250929",
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

# OpenAI for embeddings
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# --- LlamaIndex for recipe Q&A (RAG)
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

# --- LiveKit Agents + plugins
from livekit.agents import Agent, AgentSession, AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.plugins import deepgram, anthropic, silero


# ---------------------------------------------------------------------------
# Data & Index setup (RAG over your recipe/docs)
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).parent
DATA_DIR = THIS_DIR / "data"
PERSIST_DIR = THIS_DIR / "query-engine-storage"

if not PERSIST_DIR.exists():
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Global async query engine for tools
_query_engine = index.as_query_engine(use_async=True)

# ---------------------------------------------------------------------------
# Simple recipe guide state machine
# ---------------------------------------------------------------------------
def _extract_steps_from_files(data_dir: Path) -> List[str]:
    """
    Naive step extractor:
      - If a line starts with a number/period, treat as a step ("1. Preheat...")
      - Otherwise split on common keywords or headings.
    You can replace with a proper parser later.
    """
    steps: List[str] = []

    for p in sorted(data_dir.glob("**/*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".md", ".txt"}:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")

        # Prefer enumerated steps like "1. Do X"
        enumerated = re.findall(r"^\s*\d+[\.)]\s+(.*)$", text, flags=re.MULTILINE)
        if enumerated:
            steps.extend([s.strip() for s in enumerated if s.strip()])
            continue

        # Otherwise split on common cue words (very heuristic)
        chunks = re.split(r"(?:^|\n)(?:Step\s*\d+:|Instructions?:|Method:)", text, flags=re.IGNORECASE)
        if len(chunks) > 1:
            for ch in chunks:
                ch = ch.strip()
                if not ch:
                    continue
                # Take lines as atomic steps if short; otherwise split on sentences
                sentences = re.split(r"(?<=[.!?])\s+", ch)
                for s in sentences:
                    s = s.strip("-• \n\t")
                    if len(s) > 0:
                        steps.append(s)
        else:
            # Fallback: split into sentences
            sentences = re.split(r"(?<=[.!?])\s+", text)
            steps.extend([s.strip("-• \n\t") for s in sentences if len(s.strip()) > 0])

    # Deduplicate and keep only actionable-ish lines
    uniq = []
    seen = set()
    for s in steps:
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        # Filter over-long non-instruction blobs
        if len(s) <= 500:
            uniq.append(s)

    # If nothing was found, add a placeholder so the agent doesn't get stuck
    if not uniq:
        uniq = ["I couldn't find structured steps; ask me anything about the recipe."]
    return uniq


class RecipeGuide:
    def __init__(self, steps: List[str]):
        self.steps = steps
        self.i = 0  # current step index

    def current(self) -> Tuple[int, str]:
        return (self.i + 1, self.steps[self.i])

    def next(self) -> Tuple[int, str]:
        if self.i < len(self.steps) - 1:
            self.i += 1
        return self.current()

    def prev(self) -> Tuple[int, str]:
        if self.i > 0:
            self.i -= 1
        return self.current()

    def jump(self, n: int) -> Tuple[int, str]:
        n = max(1, min(n, len(self.steps)))
        self.i = n - 1
        return self.current()

    def total(self) -> int:
        return len(self.steps)


RECIPE_GUIDE = RecipeGuide(_extract_steps_from_files(DATA_DIR))


# ---------------------------------------------------------------------------
# Utilities: time & unit conversion (minimal, no extra deps)
# ---------------------------------------------------------------------------
def parse_time_to_seconds(text: str) -> Optional[int]:
    """
    Parse '3 min', '2 minutes 30 seconds', '90s', '1.5 hours' → seconds
    """
    text = text.lower().strip()
    # 90s / 30sec / 30s
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(s|sec|secs|second|seconds)", text)
    if m:
        return int(float(m.group(1)))

    # 3m or 3 min
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(m|min|mins|minute|minutes)", text)
    if m:
        return int(float(m.group(1)) * 60)

    # 1.5h or hours
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours)", text)
    if m:
        return int(float(m.group(1)) * 3600)

    # compound, e.g., "2 minutes 30 seconds"
    m = re.findall(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)", text)
    if m:
        total = 0
        for num, unit in m:
            num = float(num)
            unit = unit.lower()
            if unit.startswith("h"):
                total += int(num * 3600)
            elif unit.startswith("m"):
                total += int(num * 60)
            elif unit.startswith("s"):
                total += int(num)
        return total if total > 0 else None

    # pure number → seconds
    if text.isdigit():
        return int(text)
    return None


def convert_units(amount: float, src: str, dst: str) -> Optional[float]:
    """
    Very small conversions among volume/mass: tsp, tbsp, cup, ml, l, g, kg, oz, lb.
    NOTE: mass↔volume needs density; we avoid cross-converting those.
    """
    src = src.lower().strip()
    dst = dst.lower().strip()

    vol = {
        "tsp": 5.0,
        "teaspoon": 5.0,
        "tbsp": 15.0,
        "tablespoon": 15.0,
        "cup": 240.0,
        "ml": 1.0,
        "milliliter": 1.0,
        "l": 1000.0,
        "liter": 1000.0,
    }
    mass = {
        "g": 1.0,
        "gram": 1.0,
        "kg": 1000.0,
        "kilogram": 1000.0,
        "oz": 28.3495,
        "ounce": 28.3495,
        "lb": 453.592,
        "pound": 453.592,
    }

    if src in vol and dst in vol:
        return amount * (vol[src] / vol[dst])
    if src in mass and dst in mass:
        return amount * (mass[src] / mass[dst])
    # otherwise require density → return None
    return None


# ---------------------------------------------------------------------------
# Tools (callable by the LLM)
# ---------------------------------------------------------------------------
@llm.function_tool
async def recipe_current() -> str:
    """Return the current step number and text."""
    step_no, text = RECIPE_GUIDE.current()
    return f"Step {step_no}/{RECIPE_GUIDE.total()}: {text}"


@llm.function_tool
async def recipe_next() -> str:
    """Advance to the next step and return it."""
    step_no, text = RECIPE_GUIDE.next()
    return f"Step {step_no}/{RECIPE_GUIDE.total()}: {text}"


@llm.function_tool
async def recipe_prev() -> str:
    """Go back one step and return it."""
    step_no, text = RECIPE_GUIDE.prev()
    return f"Step {step_no}/{RECIPE_GUIDE.total()}: {text}"


@llm.function_tool
async def recipe_jump(step_number: int) -> str:
    """Jump to a specific step number (1-based) and return it."""
    step_no, text = RECIPE_GUIDE.jump(step_number)
    return f"Step {step_no}/{RECIPE_GUIDE.total()}: {text}"


@llm.function_tool
async def recipe_query(question: str) -> str:
    """
    Answer a question grounded in the recipe/docs, e.g.:
      - 'How long should I stir?'
      - 'What temperature is the oven?'
    """
    res = await _query_engine.aquery(question)
    return str(res)


@llm.function_tool
async def set_timer(duration: str) -> str:
    """
    Speak a timer reminder after the given duration (e.g., '3 minutes', '90s').
    Returns the parsed duration in seconds if successful.
    """
    secs = parse_time_to_seconds(duration)
    if not secs or secs <= 0:
        return "I couldn't parse that time."
    # Schedule a background task that speaks when done.
    async def _alarm():
        await asyncio.sleep(secs)
        # We'll say the reminder from within the session via a global registry set at runtime
        if SESSION_SINGLETON is not None:
            await SESSION_SINGLETON.say("Timer done. Check your pan.")
    asyncio.create_task(_alarm())
    return f"Okay, timer set for {secs} seconds."


@llm.function_tool
async def convert(amount: float, src_unit: str, dst_unit: str) -> str:
    """
    Convert among common kitchen units (volume or mass only).
    Example: 2, 'tbsp', 'tsp' → 6
    """
    out = convert_units(amount, src_unit, dst_unit)
    if out is None:
        return "Sorry, I can only convert within volume or within mass without density."
    # keep clean decimal representation
    if abs(out - round(out, 3)) < 1e-9:
        out = round(out, 3)
    return f"{amount} {src_unit} ≈ {out} {dst_unit}"


@llm.function_tool
async def substitutions(ingredient: str) -> str:
    """
    Offer common substitutions (very lightweight, extend as needed).
    """
    table = {
        "buttermilk": "Mix 1 cup milk with 1 tbsp lemon juice or vinegar, rest 5–10 min.",
        "egg": "For baking: 1 tbsp ground flax + 3 tbsp water (rest 5 min) per egg.",
        "brown sugar": "1 cup white sugar + 1 tbsp molasses.",
        "baking powder": "¼ tsp baking soda + ½ tsp cream of tartar per 1 tsp powder.",
        "sour cream": "Plain Greek yogurt 1:1.",
        "heavy cream": "¾ cup milk + ⅓ cup butter (melted) ≈ 1 cup heavy cream (not for whipping).",
    }
    key = ingredient.strip().lower()
    if key in table:
        return f"Substitute for {ingredient}: {table[key]}"
    return f"I don't have a preset sub for {ingredient}. Ask me about an alternative based on context."


@llm.function_tool
async def food_safety_temp(protein: str) -> str:
    """
    Safe internal temperatures (USDA-style quick refs).
    """
    prot = protein.lower().strip()
    temps = {
        "chicken": "165°F (74°C) — whole or ground.",
        "turkey": "165°F (74°C) — whole or ground.",
        "beef": "145°F (63°C) whole cuts (rest 3 min), 160°F (71°C) ground.",
        "pork": "145°F (63°C) whole cuts (rest 3 min), 160°F (71°C) ground.",
        "fish": "145°F (63°C) or until flesh flakes and is opaque.",
        "eggs": "Cook until yolk & white are firm; custards to 160°F (71°C).",
    }
    return temps.get(prot, "I don't have that one; typical safe range is 145–165°F depending on the protein.")


# Will be set by entrypoint so tool timers can speak
SESSION_SINGLETON: Optional[AgentSession] = None


# ---------------------------------------------------------------------------
# Agent entrypoint
# ---------------------------------------------------------------------------
async def entrypoint(ctx: JobContext):
    global SESSION_SINGLETON

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # --- Claude LLM via Anthropic plugin
    llm_model = anthropic.LLM(model="claude-sonnet-4-5-20250929")

    agent = Agent(
        instructions=(
            "You are a friendly, concise, hands-free Cooking Companion. "
            "Guide the user step-by-step through the recipe. "
            "Prefer short sentences. Avoid punctuation that is hard to pronounce. "
            "When unsure about exact measurements or durations, consult the recipe_query tool. "
            "Offer safety reminders where helpful. "
            "Encourage the user to say 'next', 'repeat', or 'set a timer for X minutes'."
        ),
        vad=silero.VAD.load(),            # VAD
        stt=deepgram.STT(),               # Speech-to-Text
        llm=llm_model,                    # Claude
        tts=deepgram.TTS(),              # Swap to another TTS anytime
        tools=[
            recipe_current,
            recipe_next,
            recipe_prev,
            recipe_jump,
            recipe_query,
            set_timer,
            convert,
            substitutions,
            food_safety_temp,
        ],
    )

    session = AgentSession()
    SESSION_SINGLETON = session  # so timers can speak back
    await session.start(agent=agent, room=ctx.room)

    # Opening line tailored by whether we parsed steps
    if RECIPE_GUIDE.total() > 1:
        await session.say(
            f"Hi! I loaded a recipe with {RECIPE_GUIDE.total()} steps. "
            f"Say 'start' or 'what's step one?'.",
            allow_interruptions=True,
        )
    else:
        await session.say(
            "Hi! I loaded your recipe. Ask me anything or say 'what's next?'.",
            allow_interruptions=True,
        )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
