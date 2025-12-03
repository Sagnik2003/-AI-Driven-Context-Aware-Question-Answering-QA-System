"""
module5_llm.py

LLM-based final answer generation module (Module 5).

API:
  generate_final_answer(plan_result, llm_backend='auto', print_output=False, language='en')

 - plan_result: dict produced by planning_module.plan_for_query(...) or the pipeline's plan
 - llm_backend: 'auto'|'openai'|'transformers'|'mock'
     - 'auto' tries openai -> transformers -> mock
 - print_output: if True prints prompt and final response (single-query debug)
 - language: target language code (default 'en'); only affects template instructions

Returns:
  {
    "prompt": "<constructed prompt str>",
    "model_used": "<which backend>",
    "final_text": "<generated answer>",
    "safety_checks": {"grounded": bool, "used_snippet": bool, "snippet_ids": [...]},
    "source_snippets": [...],
    "explainability": { "prompt_reasoning": "...", "safety_measures": "..." }
  }
"""

import os
import json
import textwrap
from typing import Dict, Any, List, Optional

# Try optional imports but gracefully handle missing packages
try:
    import openai
except Exception:
    openai = None

try:
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
except Exception:
    pipeline = None

# ---------------------------
# Utilities
# ---------------------------
def _safe_snippet_text(snippet_obj: Dict[str, Any]) -> str:
    """Return a clean snippet text representation for the prompt."""
    if not snippet_obj:
        return ""
    title = snippet_obj.get("title", "")
    text = snippet_obj.get("text") or snippet_obj.get("snippet") or ""
    source = snippet_obj.get("source", snippet_obj.get("source_file", "unknown"))
    return f"{title}\n{ text.strip() }\nSource: {source}"

def _collect_best_snippet(plan_result: Dict[str,Any]) -> Optional[Dict[str,Any]]:
    """
    Prefer final_snippet from retriever; else prefer top A* candidate; else BFS; else none.
    """
    if not plan_result:
        return None
    fr = plan_result.get("final_snippet")
    if fr:
        return fr
    # look into retrieval candidates
    retr = plan_result.get("a_star_candidates") or plan_result.get("bfs_candidates")
    if retr and isinstance(retr, list) and len(retr) > 0:
        return retr[0]
    return None

def _build_intent_data(plan_result: Dict[str,Any]) -> Dict[str,str]:
    """Normalize intent/feature info for prompt."""
    intent = None; confidence = None; location = None; urgency = None
    posterior = plan_result.get("posterior") or {}
    if posterior:
        top_intent, top_prob = sorted(posterior.items(), key=lambda x:x[1], reverse=True)[0]
        intent = top_intent
        confidence = f"{top_prob:.2%}"
    features = plan_result.get("features") or {}
    location = features.get("Location") if features else None
    urgency = features.get("Urgency") if features else None
    # fallback to plan_result top_intents if missing
    if not intent:
        tis = plan_result.get("top_intents") or []
        if tis:
            intent = tis[0]
    return {"intent": intent or "Unknown", "confidence": confidence or "Unknown", "location": location or "Unknown", "urgency": urgency or "Unknown"}

# ---------------------------
# Prompt constructor (safety-first)
# ---------------------------
SYSTEM_ROLE = (
    "You are a compassionate, safety-first government assistant. "
    "Your replies must be strictly grounded in the provided OFFICIAL KNOWLEDGE SNIPPET(s). "
    "If the snippet does NOT answer the user's question, acknowledge the gap and recommend "
    "safe next steps (ask clarification, refer to official channels, or escalate). "
    "Do NOT hallucinate instructions beyond the snippet. Keep tone empathetic and clear."
)

def construct_prompt_from_plan(plan_result: Dict[str,Any], language: str = "en") -> Dict[str, Any]:
    """
    Returns prompt string and metadata.
    The prompt includes:
      - system instructions
      - user query
      - BN-derived intent/context
      - Planner decision (action)
      - Official snippet(s) (explicitly marked as source of truth)
      - Response guidelines: tone, grounding, constraints
    """
    query = plan_result.get("query", "")
    intent_data = _build_intent_data(plan_result)
    action_plan = plan_result.get("decision", {}).get("action_sequence", [])
    action_summary = ", ".join(action_plan) if action_plan else "No action plan provided"

    snippet = _collect_best_snippet(plan_result)
    snippet_text = _safe_snippet_text(snippet) if snippet else ""

    # build fallback list of sources for provenance
    sources = []
    if snippet:
        sources.append(snippet.get("id") or snippet.get("source") or snippet.get("source_file"))
    # add candidate ids if available
    for ctype in ("a_star_candidates", "bfs_candidates"):
        for c in (plan_result.get(ctype) or []):
            sid = c.get("id")
            if sid and sid not in sources:
                sources.append(sid)

    # Response guideline depending on urgency
    tone = "empathetic and authoritative" if intent_data.get("urgency","Low") == "High" else "helpful and calm"

    prompt = textwrap.dedent(f"""
    ### SYSTEM INSTRUCTIONS ###
    {SYSTEM_ROLE}

    ### USER QUERY ###
    "{query}"

    ### INFERRED CONTEXT ###
    - Inferred intent: {intent_data.get('intent')} (confidence: {intent_data.get('confidence')})
    - Location: {intent_data.get('location')}
    - Urgency: {intent_data.get('urgency')}

    ### PLANNER DECISION ###
    Action plan: {action_summary}

    ### OFFICIAL KNOWLEDGE SNIPPET(S) - SOURCE OF TRUTH ###
    {snippet_text if snippet_text else '[NO SNIPPET FOUND IN KB]'}
    -- End of official snippet(s)

    ### RESPONSE GUIDELINES ###
    - You MUST only use the Official Knowledge Snippet(s) above for facts and instructions.
    - If the snippet fully answers the user's query, restate the instruction clearly and concisely.
    - If it does not, say: "I don't have official information about that exact point. Please provide more details or contact local authorities." and recommend safe next steps.
    - Keep tone: {tone}.
    - Language: {language}
    - Formatting: short paragraphs or numbered steps, emphasize safety actions first.
    """)

    meta = {
        "snippet_ids": sources,
        "intent_data": intent_data,
        "action_plan": action_plan,
        "snippet_present": bool(snippet)
    }
    return {"prompt": prompt.strip(), "meta": meta, "snippet_obj": snippet}

# ---------------------------
# LLM backends
# ---------------------------
def _generate_with_openai(prompt: str, model: str = "gpt-4", max_tokens: int = 300) -> str:
    if openai is None:
        raise RuntimeError("openai package not available")
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    openai.api_key = key
    messages = [
        {"role":"system", "content": SYSTEM_ROLE},
        {"role":"user", "content": prompt}
    ]
    resp = openai.ChatCompletion.create(model=model, messages=messages, max_tokens=max_tokens, temperature=0.0)
    text = resp["choices"][0]["message"]["content"].strip()
    return text

def _generate_with_transformers(prompt: str, model_name: str = "gpt2", max_length: int = 256) -> str:
    if pipeline is None:
        raise RuntimeError("transformers pipeline not available")
    # try to create a text-generation pipeline; if model isn't downloaded and internet disabled, this will fail.
    try:
        gen = pipeline("text-generation", model=model_name, tokenizer=model_name)
    except Exception as e:
        # try generic AutoModel (less likely to work offline)
        raise RuntimeError(f"Failed to initialize transformers pipeline: {e}")
    outputs = gen(prompt, max_length=max_length, do_sample=False)
    text = outputs[0]["generated_text"]
    # some tokenizers include prompt; strip prompt prefix if present
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text

def _generate_with_mock(prompt: str, snippet_obj: Optional[Dict[str,Any]], intent_data: Dict[str,str], action_plan: List[str]) -> str:
    """
    Template-based generator (deterministic). Always grounded in snippet_obj (if present).
    """
    if not snippet_obj or not snippet_obj.get("text"):
        # no official snippet: be transparent and recommend safe next steps
        lines = [
            f"I understand your concern about '{intent_data.get('intent')}'.",
            "I don't have an official guideline in our knowledge base that directly answers your question.",
            "Safe next steps:",
            "- Please provide any specific local details (village/district/when the event occurred).",
            "- If there is immediate danger or a medical emergency, contact local emergency services.",
            "- You may also check the official relief portal or district office for updates.",
        ]
        if "EscalateToHuman" in action_plan:
            lines.append("- This case will be escalated to a human officer for follow-up.")
        return "\n\n".join(lines)
    # else, build answer strictly from snippet text
    snippet_text = snippet_obj.get("text", "").strip()
    # brief intro that references the snippet and safety-first tone
    intro = f"I understand you asked: \"{intent_data.get('intent')}\". Based on official guidance:"
    # if action_plan indicates DirectAnswer with safety, emphasize
    advice_lines = [s.strip() for s in snippet_text.split("\n") if s.strip()]
    # Ensure we don't add extra instructions beyond snippet; we can add meta next steps (like check official channels)
    next_steps = []
    if "AskClarification" in action_plan:
        next_steps.append("If you can provide more local details (exact location, symptoms, photos), I can assist further.")
    if "EscalateToHuman" in action_plan:
        next_steps.append("This will be escalated to a human officer for follow-up; expect a response via the official channel.")
    next_steps.append("Monitor official advisories for updates.")
    out = [intro] + [""] + advice_lines + [""] + next_steps
    return "\n".join(out)

# ---------------------------
# Top-level generator
# ---------------------------
def generate_final_answer(plan_result: Dict[str,Any], llm_backend: str = "auto",
                          print_output: bool = False, language: str = "en") -> Dict[str,Any]:
    """
    Main entrypoint for Module 5.
    """
    built = construct_prompt_from_plan(plan_result, language=language)
    prompt = built["prompt"]
    meta = built["meta"]
    snippet_obj = built["snippet_obj"]
    intent_data = meta["intent_data"]
    action_plan = meta["action_plan"]

    backend_used = None
    final_text = None
    safety_grounded = bool(snippet_obj)
    snippet_ids = meta.get("snippet_ids", [])

    # decide backend
    back = llm_backend.lower() if isinstance(llm_backend, str) else "auto"
    tried = []
    errors = []
    def _try_backend(name):
        nonlocal backend_used, final_text
        try:
            if name == "openai":
                final_text = _generate_with_openai(prompt)
                backend_used = "openai"
                return True
            if name == "transformers":
                final_text = _generate_with_transformers(prompt)
                backend_used = "transformers"
                return True
            if name == "mock":
                final_text = _generate_with_mock(prompt, snippet_obj, intent_data, action_plan)
                backend_used = "mock"
                return True
        except Exception as e:
            errors.append((name, str(e)))
            return False

    if back == "auto":
        # prefer OpenAI (if possible), then transformers, then mock
        if openai is not None and (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")):
            if _try_backend("openai"):
                pass
        if backend_used is None and pipeline is not None:
            if _try_backend("transformers"):
                pass
        if backend_used is None:
            _try_backend("mock")
    else:
        # forced backend
        if back not in ("openai", "transformers", "mock"):
            raise ValueError("llm_backend must be one of 'auto'|'openai'|'transformers'|'mock'")
        _try_backend(back)

    # final safety enforcement: ensure generated text is grounded in snippet(s)
    used_snippet = False
    if final_text and snippet_obj and snippet_obj.get("text"):
        # naive check: ensure snippet sentence fragments appear or at least final_text references "official" or "use boiled" etc.
        snippet_txt = snippet_obj.get("text").lower()
        # find any short verbatim phrases from snippet in final_text
        matches = 0
        for phrase in ["boil", "bottl", "avoid drinking", "check local", "contact", "compensation", "use boiled", "seek medical"]:
            if phrase in snippet_txt and phrase in final_text.lower():
                matches += 1
        used_snippet = matches > 0
    elif final_text and not snippet_obj:
        used_snippet = False

    # If LLM produced something not grounded (mock will always be grounded), enforce a safe fallback
    if not used_snippet and snippet_obj:
        # generate a safe scaffold that quotes the snippet exactly
        quote = snippet_obj.get("text").strip()
        final_text = f"Official guidance (quoted):\n\n{quote}\n\nIf you need further help, please provide more details or contact local authorities."
        used_snippet = True

    result = {
        "prompt": prompt,
        "model_used": backend_used or "none",
        "final_text": final_text,
        "safety_checks": {"grounded": safety_grounded, "used_snippet": used_snippet, "snippet_ids": snippet_ids},
        "source_snippets": [snippet_obj] if snippet_obj else [],
        "explainability": {
            "prompt_reasoning": "Prompt requests a safety-first assistant and explicitly marks Official Knowledge Snippet(s) as the only source of facts. The action plan and BN features set tone and escalation advice.",
            "safety_measures": "If snippet missing or LLM output not grounded, the module falls back to quoting snippet or admitting lack of info and recommending escalation/clarification."
        },
        "errors": errors
    }

    if print_output:
        print("\n--- Prompt (sent to LLM) ---\n")
        print(prompt)
        print("\n--- Final text ---\n")
        print(final_text)
        if errors:
            print("\n--- Backend errors ---")
            print(errors)

    return result

# ---------------------------
# If run as script: small demo (prints outputs)
# ---------------------------
if __name__ == "__main__":
    # example usage: build a fake plan_result similar to your pipeline
    example_plan = {
        "query": "Is it safe to drink water here?",
        "features": {"QueryType":"Water", "KeywordAmbiguity":"Low", "Urgency":"High", "Location":"Flooded"},
        "posterior": {"WaterSafety": 0.95, "Health":0.03, "Compensation":0.01, "PestControl":0.01},
        "top_intents": ["WaterSafety"],
        "final_snippet": {
            "id":"s1",
            "title":"Boil water advisory",
            "text":"Please avoid drinking local water until safety is confirmed. Use boiled or bottled water as per government health advisories.",
            "source":"Health Guidelines"
        },
        "decision": {"action_sequence": ["IdentifyIntent","RetrieveInformation","DirectAnswer"]},
        "final_plan": [{"step_id":"IdentifyIntent"},{"step_id":"RetrieveInformation"},{"step_id":"DirectAnswer"}]
    }

    out = generate_final_answer(example_plan, llm_backend="auto", print_output=True)
    print("\n--- JSON Result (summary) ---")
    print(json.dumps({k:v for k,v in out.items() if k!='prompt'}, indent=2))
