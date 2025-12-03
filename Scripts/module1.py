import re
import pandas as pd
import uuid
import datetime
import json

# --- Prior and conditional tables (unchanged, but grouped here) ---
intents = ["WaterSafety", "Compensation", "PestControl", "Health"]
P_intent = {
    "WaterSafety": 0.25,
    "Compensation": 0.25,
    "PestControl": 0.25,
    "Health": 0.25
}

P_query_given_intent = {
    "WaterSafety":  {"Water": 0.8, "Compensation": 0.05, "Agriculture": 0.05, "Health": 0.05, "Other": 0.05},
    "Compensation": {"Water": 0.05, "Compensation": 0.8, "Agriculture": 0.05, "Health": 0.05, "Other": 0.05},
    "PestControl":  {"Water": 0.05, "Compensation": 0.05, "Agriculture": 0.8, "Health": 0.05, "Other": 0.05},
    "Health":       {"Water": 0.05, "Compensation": 0.05, "Agriculture": 0.05, "Health": 0.8, "Other": 0.05},
}

P_ambig_given_intent = {
    "WaterSafety":  {"Low": 0.8, "High": 0.2},
    "Compensation": {"Low": 0.6, "High": 0.4},
    "PestControl":  {"Low": 0.7, "High": 0.3},
    "Health":       {"Low": 0.65, "High": 0.35},
}

P_urgency_given_intent = {
    "WaterSafety":  {"High": 0.8, "Low": 0.2},
    "Compensation": {"High": 0.6, "Low": 0.4},
    "PestControl":  {"High": 0.55, "Low": 0.45},
    "Health":       {"High": 0.7, "Low": 0.3},
}

P_location_given_intent = {
    "WaterSafety":  {"Flooded": 0.5, "Rural": 0.2, "Urban": 0.2, "Nationwide": 0.1},
    "Compensation": {"Flooded": 0.1, "Rural": 0.3, "Urban": 0.2, "Nationwide": 0.4},
    "PestControl":  {"Flooded": 0.05, "Rural": 0.65, "Urban": 0.15, "Nationwide": 0.15},
    "Health":       {"Flooded": 0.2, "Rural": 0.25, "Urban": 0.35, "Nationwide": 0.2},
}

# --- Improved feature extractor ---
def extract_features(query):
    q = query.lower().strip()

    # Query type - slightly expanded keyword list
    if re.search(r'\b(water|drink|contaminat|boil|potable|safe to drink)\b', q):
        qtype = "Water"
    elif re.search(r'\b(compensation|money|relief|payment|claim|compensat)\b', q):
        qtype = "Compensation"
    elif re.search(r'\b(crop|pest|insect|farm|infestation|pesticide)\b', q):
        qtype = "Agriculture"
    elif re.search(r'\b(fever|disease|medicine|doctor|hospital|sick|health)\b', q):
        qtype = "Health"
    else:
        qtype = "Other"

    # Ambiguity: treat short wh-questions as high ambiguity; explicit "where exactly" reduces ambiguity
    wh_match = re.search(r'\b(what|when|how|why|where|who)\b', q)
    is_short = len(q.split()) < 7
    # but if user included clarifying phrase, mark low ambiguity
    clarifying_phrases = bool(re.search(r'\b(in my (village|area|district)|which (scheme|office)|which (village|town))\b', q))
    if wh_match and is_short and not clarifying_phrases:
        ambiguity = "High"
    else:
        ambiguity = "Low"

    # Urgency: expand tokens + heuristic if question contains 'please help' or exclamation
    urgent = bool(re.search(r'\b(urgent|immediately|now|asap|help soon|please help|danger|risk|safe)\b', q))
    urgency = "High" if urgent else "Low"

    # Location detection (looks for explicit local cues)
    if re.search(r'\b(flood|flooded|rain|river|waterlogged|submerged)\b', q):
        location = "Flooded"
    elif re.search(r'\b(village|farm|rural|hamlet)\b', q):
        location = "Rural"
    elif re.search(r'\b(city|urban|town|metro|municipal)\b', q):
        location = "Urban"
    else:
        location = "Nationwide"

    return {
        "QueryType": qtype,
        "KeywordAmbiguity": ambiguity,
        "Urgency": urgency,
        "Location": location
    }

# --- Posterior calculation (same logic, just explicit) ---
def posterior_intent(observed):
    unnorm = {}
    for intent in intents:
        p = P_intent[intent]
        p *= P_query_given_intent[intent].get(observed["QueryType"], 0.01)
        p *= P_ambig_given_intent[intent].get(observed["KeywordAmbiguity"], 0.01)
        p *= P_urgency_given_intent[intent].get(observed["Urgency"], 0.01)
        p *= P_location_given_intent[intent].get(observed["Location"], 0.01)
        unnorm[intent] = p
    total = sum(unnorm.values()) or 1e-12
    posterior = {k: v / total for k, v in unnorm.items()}
    return posterior

# --- Planner (Module C) stub: generate small plan based on top intent(s) and risk ---
def generate_plan(session_id, features, posterior, top_k=2):
    # risk heuristic: High urgency + Flooded + WaterSafety or Health -> moderate/high risk
    top_intents = sorted(posterior.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_list = [t[0] for t in top_intents]
    risk = "low"
    if features['Urgency'] == "High" and features['Location'] == "Flooded" and ("WaterSafety" in top_list or "Health" in top_list):
        risk = "high"
    elif features['Urgency'] == "High":
        risk = "moderate"
    plan = [
        {"step_id": "s1", "action": "RetrieveSnippets", "args": {"intents": top_list}},
        {"step_id": "s2", "action": "AssessRisk", "args": {"features": features}},
        {"step_id": "s3", "action": "PrepareDraftAnswer", "args": {"templates":"intent-aware"}}
    ]
    return {"session_id": session_id, "plan": plan, "risk": risk, "top_intents": top_list}

# --- RL action selector (Module D) stub: deterministic policy from thresholds ---
def select_action(posterior, risk):
    # policy thresholds (changed per workflow guidance):
    # >= 0.80 -> DirectAnswer
    # 0.50 - 0.80 -> AskClarification
    # < 0.50 -> EscalateToHuman
    sorted_intents = sorted(posterior.items(), key=lambda x: x[1], reverse=True)
    top_intent, top_prob = sorted_intents[0]
    second_prob = sorted_intents[1][1]

    if top_prob >= 0.80 and risk == "low":
        action = "DirectAnswer"
    elif top_prob >= 0.80 and risk in ("moderate", "high"):
        # even if confident, be conservative if risk high
        action = "DirectAnswerWithDisclaimer"
    elif 0.50 <= top_prob < 0.80:
        action = "AskClarification"
    else:
        action = "EscalateToHuman"

    # include decision rationale for logging / explainability
    rationale = {
        "top_intent": top_intent,
        "top_prob": top_prob,
        "second_prob": second_prob,
        "risk": risk
    }
    return action, rationale

# --- LLM response generator stub (Module E) ---
def generate_response(action, top_intent, features):
    # templates / safe defaults
    responses = {
        "WaterSafety":  "Please avoid drinking local water until safety is confirmed. Use boiled or bottled water as per government health advisories.",
        "Compensation": "Compensation for affected families is being processed through the District Office. Check the official relief portal for your area.",
        "PestControl":  "You can report pest outbreaks to your local agriculture officer. Approved pesticides and guidance are available on the Agri Portal.",
        "Health":       "Please contact your nearest health center for medical assistance. Follow official advisories on sanitation and disease prevention."
    }

    # Clarification questions depending on intent
    clarifications = {
        "WaterSafety": "Do you mean tap/piped water at your home, or open water (river/pond)? Which village/district?",
        "Compensation": "Are you asking about flood relief, crop insurance, or another compensation scheme? Which district?",
        "PestControl": "Which crop is affected and how large is the affected area?",
        "Health": "How many people are affected and what symptoms are you seeing? Is anyone in immediate danger?"
    }

    if action == "DirectAnswer":
        answer_text = responses.get(top_intent, "I can help — can you provide more details?")
        metadata = {"disclaimer": None}
    elif action == "DirectAnswerWithDisclaimer":
        answer_text = responses.get(top_intent, "I can help — can you provide more details?")
        metadata = {"disclaimer": "This is informational. In emergencies, contact local authorities / health services."}
    elif action == "AskClarification":
        answer_text = clarifications.get(top_intent, "Could you provide more details?")
        metadata = {"disclaimer": None}
    else:  # EscalateToHuman
        answer_text = "Your request requires human review. We will escalate this to the support team. Meanwhile, please share any local identifiers (village/district)."
        metadata = {"disclaimer": "Escalation initiated."}

    return {"answer_text": answer_text, "metadata": metadata}

# --- Logging stub (to persist what the system saw / decided) ---
def log_interaction(record):
    # in production you'd persist to DB / message queue. Here we print a compact JSON for traceability.
    print("\n--- LOG (store this in telemetry) ---")
    print(json.dumps(record, indent=2, default=str))
    print("--- END LOG ---\n")

# --- Put everything together: answer_query returns a structured object ---
def answer_query(query, session_id=None):
    if session_id is None:
        session_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()

    features = extract_features(query)
    posterior = posterior_intent(features)
    sorted_post = sorted(posterior.items(), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(sorted_post, columns=["Intent", "Probability"])

    # Planner
    planner_output = generate_plan(session_id, features, posterior, top_k=2)

    # Policy / action selection
    action, rationale = select_action(posterior, planner_output["risk"])

    top_intent = rationale["top_intent"]
    response_obj = generate_response(action, top_intent, features)

    # Build final structured response for the frontend / next modules
    result = {
        "session_id": session_id,
        "timestamp": timestamp,
        "query": query,
        "features": features,
        "posterior": posterior,
        "top_intents": planner_output["top_intents"],
        "plan": planner_output["plan"],
        "risk": planner_output["risk"],
        "selected_action": action,
        "decision_rationale": rationale,
        "response": response_obj
    }

    # Logging for telemetry / RL
    log_record = {
        "session_id": session_id,
        "timestamp": timestamp,
        "features": features,
        "posterior": posterior,
        "action": action,
        "rationale": rationale,
        "response_text": response_obj["answer_text"],
        "metadata": response_obj["metadata"]
    }
    log_interaction(log_record)

    # Also print a human-friendly summary (optional)
    print("\nQuery:", query)
    print("\nExtracted Features:", features)
    print("\nPosterior Probabilities:")
    print(df.to_string(index=False))
    print(f"\nSelected Action: {action}")
    print("\nSystem Response:", response_obj["answer_text"])
    if response_obj["metadata"].get("disclaimer"):
        print("\nDisclaimer:", response_obj["metadata"]["disclaimer"])
    print("\nExplanation: Routing based on posterior probability and risk heuristic (see log).")

    return result

# --- Example queries (same as yours) ---
queries = [
    "Is it safe to drink water here?",
    "When will compensation come after the flood?",
    "What to do for crop pest attack?",
    "There is fever spreading in our village, what should we do?"
]

if __name__ == "__main__":
    for q in queries:
        answer_query(q)
        print("\n" + "-"*80)
