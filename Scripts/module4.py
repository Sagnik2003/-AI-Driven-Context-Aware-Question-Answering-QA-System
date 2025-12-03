
from collections import deque
from heapq import heappush, heappop
from typing import List, Dict, Tuple, Optional, Any
import re
import json
import uuid
import datetime
import math
import random
import os
import pickle

# ---------------------------
# Module A: Bayesian Intent Reasoning (feature extractor + posterior)
# ---------------------------
KNOWLEDGE_BASE_PDF = "project statement data"
INTENTS = ["WaterSafety", "Compensation", "PestControl", "Health"]

P_intent = {i: 0.25 for i in INTENTS}
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

def extract_features(query: str) -> Dict[str,str]:
    q = query.lower().strip()
    if re.search(r'\b(water|drink|contaminat|boil|potable|safe to drink)\b', q):
        qtype = "Water"
    elif re.search(r'\b(compensation|money|relief|payment|claim|compensat)\b', q):
        qtype = "Compensation"
    elif re.search(r'\b(crop|pest|insect|farm|infestation|pesticide|locust)\b', q):
        qtype = "Agriculture"
    elif re.search(r'\b(fever|disease|medicine|doctor|hospital|sick|health|vomit)\b', q):
        qtype = "Health"
    else:
        qtype = "Other"

    wh_match = re.search(r'\b(what|when|how|why|where|who)\b', q)
    is_short = len(q.split()) < 7
    clarifying_phrases = bool(re.search(r'\b(in my (village|area|district)|which (scheme|office)|which (village|town))\b', q))
    ambiguity = "High" if (wh_match and is_short and not clarifying_phrases) else "Low"

    urgent = bool(re.search(r'\b(urgent|immediately|now|asap|help soon|please help|danger|risk|safe)\b', q))
    urgency = "High" if urgent else "Low"

    if re.search(r'\b(flood|flooded|rain|river|waterlogged|submerged)\b', q):
        location = "Flooded"
    elif re.search(r'\b(village|farm|rural|hamlet)\b', q):
        location = "Rural"
    elif re.search(r'\b(city|urban|town|metro|municipal)\b', q):
        location = "Urban"
    else:
        location = "Nationwide"

    return {"QueryType": qtype, "KeywordAmbiguity": ambiguity, "Urgency": urgency, "Location": location}

def posterior_intent(features: Dict[str,str]) -> Dict[str,float]:
    unnorm = {}
    for intent in INTENTS:
        p = P_intent[intent]
        p *= P_query_given_intent[intent].get(features["QueryType"], 0.01)
        p *= P_ambig_given_intent[intent].get(features["KeywordAmbiguity"], 0.01)
        p *= P_urgency_given_intent[intent].get(features["Urgency"], 0.01)
        p *= P_location_given_intent[intent].get(features["Location"], 0.01)
        unnorm[intent] = p
    s = sum(unnorm.values()) or 1e-12
    return {k: v / s for k, v in unnorm.items()}

# ---------------------------
# Module B: Search Retrieval (TF-IDF + graph)
# ---------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


KB_SNIPPETS = [
    {"id":"s1","title":"Boil water advisory",
     "text":"If water contamination is suspected, boil water for 1 minute before drinking. Use bottled water if available.",
     "source":"Health Guidelines","last_updated":"2024-03-01",
     "intent_tags":["WaterSafety","Health"], "region_tags":["Nationwide","Flooded"], "source_file":KNOWLEDGE_BASE_PDF},
    {"id":"s2","title":"Municipal tap water guidance",
     "text":"After flooding, municipal water supply may be compromised. Check local municipal advisories before using tap water.",
     "source":"Municipal Alerts","last_updated":"2024-08-12",
     "intent_tags":["WaterSafety"], "region_tags":["Urban","Nationwide","Flooded"], "source_file":KNOWLEDGE_BASE_PDF},
    {"id":"s3","title":"Flood relief compensation",
     "text":"Affected households can apply for flood relief compensation via the District Relief Portal. Required documents: ID, photos of damage.",
     "source":"Disaster Policy","last_updated":"2023-11-15",
     "intent_tags":["Compensation"], "region_tags":["Nationwide","Flooded"], "source_file":KNOWLEDGE_BASE_PDF},
    {"id":"s4","title":"Crop pest management (locusts)",
     "text":"For locust infestations, notify your agriculture officer. Follow approved pesticide application rates and safety procedures.",
     "source":"Agriculture FAQ","last_updated":"2024-06-05",
     "intent_tags":["PestControl"], "region_tags":["Rural","Nationwide"], "source_file":KNOWLEDGE_BASE_PDF},
    {"id":"s5","title":"Vector-borne disease prevention",
     "text":"After floods, watch for mosquito-borne diseases. Use bed nets, remove standing water and seek medical help for fever symptoms.",
     "source":"Health Guidelines","last_updated":"2024-07-01",
     "intent_tags":["Health","WaterSafety"], "region_tags":["Flooded","Rural","Urban"], "source_file":KNOWLEDGE_BASE_PDF},
    {"id":"s6","title":"Claims process for crop insurance",
     "text":"To claim crop insurance, submit field loss estimates and contact the insurance agent within the stipulated period.",
     "source":"Agriculture Policy","last_updated":"2024-01-20",
     "intent_tags":["Compensation","PestControl"], "region_tags":["Rural","Nationwide"], "source_file":KNOWLEDGE_BASE_PDF},
    {"id":"s7","title":"Emergency numbers",
     "text":"In case of health emergencies, call the emergency health hotline or go to the nearest health center immediately.",
     "source":"Help Desk","last_updated":"2022-12-01",
     "intent_tags":["Health"], "region_tags":["Nationwide"], "source_file":KNOWLEDGE_BASE_PDF},
    {"id":"s8","title":"Safe food and water after a disaster",
     "text":"Avoid open-sourced water. Store water in clean containers. If unsure, disinfect and boil before use.",
     "source":"Health Advisory","last_updated":"2024-05-10",
     "intent_tags":["WaterSafety","Health"], "region_tags":["Nationwide","Flooded"], "source_file":KNOWLEDGE_BASE_PDF},
    {"id":"s9","title":"How compensation is disbursed",
     "text":"Compensation is disbursed after damage verification by a field officer; processing times vary across districts.",
     "source":"Disaster Policy","last_updated":"2023-10-11",
     "intent_tags":["Compensation"], "region_tags":["Nationwide"], "source_file":KNOWLEDGE_BASE_PDF},
    {"id":"s10","title":"Pesticide safety and reporting",
     "text":"Report severe crop damage to the local Agriculture Helpline. Use protective gear while applying pesticides.",
     "source":"Agriculture FAQ","last_updated":"2024-04-04",
     "intent_tags":["PestControl"], "region_tags":["Rural","Nationwide"], "source_file":KNOWLEDGE_BASE_PDF}
]

KB_GRAPH_ADJ = {
    "s1": ["s2","s8"],
    "s2": ["s1","s8","s9"],
    "s3": ["s9","s6"],
    "s4": ["s10","s6"],
    "s5": ["s1","s8","s7"],
    "s6": ["s3","s4"],
    "s7": ["s5"],
    "s8": ["s1","s2","s5"],
    "s9": ["s3","s2"],
    "s10":["s4"]
}

def normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower().strip())

CORPUS = [ normalize(s["title"] + ". " + s["text"]) for s in KB_SNIPPETS ]
IDS = [ s["id"] for s in KB_SNIPPETS ]
VECTORIZER = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
TFIDF_MATRIX = VECTORIZER.fit_transform(CORPUS)

def bfs_search(query: str, posterior: Optional[Dict[str,float]] = None, top_intents: Optional[List[str]] = None,
               max_depth: int = 3, max_results: int = 3, region_filter: Optional[str] = None) -> List[Dict]:
    q = normalize(query)
    q_tokens = set(re.findall(r'\w+', q))
    seed_nodes = []
    if top_intents:
        for s in KB_SNIPPETS:
            if set(s.get("intent_tags", [])) & set(top_intents):
                seed_nodes.append(s["id"])
    if not seed_nodes:
        for s in KB_SNIPPETS:
            doc_tokens = set(re.findall(r'\w+', normalize(s["title"] + " " + s["text"])))
            if q_tokens & doc_tokens:
                seed_nodes.append(s["id"])
    if not seed_nodes:
        seed_nodes = [IDS[0]]

    visited = set(); queue = deque()
    for id0 in seed_nodes:
        queue.append((id0, 0)); visited.add(id0)
    visited_nodes = []
    while queue:
        node, depth = queue.popleft()
        visited_nodes.append(node)
        if depth < max_depth:
            for nb in KB_GRAPH_ADJ.get(node, []):
                if nb not in visited:
                    visited.add(nb); queue.append((nb, depth+1))

    q_vec = VECTORIZER.transform([q])
    indices = [IDS.index(nid) for nid in visited_nodes if nid in IDS]
    results = []
    if not indices:
        return results
    doc_vecs = TFIDF_MATRIX[indices]
    sims = cosine_similarity(q_vec, doc_vecs).flatten()
    for idx, sim in zip(indices, sims):
        snippet = KB_SNIPPETS[idx]
        if region_filter and region_filter not in snippet.get("region_tags", []):
            continue
        boost = 0.0
        if posterior:
            for it in snippet.get("intent_tags", []):
                boost += posterior.get(it, 0.0) * 0.3
        final_score = float(sim) + boost
        r = dict(snippet); r.update({"base_sim": float(sim), "score": final_score, "method":"BFS"})
        results.append(r)
    results = sorted(results, key=lambda r: r["score"], reverse=True)[:max_results]
    return results

def a_star_search(query: str, posterior: Optional[Dict[str,float]] = None, top_intents: Optional[List[str]] = None,
                  max_expansions: int = 200, max_results: int = 3, region_filter: Optional[str] = None) -> List[Dict]:
    q = normalize(query); q_vec = VECTORIZER.transform([q])
    doc_sims = cosine_similarity(q_vec, TFIDF_MATRIX).flatten()
    heuristics = { IDS[i]: 1.0 - float(doc_sims[i]) for i in range(len(IDS)) }
    frontier = []
    for nid in IDS:
        h = heuristics[nid]; f = 0.0 + h
        heappush(frontier, (f, 0, nid, [nid]))
    expanded = set(); results = []; expansions = 0; seen_results = set()
    while frontier and expansions < max_expansions and len(results) < max_results:
        f, g, node, path = heappop(frontier)
        if node in expanded: continue
        expanded.add(node); expansions += 1
        snippet = KB_SNIPPETS[IDS.index(node)]
        candidate_allowed = True
        if region_filter and region_filter not in snippet.get("region_tags", []):
            candidate_allowed = False
        if candidate_allowed:
            boost = 0.0
            if posterior:
                for it in snippet.get("intent_tags", []):
                    boost += posterior.get(it, 0.0) * 0.3
            sim = float(doc_sims[IDS.index(node)])
            final_score = sim + boost
            candidate = dict(snippet)
            candidate.update({"base_sim": sim, "score": final_score, "method":"A*", "g": g, "h": heuristics[node], "path": path})
            if node not in seen_results:
                seen_results.add(node); results.append(candidate)
        for nb in KB_GRAPH_ADJ.get(node, []):
            if nb in expanded: continue
            ng = g + 1; nh = heuristics[nb]; nf = ng + nh
            heappush(frontier, (nf, ng, nb, path + [nb]))
    results = sorted(results, key=lambda r: r["score"], reverse=True)[:max_results]
    return results

def retrieve_with_bn(query: str, verbose: bool = False, print_output: bool = False, max_results: int = 5) -> Dict:
    features = extract_features(query)
    posterior = posterior_intent(features)
    top_sorted = sorted(posterior.items(), key=lambda x: x[1], reverse=True)[:2]
    top_intents = [t[0] for t in top_sorted]
    region_filter = None
    if features.get("Location") in ("Flooded","Rural","Urban"):
        region_filter = features["Location"]
    bfs_results = bfs_search(query, posterior=posterior, top_intents=top_intents, region_filter=region_filter, max_results=max_results)
    a_results = a_star_search(query, posterior=posterior, top_intents=top_intents, region_filter=region_filter, max_results=max_results)
    final = None; chosen_by = None
    if a_results and bfs_results:
        bfs_top = bfs_results[0]; a_top = a_results[0]
        if a_top["score"] >= bfs_top["score"] - 1e-6:
            final = a_top; chosen_by = "A*"
        else:
            final = bfs_top; chosen_by = "BFS"
    elif a_results:
        final = a_results[0]; chosen_by = "A*"
    elif bfs_results:
        final = bfs_results[0]; chosen_by = "BFS"
    result = {
        "query": query,
        "features": features,
        "posterior": posterior,
        "top_intents": top_intents,
        "bfs_candidates": bfs_results,
        "a_star_candidates": a_results,
        "final_snippet": final,
        "chosen_by": chosen_by,
        "provenance": {"knowledge_pdf": KNOWLEDGE_BASE_PDF}
    }
    if verbose:
        print("\n[Retriever verbose]")
        print("Features:", features)
        print("Posterior:", posterior)
        print("-- BFS --")
        for r in bfs_results: print(r["id"], r["title"], r["score"])
        print("-- A* --")
        for r in a_results: print(r["id"], r["title"], r["score"])
        if final: print("Selected:", final["id"], final["title"])
    elif print_output:
        print("\nQuery:", query)
        print("Top intents:", top_intents)
        print("Chosen snippet:", final["id"] if final else None)
    return result

# ---------------------------
# Module C: Planner
# ---------------------------
class Action:
    def __init__(self, name: str, precond: List[str], effects: List[str], meta: Optional[Dict]=None):
        self.name = name
        self.precond = set(precond)
        self.effects = set(effects)
        self.meta = meta or {}
    def __repr__(self):
        return f"Action({self.name})"

BASE_ACTIONS = [
    Action("IdentifyIntent", ["QueryReceived"], ["IntentIdentified"], {"module":"BN"}),
    Action("RetrieveInformation", ["IntentIdentified"], ["InformationRetrieved"], {"module":"Retriever"}),
    Action("CheckRisk", ["InformationRetrieved"], ["RiskChecked"], {"module":"RuleEngine"}),
    Action("AskClarification", ["IntentIdentified"], ["Clarified"], {"module":"Dialog"}),
    Action("GeneratePreliminaryAnswer", ["RiskChecked"], ["AnswerDraft"], {"module":"AnswerGen"}),
    Action("HandleLanguageAdaptation", ["AnswerDraft"], ["AnswerTranslated"], {"module":"NLP"}),
    Action("FinalizeAnswer", ["AnswerTranslated"], ["AnswerDelivered"], {"module":"AnswerGen"}),
    Action("ProvideDisclaimer", ["AnswerDraft"], ["DisclaimerAdded"], {"module":"Safety"}),
    Action("EscalateToHuman", ["RiskChecked"], ["Escalated"], {"module":"Ops"}),
    Action("DirectAnswer", ["InformationRetrieved"], ["AnswerDelivered"], {"module":"AnswerGen"}),
    Action("DirectAnswerWithDisclaimer", ["InformationRetrieved"], ["AnswerDelivered"], {"module":"AnswerGen"}),
]

class GraphPlan:
    def __init__(self, actions: List[Action], init_props: List[str], goal_props: List[str]):
        self.actions = actions; self.init = set(init_props); self.goal = set(goal_props)
    def expand(self, max_layers: int = 10):
        state_layers = [set(self.init)]; action_layers = []
        for i in range(max_layers):
            cur = state_layers[-1].copy()
            applicable = []
            for a in self.actions:
                if a.precond.issubset(state_layers[-1]):
                    applicable.append(a)
            new_props = cur.copy()
            for a in applicable:
                new_props |= a.effects
            action_layers.append(applicable)
            state_layers.append(new_props)
            if self.goal.issubset(new_props): break
            if state_layers[-1] == state_layers[-2]: break
        return state_layers, action_layers
    def extract_linear_plan(self, state_layers, action_layers):
        plan = []; achieved = set(self.init)
        for layer_actions in action_layers:
            for a in layer_actions:
                if any(e not in achieved for e in a.effects):
                    plan.append(a.name)
                    achieved |= a.effects
                    if self.goal.issubset(achieved): return plan
        return plan

class POPPlanner:
    def __init__(self, actions: List[Action]): self.actions = actions
    def plan(self, init: List[str], goal: List[str]) -> List[Action]:
        plan_actions: List[Action] = []; achieved = set(init); pending = set(goal)
        while pending:
            cond = pending.pop()
            if cond in achieved: continue
            found = None
            for a in self.actions:
                if cond in a.effects:
                    found = a; break
            if found is None:
                achieved.add(cond); continue
            if found not in plan_actions: plan_actions.append(found)
            for p in found.precond:
                if p not in achieved: pending.add(p)
            achieved |= found.effects
        ordering: List[Action] = []; seen = set()
        def visit(action: Action):
            if action.name in seen: return
            for pre in action.precond:
                for a in plan_actions:
                    if pre in a.effects:
                        visit(a)
            if action.name not in seen:
                ordering.append(action); seen.add(action.name)
        for a in plan_actions: visit(a)
        return ordering

def build_actions_for_context(risk: Optional[str], posterior: Optional[Dict[str,float]], top_intents: Optional[List[str]]) -> List[Action]:
    actions = []; name2action = {}
    for a in BASE_ACTIONS:
        newa = Action(a.name, list(a.precond), list(a.effects), meta=dict(a.meta))
        actions.append(newa); name2action[newa.name] = newa
    top_prob = None
    if posterior:
        sorted_p = sorted(posterior.items(), key=lambda x:x[1], reverse=True)
        top_prob = sorted_p[0][1] if sorted_p else None
    if top_prob is not None and top_prob < 0.7:
        a = name2action.get("AskClarification")
        if a: a.precond = set(["IntentIdentified"])
    if risk == "high":
        esc = name2action.get("EscalateToHuman")
        if esc:
            esc.precond = set(["RiskChecked"]); esc.effects = set(["Escalated"])
        pd = name2action.get("ProvideDisclaimer")
        if pd:
            pd.precond = set(["AnswerDraft"]); pd.effects = set(["DisclaimerAdded"])
        finalize = name2action.get("FinalizeAnswer")
        if finalize: finalize.precond = set(["AnswerTranslated", "DisclaimerAdded"])
    else:
        if top_prob is not None and top_prob >= 0.8 and (not risk or risk == "low"):
            da = name2action.get("DirectAnswer")
            if da: da.precond = set(["InformationRetrieved"]); da.effects = set(["AnswerDelivered"])
    if top_intents and "Compensation" in top_intents:
        for a in actions:
            if a.name == "RetrieveInformation":
                a.meta["preferred_area"] = "Compensation"; a.meta["note"] = "Prefer compensation docs"
    return actions

def plan_for_query(query: str, call_bn: bool = True, call_retriever: bool = True,
                   verbose: bool = False, print_output: bool = False) -> Dict[str,Any]:
    session_id = str(uuid.uuid4()); timestamp = datetime.datetime.now().isoformat()
    features = None; posterior = None; top_intents = None; bn_error = None
    if call_bn:
        try:
            features = extract_features(query); posterior = posterior_intent(features)
            top_intents = [t[0] for t in sorted(posterior.items(), key=lambda x:x[1], reverse=True)[:2]]
        except Exception as e:
            bn_error = str(e); features = None; posterior = None; top_intents = None
    retrieval_output = None; retr_error = None
    if call_retriever:
        try:
            retrieval_output = retrieve_with_bn(query)
        except Exception as e:
            retr_error = str(e); retrieval_output = None
    if retrieval_output:
        if retrieval_output.get("posterior") and not posterior: posterior = retrieval_output.get("posterior")
        if retrieval_output.get("features") and not features: features = retrieval_output.get("features")
        if retrieval_output.get("top_intents") and not top_intents: top_intents = retrieval_output.get("top_intents")
    if features:
        if features.get("Urgency") == "High" and features.get("Location") == "Flooded":
            risk = "high"
        elif features.get("Urgency") == "High":
            risk = "moderate"
        else:
            risk = "low"
    else:
        risk = "low"
    actions = build_actions_for_context(risk=risk, posterior=posterior, top_intents=top_intents)
    gp = GraphPlan(actions, ["QueryReceived"], ["AnswerDelivered"])
    state_layers, action_layers = gp.expand(max_layers=10)
    linear_plan = gp.extract_linear_plan(state_layers, action_layers)
    pop = POPPlanner(actions)
    pop_ordering_actions = pop.plan(["QueryReceived"], ["AnswerDelivered"])
    pop_ordering = [a.name for a in pop_ordering_actions]
    decision = {"rationale": [], "action_sequence": []}
    top_prob = None
    if posterior:
        sortedp = sorted(posterior.items(), key=lambda x:x[1], reverse=True)
        top_prob = sortedp[0][1]
    name_to_action = {a.name: a for a in actions}
    final_seq = []
    if top_prob is not None and top_prob >= 0.8 and risk == "low":
        if "IdentifyIntent" in name_to_action: final_seq.append("IdentifyIntent")
        if "RetrieveInformation" in name_to_action: final_seq.append("RetrieveInformation")
        if "DirectAnswer" in name_to_action: final_seq.append("DirectAnswer")
        else:
            if "GeneratePreliminaryAnswer" in name_to_action: final_seq.append("GeneratePreliminaryAnswer")
            if "HandleLanguageAdaptation" in name_to_action: final_seq.append("HandleLanguageAdaptation")
            if "FinalizeAnswer" in name_to_action: final_seq.append("FinalizeAnswer")
        decision["rationale"].append("High confidence & low risk -> direct")
    else:
        if "IdentifyIntent" in name_to_action: final_seq.append("IdentifyIntent")
        if top_prob is not None and top_prob < 0.7 and "AskClarification" in name_to_action:
            final_seq.append("AskClarification"); decision["rationale"].append("Low BN confidence -> clarify")
        if "RetrieveInformation" in name_to_action: final_seq.append("RetrieveInformation")
        if "CheckRisk" in name_to_action: final_seq.append("CheckRisk")
        if risk == "high" and "EscalateToHuman" in name_to_action:
            final_seq.append("EscalateToHuman"); decision["rationale"].append("High risk -> escalate")
        else:
            if "GeneratePreliminaryAnswer" in name_to_action: final_seq.append("GeneratePreliminaryAnswer")
            if risk in ("moderate","high") and "ProvideDisclaimer" in name_to_action: final_seq.append("ProvideDisclaimer")
            if "HandleLanguageAdaptation" in name_to_action: final_seq.append("HandleLanguageAdaptation")
            if "FinalizeAnswer" in name_to_action: final_seq.append("FinalizeAnswer")
    decision["action_sequence"] = final_seq
    actions_by_name = {a.name: a for a in actions}
    structured_steps = []
    for step_name in final_seq:
        act = actions_by_name.get(step_name)
        step_obj = {"step_id": step_name, "precond": sorted(list(act.precond)) if act else [], "effects": sorted(list(act.effects)) if act else [], "meta": act.meta if act else {}}
        if step_name == "RetrieveInformation":
            step_obj["retrieval"] = retrieval_output
        structured_steps.append(step_obj)
    plan_result = {
        "session_id": session_id, "timestamp": timestamp, "query": query, "features": features,
        "posterior": posterior, "top_intents": top_intents, "risk": risk,
        "graphplan": {"state_layers":[sorted(list(s)) for s in state_layers], "action_layers":[[a.name for a in layer] for layer in action_layers], "linear_plan_candidate": linear_plan},
        "pop_plan": pop_ordering, "final_plan": structured_steps, "decision": decision, "provenance": {"knowledge_pdf": KNOWLEDGE_BASE_PDF}, "bn_error": bn_error, "retr_error": retr_error
    }
    if verbose:
        print("\n[Planner verbose]")
        print("Query:", query)
        print("Features:", features)
        print("Posterior:", posterior)
        print("Risk:", risk)
        print("Final action sequence:", final_seq)
    elif print_output:
        print(f"\nPlanner for: {query}")
        print("Risk:", risk)
        if posterior:
            print("Top intents:", sorted(posterior.items(), key=lambda x:x[1], reverse=True)[:3])
        print("Planned steps:")
        for s in structured_steps:
            print(" -", s["step_id"], "| precond:", s["precond"], "| effects:", s["effects"])
            if s.get("retrieval"):
                bf = s["retrieval"].get("bfs_candidates") if s["retrieval"] else None
                a = s["retrieval"].get("a_star_candidates") if s["retrieval"] else None
                print("   retrieval: bfs:", [r["id"] for r in (bf or [])], "a*:", [r["id"] for r in (a or [])])
    return plan_result

# ---------------------------
# Module D: RL Policy (Q-learning)
# ---------------------------
INTENT_TO_IDX = {name: i for i, name in enumerate(INTENTS)}
CONF_BINS = ["Low","Med","High"]
LOCATION_BINS = ["Flooded","Rural","Urban","Nationwide"]
RISK_BINS = ["Low","High"]
ACTION_NAMES = {0:"DirectAnswer",1:"AskClarification",2:"EscalateToHuman",3:"GenerateExplanation"}
ACTION_COUNT = len(ACTION_NAMES)

def build_states_and_adj():
    states = []
    for intent in range(len(INTENTS)):
        for conf in range(len(CONF_BINS)):
            for loc in range(len(LOCATION_BINS)):
                for risk in range(len(RISK_BINS)):
                    states.append((intent, conf, loc, risk))
    adj = {s: [] for s in states}
    for s1 in states:
        for s2 in states:
            if s1 == s2: continue
            same_intent = (s1[0] == s2[0])
            same_location = (s1[2] == s2[2])
            conf_close = abs(s1[1] - s2[1]) == 1
            risk_diff = (s1[3] != s2[3])
            if same_intent and same_location and (conf_close or risk_diff):
                adj[s1].append(s2)
    return states, adj

def compute_reward(state: Tuple[int,int,int,int], action: int) -> float:
    intent, conf, loc, risk = state
    if action == 0 and conf == 2 and risk == 0: return 10.0
    if action == 1 and conf == 0: return 6.0
    if action == 0 and conf == 0 and risk == 1: return -8.0
    if risk == 1 and action != 1: penalty = -10.0
    else: penalty = 0.0
    if action == 2 and risk == 1: return 8.0
    if action == 3 and conf == 1: return 4.0 + penalty
    return penalty

def train_q_table(eta: float = 0.1, gamma: float = 0.9, max_epochs: int = 5000, tolerance: float = 1e-5, random_seed: Optional[int] = 42):
    if random_seed is not None: random.seed(random_seed)
    states, adj = build_states_and_adj()
    Q = {(s,a):0.0 for s in states for a in range(ACTION_COUNT)}
    for epoch in range(max_epochs):
        max_change = 0.0
        for s in states:
            neighbors = adj[s]
            for a in range(ACTION_COUNT):
                old_q = Q[(s,a)]
                r = compute_reward(s,a)
                best_neighbor_q = 0.0
                if neighbors:
                    for s_next in neighbors:
                        best_for_this_neighbor = None
                        for a_next in range(ACTION_COUNT):
                            q_val = Q[(s_next, a_next)]
                            if best_for_this_neighbor is None or q_val > best_for_this_neighbor:
                                best_for_this_neighbor = q_val
                        if best_for_this_neighbor is not None and best_for_this_neighbor > best_neighbor_q:
                            best_neighbor_q = best_for_this_neighbor
                target = r + gamma * best_neighbor_q
                new_q = (1 - eta) * old_q + eta * target
                Q[(s,a)] = new_q
                change = abs(new_q - old_q)
                if change > max_change: max_change = change
        if max_change < tolerance: break
    return Q

def best_action_for_state_with_Q(state: Tuple[int,int,int,int], Q: Dict) -> Tuple[int,float]:
    best_a = None; best_q = None
    for a in range(ACTION_COUNT):
        q_val = Q.get((state,a), float("-inf"))
        if best_q is None or q_val > best_q:
            best_q = q_val; best_a = a
    return best_a, best_q

def discretize_state_from_features(features: Dict[str,Any], posterior: Optional[Dict[str,float]] = None) -> Tuple[int,int,int,int]:
    intent_idx = 0
    if posterior:
        top_intent = max(posterior.items(), key=lambda kv: kv[1])[0]
        intent_idx = INTENT_TO_IDX.get(top_intent, 0)
    else:
        qtype = features.get("QueryType","Other")
        if qtype == "Water": intent_idx = INTENT_TO_IDX["WaterSafety"]
        elif qtype == "Compensation": intent_idx = INTENT_TO_IDX["Compensation"]
        elif qtype == "Agriculture": intent_idx = INTENT_TO_IDX["PestControl"]
        elif qtype == "Health": intent_idx = INTENT_TO_IDX["Health"]
        else: intent_idx = 0
    conf_bin = 0
    if posterior:
        top_prob = max(posterior.values())
        if top_prob >= 0.8: conf_bin = 2
        elif top_prob >= 0.5: conf_bin = 1
        else: conf_bin = 0
    else:
        amb = features.get("KeywordAmbiguity","Low")
        conf_bin = 0 if amb == "High" else 1
    loc = features.get("Location","Nationwide")
    if loc not in LOCATION_BINS:
        if "flood" in str(loc).lower(): loc_idx = 0
        elif "village" in str(loc).lower() or "rural" in str(loc).lower(): loc_idx = 1
        elif "city" in str(loc).lower() or "urban" in str(loc).lower(): loc_idx = 2
        else: loc_idx = 3
    else:
        loc_idx = LOCATION_BINS.index(loc)
    risk_bin = 0
    if features.get("Urgency") == "High" and features.get("Location") == "Flooded": risk_bin = 1
    else: risk_bin = 0
    return (intent_idx, conf_bin, loc_idx, risk_bin)

class RLPolicy:
    def __init__(self, Q_table: Optional[Dict] = None):
        self.Q = Q_table; self._trained_default = False
    def ensure_trained(self):
        if self.Q is None:
            self.Q = train_q_table()
            self._trained_default = True
    def select_action_for_state(self, state: Tuple[int,int,int,int]) -> Dict[str,Any]:
        self.ensure_trained()
        a, qv = best_action_for_state_with_Q(state, self.Q)
        qvals = [self.Q.get((state, aa), 0.0) for aa in range(ACTION_COUNT)]
        max_q = max(qvals) if qvals else 0.0
        exps = [math.exp(q - max_q) for q in qvals]
        ssum = sum(exps) if sum(exps) != 0 else 1.0
        probs = [e/ssum for e in exps]
        return {"rl_action_id": a, "rl_action_name": ACTION_NAMES[a], "q_value": qv, "q_values": qvals, "action_probs": probs}
    def select_action_from_features(self, features: Dict[str,Any], posterior: Optional[Dict[str,float]] = None) -> Dict[str,Any]:
        state = discretize_state_from_features(features, posterior)
        return self.select_action_for_state(state)
    def decide_action_for_plan_result(self, plan_result: Dict[str,Any]) -> Dict[str,Any]:
        features = plan_result.get("features") or {}
        posterior = plan_result.get("posterior")
        rl_out = self.select_action_from_features(features, posterior)
        rl_a = rl_out["rl_action_id"]
        mapping = {0:"DirectAnswer", 1:"AskClarification", 2:"EscalateToHuman", 3:"GeneratePreliminaryAnswer"}
        mapped_step = mapping.get(rl_a, None)
        decision_obj = {"session_id": plan_result.get("session_id"), "query": plan_result.get("query"), "rl_decision": rl_out, "mapped_planner_step": mapped_step}
        return decision_obj

_global_policy: Optional[RLPolicy] = None
def get_default_rl_policy() -> RLPolicy:
    global _global_policy
    if _global_policy is None:
        _global_policy = RLPolicy()
    return _global_policy

# ---------------------------
# Demo main
# ---------------------------
def main():
    demo_queries = [
        "Is it safe to drink water here?",
        "When will compensation come after the flood?",
        "What to do for crop pest attack?",
        "There is fever spreading in our village, what should we do?",
        "Crops infested by locusts, need immediate pesticide advice"
    ]
    demo_results = []
    rl = get_default_rl_policy()
    for q in demo_queries:
        plan = plan_for_query(q, call_bn=True, call_retriever=True, verbose=False, print_output=True)
        # get RL decision and integrate (example: if RL wants AskClarification, insert it)
        rl_dec = rl.decide_action_for_plan_result(plan)
        mapped = rl_dec["mapped_planner_step"]
        if mapped == "AskClarification":
            if not any(s["step_id"] == "AskClarification" for s in plan["final_plan"]):
                plan["final_plan"].insert(1, {"step_id":"AskClarification","precond":["IntentIdentified"],"effects":["Clarified"],"meta":{"source":"rl_policy"}})
        # attach rl_decision for telemetry
        plan["rl_decision"] = rl_dec
        demo_results.append(plan)
    out_path = "demo_plans.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(demo_results, f, indent=2, ensure_ascii=False)
        print(f"\nDemo plans written to: {os.path.abspath(out_path)}")
    except Exception as e:
        print("Failed to write demo plans file:", e)

if __name__ == "__main__":
    main()
