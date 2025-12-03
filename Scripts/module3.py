

from collections import deque
from heapq import heappush, heappop
from typing import List, Dict, Set, Optional, Any
import json
import uuid
import datetime
import os

# Attempt to import your BN and retriever modules; if not present, planner will still work in a degraded mode.
try:
    import module1 as bir
except Exception:
    bir = None

try:
    from module2 import retrieve_with_bn
except Exception:
    retrieve_with_bn = None

# Developer-provided file for provenance (will be converted to URL by system as needed)
KNOWLEDGE_BASE_PDF = "/mnt/data/AI_project(theory).pdf"

# Define Action dataclass-like structure (simple)
class Action:
    def __init__(self, name: str, precond: List[str], effects: List[str], meta: Optional[Dict]=None):
        self.name = name
        self.precond = set(precond)
        self.effects = set(effects)
        self.meta = meta or {}
    def to_dict(self):
        return {"name": self.name, "precond": sorted(list(self.precond)),
                "effects": sorted(list(self.effects)), "meta": self.meta}
    def __repr__(self):
        return f"Action({self.name})"

# Base action library (generic pipeline actions).
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

# GraphPlan class (forward expansion)
class GraphPlan:
    def __init__(self, actions: List[Action], init_props: List[str], goal_props: List[str]):
        self.actions = actions
        self.init = set(init_props)
        self.goal = set(goal_props)

    def expand(self, max_layers: int = 10):
        state_layers = [set(self.init)]
        action_layers = []
        for i in range(max_layers):
            cur_state = state_layers[-1].copy()
            applicable = []
            for a in self.actions:
                if a.precond.issubset(state_layers[-1]):
                    applicable.append(a)
            new_props = cur_state.copy()
            for a in applicable:
                new_props |= a.effects
            action_layers.append(applicable)
            state_layers.append(new_props)
            if self.goal.issubset(new_props):
                break
            if state_layers[-1] == state_layers[-2]:
                break
        return state_layers, action_layers

    def extract_linear_plan(self, state_layers, action_layers):
        plan = []
        achieved = set(self.init)
        for layer_idx, layer_actions in enumerate(action_layers):
            for a in layer_actions:
                if any((e not in achieved) for e in a.effects):
                    plan.append(a.name)
                    achieved |= a.effects
                    if self.goal.issubset(achieved):
                        return plan
        return plan

# POP-like planner (construct ordering that satisfies dependencies)
class POPPlanner:
    def __init__(self, actions: List[Action]):
        self.actions = actions

    def plan(self, init: List[str], goal: List[str]) -> List[Action]:
        plan_actions: List[Action] = []
        achieved = set(init)
        pending = set(goal)
        while pending:
            cond = pending.pop()
            if cond in achieved:
                continue
            found = None
            for a in self.actions:
                if cond in a.effects:
                    found = a
                    break
            if found is None:
                achieved.add(cond)
                continue
            if found not in plan_actions:
                plan_actions.append(found)
            for p in found.precond:
                if p not in achieved:
                    pending.add(p)
            achieved |= found.effects

        ordering: List[Action] = []
        seen: Set[str] = set()
        def visit(action: Action):
            if action.name in seen:
                return
            for pre in action.precond:
                for a in plan_actions:
                    if pre in a.effects:
                        visit(a)
            if action.name not in seen:
                ordering.append(action)
                seen.add(action.name)
        for a in plan_actions:
            visit(a)
        return ordering

# Utility: build dynamic action set based on risk / BN features
def build_actions_for_context(risk: Optional[str], posterior: Optional[Dict[str,float]], top_intents: Optional[List[str]]) -> List[Action]:
    actions = []
    name2action = {}
    for a in BASE_ACTIONS:
        newa = Action(a.name, list(a.precond), list(a.effects), meta=dict(a.meta))
        actions.append(newa)
        name2action[newa.name] = newa

    top_prob = None
    if posterior:
        sorted_p = sorted(posterior.items(), key=lambda x: x[1], reverse=True)
        top_prob = sorted_p[0][1] if sorted_p else None

    if top_prob is not None and top_prob < 0.7:
        a = name2action.get("AskClarification")
        if a:
            a.precond = set(["IntentIdentified"])

    if risk == "high":
        esc = name2action.get("EscalateToHuman")
        if esc:
            esc.precond = set(["RiskChecked"])
            esc.effects = set(["Escalated"])
        pd = name2action.get("ProvideDisclaimer")
        if pd:
            pd.precond = set(["AnswerDraft"])
            pd.effects = set(["DisclaimerAdded"])
        finalize = name2action.get("FinalizeAnswer")
        if finalize:
            finalize.precond = set(["AnswerTranslated", "DisclaimerAdded"])
    else:
        if top_prob is not None and top_prob >= 0.8 and (not risk or risk == "low"):
            da = name2action.get("DirectAnswer")
            if da:
                da.precond = set(["InformationRetrieved"])
                da.effects = set(["AnswerDelivered"])

    if top_intents and "Compensation" in top_intents:
        for a in actions:
            if a.name == "RetrieveInformation":
                a.meta["preferred_area"] = "Compensation"
                a.meta["note"] = "Prefer docs about compensation and disbursement process"

    return actions

# High-level planner function to integrate BN and retrieval modules
def plan_for_query(query: str,
                   call_bn: bool = True,
                   call_retriever: bool = True,
                   verbose: bool = False,
                   print_output: bool = False) -> Dict[str, Any]:
    session_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()

    # BN
    features = None
    posterior = None
    top_intents = None
    bn_error = None
    if call_bn and bir:
        try:
            features = bir.extract_features(query)
            posterior = bir.posterior_intent(features)
            top_intents = [t[0] for t in sorted(posterior.items(), key=lambda x:x[1], reverse=True)[:2]]
        except Exception as e:
            bn_error = str(e)
            features = None
            posterior = None
            top_intents = None
    elif call_bn and not bir:
        bn_error = "Bayesian_Intent_Reasoning module not available."

    # Retrieval
    retrieval_output = None
    retr_error = None
    if call_retriever and retrieve_with_bn:
        try:
            retrieval_output = retrieve_with_bn(query)
        except Exception as e:
            retr_error = str(e)
            retrieval_output = None
    elif call_retriever and not retrieve_with_bn:
        retr_error = "search_retrieval.retrieve_with_bn not available."

    # prefer retrieval-provided outputs if BN didn't supply them
    if retrieval_output:
        if retrieval_output.get("posterior") and not posterior:
            posterior = retrieval_output.get("posterior")
        if retrieval_output.get("features") and not features:
            features = retrieval_output.get("features")
        if retrieval_output.get("top_intents") and not top_intents:
            top_intents = retrieval_output.get("top_intents")

    # risk heuristic
    if features:
        if features.get("Urgency") == "High" and features.get("Location") == "Flooded":
            risk = "high"
        elif features.get("Urgency") == "High":
            risk = "moderate"
        else:
            risk = "low"
    else:
        risk = "low"

    # Build actions and planners
    actions = build_actions_for_context(risk=risk, posterior=posterior, top_intents=top_intents)
    goal_props = ["AnswerDelivered"]
    init_props = ["QueryReceived"]

    gp = GraphPlan(actions, init_props, goal_props)
    state_layers, action_layers = gp.expand(max_layers=10)
    linear_plan = gp.extract_linear_plan(state_layers, action_layers)

    pop = POPPlanner(actions)
    pop_ordering_actions = pop.plan(init_props, goal_props)
    pop_ordering = [a.name for a in pop_ordering_actions]

    # Compose final actionable plan
    decision = {"rationale": [], "action_sequence": []}

    top_prob = None
    if posterior:
        sortedp = sorted(posterior.items(), key=lambda x:x[1], reverse=True)
        top_prob = sortedp[0][1]

    name_to_action = {a.name: a for a in actions}

    final_seq = []
    if top_prob is not None and top_prob >= 0.8 and risk == "low":
        if "IdentifyIntent" in name_to_action:
            final_seq.append("IdentifyIntent")
        if "RetrieveInformation" in name_to_action:
            final_seq.append("RetrieveInformation")
        if "DirectAnswer" in name_to_action:
            final_seq.append("DirectAnswer")
        else:
            if "GeneratePreliminaryAnswer" in name_to_action:
                final_seq.append("GeneratePreliminaryAnswer")
            if "HandleLanguageAdaptation" in name_to_action:
                final_seq.append("HandleLanguageAdaptation")
            if "FinalizeAnswer" in name_to_action:
                final_seq.append("FinalizeAnswer")
        decision["rationale"].append("High confidence & low risk -> prefer direct answer path")
    else:
        if "IdentifyIntent" in name_to_action:
            final_seq.append("IdentifyIntent")
        if top_prob is not None and top_prob < 0.7 and "AskClarification" in name_to_action:
            final_seq.append("AskClarification")
            decision["rationale"].append("Low BN confidence -> ask clarification")
        if "RetrieveInformation" in name_to_action:
            final_seq.append("RetrieveInformation")
        if "CheckRisk" in name_to_action:
            final_seq.append("CheckRisk")
        if risk == "high" and "EscalateToHuman" in name_to_action:
            final_seq.append("EscalateToHuman")
            decision["rationale"].append("High risk detected -> escalate to human")
        else:
            if "GeneratePreliminaryAnswer" in name_to_action:
                final_seq.append("GeneratePreliminaryAnswer")
            if risk in ("moderate","high") and "ProvideDisclaimer" in name_to_action:
                final_seq.append("ProvideDisclaimer")
            if "HandleLanguageAdaptation" in name_to_action:
                final_seq.append("HandleLanguageAdaptation")
            if "FinalizeAnswer" in name_to_action:
                final_seq.append("FinalizeAnswer")

    decision["action_sequence"] = final_seq

    actions_by_name = {a.name: a for a in actions}
    structured_steps = []
    for step_name in final_seq:
        act = actions_by_name.get(step_name)
        step_obj = {
            "step_id": step_name,
            "precond": sorted(list(act.precond)) if act else [],
            "effects": sorted(list(act.effects)) if act else [],
            "meta": act.meta if act else {}
        }
        if step_name == "RetrieveInformation":
            step_obj["retrieval"] = retrieval_output
        structured_steps.append(step_obj)

    plan_result = {
        "session_id": session_id,
        "timestamp": timestamp,
        "query": query,
        "features": features,
        "posterior": posterior,
        "top_intents": top_intents,
        "risk": risk,
        "graphplan": {
            "state_layers": [sorted(list(s)) for s in state_layers],
            "action_layers": [[a.name for a in layer] for layer in action_layers],
            "linear_plan_candidate": linear_plan
        },
        "pop_plan": pop_ordering,
        "final_plan": structured_steps,
        "decision": decision,
        "provenance": {"knowledge_pdf": KNOWLEDGE_BASE_PDF},
        "bn_error": bn_error,
        "retr_error": retr_error
    }

    if verbose:
        print("\n[VERBOSE PLANNER OUTPUT]")
        print("Query:", query)
        print("Features:", features)
        print("Posterior (top3):", sorted(posterior.items(), key=lambda x:x[1], reverse=True)[:3] if posterior else None)
        print("Risk:", risk)
        print("Final action sequence:", final_seq)
        print("Decision rationale:", decision["rationale"])
        print("GraphPlan linear candidate:", linear_plan)
        print("POP ordering:", pop_ordering)
        print("[END VERBOSE]\n")
    elif print_output:
        print(f"\nPlanner result for query: {query}")
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
        print("Decision rationale:", decision["rationale"])
        print("Provenance:", KNOWLEDGE_BASE_PDF)

    return plan_result

# ---------------------------
# Main demo runner (executes only when file run as script)
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
    for q in demo_queries:
        # print compact single-query output and collect the returned plan
        plan = plan_for_query(q, call_bn=True, call_retriever=True, verbose=False, print_output=True)
        demo_results.append(plan)

    # Save demo plans to JSON for inspection / unit tests
    out_path = "demo_plans.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(demo_results, f, indent=2, ensure_ascii=False)
        print(f"\nDemo plans written to: {os.path.abspath(out_path)}")
    except Exception as e:
        print(f"Failed to write demo plans file: {e}")

if __name__ == "__main__":
    main()
