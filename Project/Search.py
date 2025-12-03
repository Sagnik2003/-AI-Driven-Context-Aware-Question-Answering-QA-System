
"""


Integrates:
  - Bayesian intent reasoning module: `Bayesian_Intent_Reasoning`
  - Search-based retrieval: BFS (uninformed) and A* (informed with TF-IDF heuristic)

Usage:
  python search_retrieval.py
  Type queries at the prompt. The script will call the BN module, then run retrievals
  that are biased by the BN posterior/top_intents and features (location/urgency).
"""

from collections import deque
from heapq import heappush, heappop
from typing import List, Dict, Tuple
import re
import json
import sys
import math

# External deps: sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


try:
    import Bayesian_Intent_Reasoning as bir
except Exception as e:
    # If the user named the module differently or hasn't created it yet,
    # give a clear instruction (but still allow retrieval to run standalone).
    print("Warning: could not import 'Bayesian_Intent_Reasoning'.\n"
          "Make sure you have a Python file named 'Bayesian_Intent_Reasoning.py' "
          "with functions `extract_features(query)` and `posterior_intent(features)`.\n"
          f"Import error: {e}\n"
          "Proceeding with retrieval only (will use plain text seeding).")
    bir = None

# ---------------------------
# Knowledge base (snippets) - include intent_tags and region_tags for biasing
# ---------------------------
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

# graph adjacency: topical/administrative links between snippets
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

# Build TF-IDF corpus & vectorizer
def normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower().strip())

CORPUS = [ normalize(s["title"] + ". " + s["text"]) for s in KB_SNIPPETS ]
IDS = [ s["id"] for s in KB_SNIPPETS ]
VECTORIZER = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
TFIDF_MATRIX = VECTORIZER.fit_transform(CORPUS)  # (n_docs, n_features)


# ---------------------------
# Retrieval: BFS (uninformed) - uses BN output to seed and bias
# ---------------------------
def bfs_search(query: str, posterior: Dict[str,float]=None, top_intents: List[str]=None,
               max_depth: int = 3, max_results: int = 3, region_filter: str = None) -> List[Dict]:
    q = normalize(query)
    query_tokens = set(re.findall(r'\w+', q))

    # Seed selection: prefer snippets that match top_intents
    seed_nodes = []
    if top_intents:
        for s in KB_SNIPPETS:
            if set(s.get("intent_tags", [])) & set(top_intents):
                seed_nodes.append(s["id"])
    # fallback to keyword-seed if no intent tags matched
    if not seed_nodes:
        for s in KB_SNIPPETS:
            doc_tokens = set(re.findall(r'\w+', normalize(s["title"] + " " + s["text"])))
            if query_tokens & doc_tokens:
                seed_nodes.append(s["id"])
    # ultimate fallback
    if not seed_nodes:
        seed_nodes = [IDS[0]]

    visited = set()
    queue = deque()
    for id0 in seed_nodes:
        queue.append((id0, 0))
        visited.add(id0)

    visited_nodes = []
    while queue:
        node, depth = queue.popleft()
        visited_nodes.append(node)
        if depth < max_depth:
            for nb in KB_GRAPH_ADJ.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, depth + 1))

    # compute cosine similarity between query and visited nodes
    q_vec = VECTORIZER.transform([q])
    indices = [IDS.index(nid) for nid in visited_nodes if nid in IDS]

    results = []
    if not indices:
        return results

    doc_vecs = TFIDF_MATRIX[indices]
    sims = cosine_similarity(q_vec, doc_vecs).flatten()

    for idx, sim in zip(indices, sims):
        snippet = KB_SNIPPETS[idx]
        # region filtering (optional)
        if region_filter and region_filter not in snippet.get("region_tags", []):
            continue
        # boost by posterior: snippet intent tags that match posterior intents get a weighted boost
        boost = 0.0
        if posterior:
            for it in snippet.get("intent_tags", []):
                boost += posterior.get(it, 0.0) * 0.3  # tunable boost weight
        final_score = float(sim) + boost
        r = dict(snippet)
        r["base_sim"] = float(sim)
        r["score"] = float(final_score)
        r["method"] = "BFS"
        results.append(r)

    results = sorted(results, key=lambda r: r["score"], reverse=True)[:max_results]
    return results


# ---------------------------
# Retrieval: A* (informed) - uses TF-IDF heuristic and BN posterior to bias scoring
# ---------------------------
def a_star_search(query: str, posterior: Dict[str,float]=None, top_intents: List[str]=None,
                  max_expansions: int = 200, max_results: int = 3, region_filter: str = None) -> List[Dict]:
    q = normalize(query)
    q_vec = VECTORIZER.transform([q])

    # heuristic: 1 - cosine similarity (lower = better)
    doc_sims = cosine_similarity(q_vec, TFIDF_MATRIX).flatten()
    heuristics = { IDS[i]: 1.0 - float(doc_sims[i]) for i in range(len(IDS)) }

    frontier = []
    # initialize with all nodes (g=0) to allow best-first behavior with path cost
    for nid in IDS:
        h = heuristics[nid]
        f = 0.0 + h
        heappush(frontier, (f, 0, nid, [nid]))  # (f, g, node, path)

    expanded = set()
    results = []
    expansions = 0
    seen_results = set()

    while frontier and expansions < max_expansions and len(results) < max_results:
        f, g, node, path = heappop(frontier)
        if node in expanded:
            continue
        expanded.add(node)
        expansions += 1

        snippet = KB_SNIPPETS[IDS.index(node)]
        # optionally filter by region - still allow expanding neighbors even if filtered out
        candidate_allowed = True
        if region_filter and region_filter not in snippet.get("region_tags", []):
            candidate_allowed = False

        if candidate_allowed:
            # compute boost from posterior intents
            boost = 0.0
            if posterior:
                for it in snippet.get("intent_tags", []):
                    boost += posterior.get(it, 0.0) * 0.3
            sim = float(doc_sims[IDS.index(node)])
            final_score = sim + boost
            candidate = dict(snippet)
            candidate.update({
                "base_sim": sim,
                "score": final_score,
                "method": "A*",
                "g": g,
                "h": heuristics[node],
                "path": path
            })
            if node not in seen_results:
                seen_results.add(node)
                results.append(candidate)

        # expand neighbors
        for nb in KB_GRAPH_ADJ.get(node, []):
            if nb in expanded:
                continue
            ng = g + 1
            nh = heuristics[nb]
            nf = ng + nh
            heappush(frontier, (nf, ng, nb, path + [nb]))

    # final ranking by combined score
    results = sorted(results, key=lambda r: r["score"], reverse=True)[:max_results]
    return results


# ---------------------------
# Integration: call BN, then retrieval, compare and select final snippet
# ---------------------------
def retrieve_with_bn(query: str):
    # 1) get BN outputs if available
    features = None
    posterior = None
    top_intents = None
    if bir:
        # Expect bir.extract_features and bir.posterior_intent to be available
        try:
            features = bir.extract_features(query)
            posterior = bir.posterior_intent(features)
            top_sorted = sorted(posterior.items(), key=lambda x: x[1], reverse=True)[:2]
            top_intents = [t[0] for t in top_sorted]
        except Exception as e:
            print(f"Warning: BN module call failed: {e}")
            features = None
            posterior = None
            top_intents = None

    # derive region filter from features if possible
    region_filter = None
    if features and features.get("Location"):
        # map BN location to region_tags used in KB
        loc = features["Location"]
        if loc in ("Flooded", "Rural", "Urban"):
            region_filter = loc

    # 2) run BFS and A*
    bfs_results = bfs_search(query, posterior=posterior, top_intents=top_intents, region_filter=region_filter, max_results=5)
    a_results = a_star_search(query, posterior=posterior, top_intents=top_intents, region_filter=region_filter, max_results=5)

    # 3) pretty-print both lists
    print("\nQuery:", query)
    if features:
        print("BN Features:", json.dumps(features))
    if posterior:
        print("BN Posterior (top 3):", json.dumps({k:posterior[k] for k in sorted(posterior, key=posterior.get, reverse=True)[:3]}, indent=2))
    print("\n-- BFS candidates --")
    for r in bfs_results:
        print(f"[{r['id']}] {r['title']}  (score={r['score']:.3f}, base_sim={r.get('base_sim'):.3f})  source={r['source']}")

    print("\n-- A* candidates --")
    for r in a_results:
        print(f"[{r['id']}] {r['title']}  (score={r['score']:.3f}, base_sim={r.get('base_sim'):.3f}, g={r.get('g')}, h={r.get('h'):.3f})  source={r['source']}")

    # 4) Decide final snippet with simple rule:
    # prefer A* top if its score >= BFS top score - small margin; else use BFS top.
    final = None
    chosen_by = None
    if a_results and bfs_results:
        bfs_top = bfs_results[0]
        a_top = a_results[0]
        if a_top["score"] >= bfs_top["score"] - 1e-6:
            final = a_top; chosen_by = "A*"
        else:
            final = bfs_top; chosen_by = "BFS"
    elif a_results:
        final = a_results[0]; chosen_by = "A*"
    elif bfs_results:
        final = bfs_results[0]; chosen_by = "BFS"

    if final:
        print(f"\nSelected final snippet (chosen by {chosen_by}):")
        print(f"  [{final['id']}] {final['title']}")
        print(f"  Source: {final['source']} | last_updated: {final.get('last_updated')} | source_file: {final.get('source_file')}")
        print(f"  Score: {final['score']:.3f} | base_sim: {final.get('base_sim'):.3f}")
        print("  Text:", final['text'])
    else:
        print("No candidate snippet found.")

    # Return structured result for upstream modules (planner / LLM)
    return {
        "query": query,
        "features": features,
        "posterior": posterior,
        "bfs_candidates": bfs_results,
        "a_star_candidates": a_results,
        "final_snippet": final,
        "chosen_by": chosen_by
    }


# ---------------------------
# CLI interactive loop
# ---------------------------
def main():
    print("Search Retrieval integrated with Bayesian Intent Reasoning")
    print("Type your query (or 'exit' to quit).")
    while True:
        try:
            q = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("Exiting.")
            break
        _ = retrieve_with_bn(q)


if __name__ == "__main__":
    main()
