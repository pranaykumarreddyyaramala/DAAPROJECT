from flask import Flask, render_template, request, url_for, redirect
import json
import itertools

app = Flask(__name__)

# ---- Load recipes (use your recipes.json file) ----
with open("recipes.json", "r", encoding="utf-8") as f:
    recipes = json.load(f)

# Create an ingredient -> recipes index (graph-like)
ingredient_index = {}
for r in recipes:
    for ing in r["ingredients"]:
        ingredient_index.setdefault(ing.lower().strip(), set()).add(r["id"])

# ---- Substitution dictionary (common substitutions) ----
# You can extend this mapping as needed.
SUBSTITUTIONS = {
    "butter": ["margarine", "olive oil", "ghee"],
    "cream": ["yogurt", "milk + butter (small)"],
    "egg": ["banana (mashed)", "applesauce", "flaxseed + water"],
    "milk": ["soy milk", "almond milk", "water + milk powder"],
    "cheese": ["nutritional yeast", "cream cheese (if ok)"],
    "garlic": ["garlic powder", "asafoetida (hing)"],
    "onion": ["shallot", "green onion"],
    "salt": ["soy sauce (in some recipes)", "salt substitute"],
    "rice": ["quinoa (different texture)"],
    "paneer": ["tofu"],
    "chicken": ["tofu (vegetarian swap)", "seitan"],
    "tomato": ["tomato puree", "passata"],
    "sugar": ["honey", "maple syrup"],
}

# ---- Helper functions / algorithms ----

def normalize_ings(ing_list):
    return [i.lower().strip() for i in ing_list]

def match_score(recipe, available_set):
    """Score = number of matching ingredients / total required (0..1),
    and also return list of missing ingredients."""
    req = set(normalize_ings(recipe["ingredients"]))
    present = req.intersection(available_set)
    missing = sorted(list(req - available_set))
    score = len(present) / max(len(req), 1)
    return score, missing, sorted(list(present))

def build_bipartite_graph(available_ings):
    """Return mapping: ingredient -> set(recipe_ids); recipe_id -> set(ingredients)"""
    avail_norm = set(normalize_ings(available_ings))
    graph = {"ingredient_to_recipes": {}, "recipe_to_ingredients": {}}
    for ing in avail_norm:
        graph["ingredient_to_recipes"][ing] = ingredient_index.get(ing, set())
    for r in recipes:
        # only include recipes that share at least one available ingredient
        req = set(normalize_ings(r["ingredients"]))
        if req & avail_norm:
            graph["recipe_to_ingredients"][r["id"]] = req
    return graph

def greedy_recommendation(available_ings, k=3):
    """Greedy: pick recipes that cover the most *new* available ingredients each step.
       Returns list of recipe dicts (up to k) and coverage info."""
    avail_norm = set(normalize_ings(available_ings))
    remaining = set(avail_norm)
    chosen = []
    candidates = [r for r in recipes]
    # Score function: number of recipe ingredients that are in remaining
    for _ in range(k):
        best = None
        best_cover = set()
        for r in candidates:
            req = set(normalize_ings(r["ingredients"]))
            cover = req & remaining
            if len(cover) > len(best_cover):
                best = r
                best_cover = cover
        if not best or len(best_cover) == 0:
            break
        chosen.append({"recipe": best, "covered": sorted(list(best_cover))})
        remaining -= best_cover
        candidates.remove(best)
    return chosen, sorted(list(avail_norm - remaining))  # chosen + total_covered

def backtracking_best_combo(available_ings, max_recipes=3):
    """
    Backtracking search to find combination of up to max_recipes recipes
    that maximizes coverage of available ingredients (primary objective)
    and secondarily maximizes sum of match scores.
    Returns best combination (list of recipe dicts) and stats.
    """
    avail_norm = set(normalize_ings(available_ings))
    # Consider only recipes that have some intersection
    candidates = [r for r in recipes if set(normalize_ings(r["ingredients"])) & avail_norm]
    best_solution = {"recipes": [], "coverage_count": 0, "score_sum": 0, "covered": set()}

    def dfs(start, chosen, covered, score_sum):
        # update best
        if len(covered) > best_solution["coverage_count"] or (
            len(covered) == best_solution["coverage_count"] and score_sum > best_solution["score_sum"]
        ):
            best_solution["recipes"] = chosen.copy()
            best_solution["coverage_count"] = len(covered)
            best_solution["score_sum"] = score_sum
            best_solution["covered"] = covered.copy()

        if len(chosen) >= max_recipes:
            return
        for i in range(start, len(candidates)):
            r = candidates[i]
            req = set(normalize_ings(r["ingredients"]))
            new_covered = covered | (req & avail_norm)
            s, _, _ = match_score(r, avail_norm)
            chosen.append(r)
            dfs(i + 1, chosen, new_covered, score_sum + s)
            chosen.pop()

    dfs(0, [], set(), 0.0)
    return best_solution

def substitution_suggestions(missing_list):
    suggestions = {}
    for m in missing_list:
        m_norm = m.lower()
        if m_norm in SUBSTITUTIONS:
            suggestions[m] = SUBSTITUTIONS[m_norm]
    return suggestions

# ---- Flask routes ----

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recipes", methods=["POST"])
def recipe_list():
    # Ingredients provided by user
    raw = request.form.get("ingredients", "")
    available = [i.strip() for i in raw.split(",") if i.strip()]
    available_norm = set(normalize_ings(available))

    # Algorithm options from form (if any)
    insights = request.form.get("insights", "on")  # 'on' or None
    num_combo = int(request.form.get("num_combo", "3"))

    # Build results
    matched_recipes = []
    for r in recipes:
        s, missing, present = match_score(r, available_norm)
        # only include recipes that have at least one matching ingredient
        if s > 0:
            matched_recipes.append({
                "recipe": r,
                "score": round(s, 3),
                "missing": missing,
                "present": present,
                "substitutions": substitution_suggestions(missing)
            })

    # Sort matched recipes by score desc then by number of present ingredients
    matched_recipes.sort(key=lambda x: (-x["score"], -len(x["present"])))

    # Build graph (for visualization/insights)
    graph = build_bipartite_graph(available)

    # Greedy recommendation
    greedy_choices, greedy_covered = greedy_recommendation(available, k=num_combo)

    # Backtracking best combo
    bt_best = backtracking_best_combo(available, max_recipes=num_combo)

    return render_template(
        "recipe_list.html",
        recipes=matched_recipes,
        available_raw=", ".join(sorted(list(available_norm))),
        graph=graph,
        greedy_choices=greedy_choices,
        greedy_covered=greedy_covered,
        bt_best=bt_best,
        insights=(insights == "on"),
        num_combo=num_combo
    )

@app.route("/recipe/<int:recipe_id>")
def recipe_detail(recipe_id):
    recipe = next((r for r in recipes if r["id"] == recipe_id), None)
    if not recipe:
        return "Recipe not found!", 404
    return render_template("recipe_detail.html", recipe=recipe, substitutions=SUBSTITUTIONS)

if __name__ == "__main__":
    app.run(debug=True)
