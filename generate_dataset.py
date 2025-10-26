import argparse
import random
import json
import csv
import os
import string
from datetime import datetime, timedelta
import numpy as np
 
# --------- Helpers ----------
def make_dir(d):
    os.makedirs(d, exist_ok=True)
 
def rand_str(n=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n))
 
def rand_name():
    # mix of words to make more readable titles
    a = ["Red", "Blue", "Shadow", "Sky", "Star", "Iron", "Legend", "Quantum", "Night", "Echo", "Void", "Solar"]
    b = ["Saga", "Rift", "Quest", "Tale", "Drive", "Pilot", "Blade", "Field", "Chronicle", "Run"]
    return f"{random.choice(a)} {random.choice(b)} {random.randint(1,999)}"
 
def rand_date_between(start_date, end_date):
    delta = end_date - start_date
    return start_date + timedelta(days=random.randint(0, delta.days))
 
def sample_countries(n, top10_weight=0.8):
    top10 = ["USA","UK","Germany","France","Japan","Canada","Australia","Czech Republic","Spain","Poland"]
    others = ["Italy","Brazil","Sweden","Netherlands","Russia","China","India","Mexico","Argentina","Belgium"]
    # Build population: top10 repeated to bias
    pool = top10 * 10 + others
    return [random.choice(pool) for _ in range(n)]
 
# --------- Distributions ----------
def price_sample(n):
    # Right skew: mixture of many low/free prices and fewer high ones
    # We'll use a lognormal and clamp near-zero for "free"
    vals = np.random.lognormal(mean=2.5, sigma=1.2, size=n)  # median ~12
    # Introduce some free games (~10%)
    mask_free = np.random.rand(n) < 0.10
    vals[mask_free] = 0.0
    # Round to 2 decimals
    return [round(float(max(0.0, v)), 2) for v in vals]
 
def pick_k_by_zipf(pop, k_max, a=1.2):
    # Return number k drawn 1..k_max with Zipf-like: P(k) ~ 1/k^a
    # We'll generate a pmf array and sample
    ranks = np.arange(1, k_max+1, dtype=float)
    pmf = 1.0 / np.power(ranks, a)
    pmf /= pmf.sum()
    return np.random.choice(np.arange(1, k_max+1), p=pmf)
 
# --------- Generators ----------
def gen_users(n_users, start_reg, end_reg):
    users = []
    countries = sample_countries(n_users)
    for i in range(1, n_users+1):
        users.append({
            "user_id": i,
            "username": f"user_{rand_str(6)}",
            "email": f"user{i}@example.com",
            "password": rand_str(12),
            "registration_date": rand_date_between(start_reg, end_reg).isoformat(),
            "country": countries[i-1],
            "games": []  # will be filled from purchases/activities later optionally
        })
    return users
 
def gen_studios(n_studios, users):
    studios = []
    user_ids = [u["user_id"] for u in users]
    for i in range(1, n_studios+1):
        # small studio sizes mostly 1-10
        size = random.choices(population=list(range(1,11)), weights=[0.25]+[0.09]*9, k=1)[0]
        members = random.sample(user_ids, k=min(size, len(user_ids)))
        studios.append({
            "game_studio_id": i,
            "name": f"{rand_str(5)} Studio",
            "description": f"Studio description {i}",
            "users": members
        })
    return studios
 
def gen_publishers(n_publishers, users):
    publishers = []
    user_ids = [u["user_id"] for u in users]
    for i in range(1, n_publishers+1):
        size = random.choices(population=list(range(1,11)), weights=[0.25]+[0.09]*9, k=1)[0]
        members = random.sample(user_ids, k=min(size, len(user_ids)))
        publishers.append({
            "publishing_studio_id": i,
            "name": f"{rand_str(5)} Publishing",
            "description": f"Publisher description {i}",
            "users": members
        })
    return publishers
 
def gen_genres():
    base = ["Action","Adventure","RPG","Simulation","Strategy","Sports","Puzzle","Shooter",
            "Platformer","Horror","Racing","Fighting","Casual","Multiplayer","Survival",
            "Stealth","Roguelike","Visual Novel","Sandbox","Indie"]
    return [{"name": g, "description": f"{g} games"} for g in base]
 
def gen_tags(n_tags=500):
    # create a list of tags; for base we create 500 unique tags (can be adjusted)
    template = ["Multiplayer","Singleplayer","Co-op","Controller-friendly","VR-supported","4K","Accessibility","Indie",
                "Pixel-art","Open-world","Sandbox","Story-rich","Competitive","Family-friendly","Local Multiplayer"]
    tags = []
    for i in range(n_tags):
        name = template[i % len(template)] + ("" if i < len(template) else f"_{i}")
        tags.append({"name": name, "description": f"Tag {name}"})
    return tags
 
def gen_games(n_games, studios, publishers, genres, tags, release_start, release_end):
    games = []
    prices = price_sample(n_games)
    studio_ids = [s["game_studio_id"] for s in studios]
    pub_ids = [p["publishing_studio_id"] for p in publishers]
    genre_names = [g["name"] for g in genres]
    tag_names = [t["name"] for t in tags]
    now = datetime.now()
    for i in range(1, n_games+1):
        # genres 1-5
        k_gen = max(1, min(5, int(np.random.poisson(1.8))) ) # mostly 1-3
        game_genres = random.sample(genre_names, k=min(k_gen, len(genre_names)))
        # tags 0-10
        k_tag = min(20, max(0, int(np.random.poisson(1.2))))
        game_tags = random.sample(tag_names, k=min(k_tag, len(tag_names))) if k_tag>0 else []
        release_date = rand_date_between(release_start, release_end).isoformat()
        games.append({
            "game_id": i,
            "name": rand_name(),
            "price": prices[i-1],
            "achievements": [],   # will be filled by achievement generator
            "release_date": release_date,
            "supported_languages": random.sample(["EN","DE","FR","JP","CZ","PL","ES","RU","ZH"], k=random.randint(1,4)),
            "requirements": {"min": "2GB RAM, 2GHz CPU", "recommended": "8GB RAM, 4GHz CPU, GPU"},
            "tags": game_tags,
            "genres": game_genres,
            "game_studio_id": random.choice(studio_ids) if studio_ids else None,
            "publishing_studio_id": random.choice(pub_ids) if pub_ids else None,
            "reviews": [],
            "score": []
        })
    return games
 
def gen_achievements(n_ach, games, users, seed):
    achs = []
    user_ids = [u["user_id"] for u in users]
    n_games = len(games)
    # distribute achievements across games: some games more achievement-heavy
    # We'll assign achievements by sampling game index from Zipf-like
    ranks = np.arange(1, n_games+1)
    a = 1.1
    pmf = 1.0 / np.power(ranks, a)
    pmf /= pmf.sum()
    game_choices = np.random.choice(range(n_games), size=n_ach, p=pmf)
    for i, g_idx in enumerate(game_choices, start=1):
        g = games[g_idx]
        name = f"ACH-{g['name'][:6]}-{i}"
        # number of users who completed it: skewed, top users do more -> sample 0..small fraction
        n_users_completed = max(0, int(np.random.pareto(1.5) * 2))  # many zeros/low numbers
        completed_users = random.sample(user_ids, k=min(len(user_ids), n_users_completed)) if n_users_completed>0 else []
        ach = {
            "achievement_id": i,
            "game_id": g["game_id"],
            "users": completed_users,
            "name": name
        }
        achs.append(ach)
        # append id to game's achievements list
        g["achievements"].append(i)
    return achs
 
def gen_game_views(n_views, games, users):
    views = []
    game_ids = [g["game_id"] for g in games]
    # popularity: top 15% games ~80% views -> use Zipf with stronger skew
    a = 1.3
    ranks = np.arange(1, len(games)+1)
    pmf = 1.0 / np.power(ranks, a)
    pmf /= pmf.sum()
    # sample games with pmf; sample users uniformly but with bias for active users
    for vid in range(1, n_views+1):
        g_idx = np.random.choice(range(len(games)), p=pmf)
        g = games[g_idx]
        u = random.choice(users)
        num_views = max(1, int(np.random.poisson(1.5)))
        view_ts = datetime.now() - timedelta(days=random.randint(0, 365))
        views.append({
            "game_view_id": vid,
            "game_id": g["game_id"],
            "user_id": u["user_id"],
            "viewed": view_ts.isoformat(),
            "number_of_views": num_views
        })
    return views
 
def gen_game_activity(n_acts, games, users):
    acts = []
    # top 10% users account for ~70% hours -> create user activity weight distribution
    user_ids = [u["user_id"] for u in users]
    U = len(user_ids)
    # create a Pareto-distributed "activity score" to bias selection
    activity_scores = np.random.pareto(a=1.5, size=U) + 1.0
    activity_scores /= activity_scores.sum()
    for i in range(1, n_acts+1):
        # pick user by activity_scores
        u_idx = np.random.choice(range(U), p=activity_scores)
        u_id = user_ids[u_idx]
        g = random.choice(games)
        # number_of_hours: most 50-100, a few >1000 -> use lognormal with long tail
        hours = float(np.random.lognormal(mean=4.0, sigma=1.2))  # median ~54
        # clamp extreme values but allow >1000 occasionally
        if hours > 3000:
            hours = 3000.0
        hours = round(hours, 2)
        acts.append({
            "activity_id": i,
            "game_id": g["game_id"],
            "user_id": u_id,
            "number_of_hours": hours
        })
    return acts
 
def gen_reviews(n_reviews, users, games):
    reviews = []
    user_ids = [u["user_id"] for u in users]
    # Track which users have already written a review
    users_used = set()

    for i in range(1, n_reviews+1):
        # Select a user who hasn't written a review yet
        available_users = [uid for uid in user_ids if uid not in users_used]
        # Stop if we run out of available users
        if not available_users:
            break
        # Generate random user_id from available users
        u_id = random.choice(available_users)
        # Add generated user_id to set of used users
        users_used.add(u_id)

        g = random.choice(games)
        created = (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat()
        content = f"This is a review for {g['name']} by user {u_id}."
        helpful = int(np.random.zipf(2.0) if random.random() < 0.05 else np.random.poisson(1))
        review = {
            "review_id": i,
            "created": created,
            "user_id": u_id,
            "game_id": g["game_id"],
            "content": content,
            "helpful_votes": helpful
        }
        reviews.append(review)
        g["reviews"].append(i)
    return reviews
 
def gen_scores(n_scores, users, games):
    scores = []
    user_ids = [u["user_id"] for u in users]
    # ratings follow approx normal centered 7-8, clamp 0..10
    for i in range(1, n_scores+1):
        user_id = random.choice(user_ids)
        g = random.choice(games)
        val = float(np.clip(np.random.normal(loc=7.5, scale=1.5), 0.0, 10.0))
        score = {
            "score_id": i,
            "created": (datetime.now() - timedelta(days=random.randint(0,365))).isoformat(),
            "user_id": user_id,
            "game_id": g["game_id"],
            "value": round(val, 1)
        }
        scores.append(score)
        g["score"].append(i)
    return scores
 
def gen_wishlists(users, games):
    wishlists = []
    game_ids = [g["game_id"] for g in games]
    for i, u in enumerate(users, start=1):
        # most wishlists 1-10 games; 30% private
        size = np.random.choice(range(0,51), p=np.linspace(0.9,0.1,51)/np.linspace(0.9,0.1,51).sum())
        size = int(min(50, max(0, size)))
        chosen = random.sample(game_ids, k=min(size, len(game_ids))) if size>0 else []
        wl = {
            "wishlist_id": i,
            "user_id": u["user_id"],
            "visibility": "private" if random.random() < 0.3 else "public",
            "games": chosen
        }
        wishlists.append(wl)
    return wishlists
 
# --------- I/O helpers ----------
def write_jsonl(path, records):
    with open(path, "w", encoding="utf8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
 
def write_json(path, obj):
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
 
def write_csv(path, rows, header):
    with open(path, "w", newline='', encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
 
# --------- Main ----------
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    make_dir(args.outdir)
 
    # dates
    today = datetime.now()
    release_start = today - timedelta(days=365*10)  # past decade
    release_end = today
    reg_start = today - timedelta(days=365*10)
    reg_end = today
 
    print("Generating users...")
    users = gen_users(args.n_users, reg_start, reg_end)
 
    print("Generating game studios and publishers...")
    studios = gen_studios(args.n_studios, users)
    publishers = gen_publishers(args.n_publishers, users)
 
    print("Generating genres & tags...")
    genres = gen_genres()
    tags = gen_tags(n_tags=args.n_tags)
 
    print("Generating games...")
    games = gen_games(args.n_games, studios, publishers, genres, tags, release_start, release_end)
 
    print(f"Generating {args.n_achievements} achievements...")
    achievements = gen_achievements(args.n_achievements, games, users, args.seed)
 
    print(f"Generating {args.n_game_views} game views...")
    game_views = gen_game_views(args.n_game_views, games, users)
 
    print(f"Generating {args.n_activities} game activity records (play hours)...")
    activities = gen_game_activity(args.n_activities, games, users)
 
    print(f"Generating {args.n_reviews} reviews...")
    reviews = gen_reviews(args.n_reviews, users, games)
 
    print(f"Generating {args.n_scores} scores...")
    scores = gen_scores(args.n_scores, users, games)
 
    print("Generating wishlists...")
    wishlists = gen_wishlists(users, games)
 
    # write outputs
    print("Writing files...")
    write_json(os.path.join(args.outdir, "genres.json"), genres)
    write_json(os.path.join(args.outdir, "tags.json"), tags)
    write_json(os.path.join(args.outdir, "studios.json"), studios)
    write_json(os.path.join(args.outdir, "publishers.json"), publishers)
    write_jsonl(os.path.join(args.outdir, "users.jsonl"), users)
    write_jsonl(os.path.join(args.outdir, "games.jsonl"), games)
    write_jsonl(os.path.join(args.outdir, "achievements.jsonl"), achievements)
    write_jsonl(os.path.join(args.outdir, "game_views.jsonl"), game_views)
    write_jsonl(os.path.join(args.outdir, "activities.jsonl"), activities)
    write_jsonl(os.path.join(args.outdir, "reviews.jsonl"), reviews)
    write_jsonl(os.path.join(args.outdir, "scores.jsonl"), scores)
    write_jsonl(os.path.join(args.outdir, "wishlists.jsonl"), wishlists)
 
    # small manifest
    manifest = {
        "seed": args.seed,
        "n_users": args.n_users,
        "n_games": args.n_games,
        "n_achievements": args.n_achievements,
        "n_game_views": args.n_game_views,
        "n_activities": args.n_activities,
        "n_studios": args.n_studios,
        "n_publishers": args.n_publishers,
        "n_reviews": args.n_reviews,
        "n_scores": args.n_scores,
        "n_tags": args.n_tags,
        "generated_at": datetime.now().isoformat()
    }
    write_json(os.path.join(args.outdir, "manifest.json"), manifest)
 
    print("Done. Files written to:", args.outdir)
    # quick summary prints
    print(f"Users: {len(users)}, Games: {len(games)}, Achievements: {len(achievements)}, Activities: {len(activities)}")
 
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20250927)
    p.add_argument("--n-users", type=int, default=10000)
    p.add_argument("--n-games", type=int, default=5000)
    p.add_argument("--n-achievements", type=int, default=50000)
    p.add_argument("--n-game-views", type=int, default=100000)
    p.add_argument("--n-activities", type=int, default=1000000)
    p.add_argument("--n-studios", type=int, default=1000)
    p.add_argument("--n-publishers", type=int, default=500)
    p.add_argument("--n-tags", type=int, default=500)
    p.add_argument("--n-reviews", type=int, default=100000)
    p.add_argument("--n-scores", type=int, default=500000)
    p.add_argument("--outdir", type=str, default="./out_gps")
    args = p.parse_args()
    main(args)