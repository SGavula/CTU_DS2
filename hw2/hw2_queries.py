import redis

# Redis Connection
def get_redis_connection():
    return redis.Redis(host="localhost", port=6395, password="11PpiDXDWo", db=0)

# TASK 1 – Top 10 most active users (last 7 days)
def get_top10_active_users_last7days(r, day_keys, ttl_seconds=86400):
    print("Task 1: Top 10 Most Active Users (Last 7 Days)")

    # Perform union (SUM aggregation)
    r.zunionstore("activity:last7days", day_keys, aggregate="SUM")

    # Set temporary TTL
    r.expire("activity:last7days", ttl_seconds)

    # Get top 10 users by total playtime
    top10 = r.zrange("activity:last7days", 0, 9, desc=True, withscores=True)

    results = []
    for idx, (user_id, score) in enumerate(top10, start=1):
        user_id_dec = user_id.decode()
        user_key = f"user:{user_id_dec}"
        username = r.hget(user_key, "username").decode()
        results.append({
            "rank": idx,
            "user_id": user_id_dec,
            "username": username,
            "total_playtime": round(float(score), 2)
        })

    return results


# TASK 10 – Top 10 users with most achievements per game
def get_top_users_for_game(r, game_id, top_n=10):
    print(f"Task 10: Top {top_n} Users with Most Achievements in Game {game_id}")

    top_users = r.zrevrange(f"achievements_count:{game_id}", 0, top_n - 1, withscores=True)
    results = []

    for user_id, count in top_users:
        user_id_dec = user_id.decode()
        user_key = f"user:{user_id_dec}"
        username = r.hget(user_key, "username").decode()
        results.append({
            "user_id": user_id_dec,
            "username": username,
            "game_id": game_id,
            "achievements_unlocked": int(count)
        })

    return results


# TASK 11 – Top 10 largest game studios
def get_top10_studios_by_size(r):
    print("Task 11: Top 10 Largest Studios by Number of Developers")

    top = r.zrevrange("studio:counts", 0, 9, withscores=True)
    results = []

    for idx, (sid, score) in enumerate(top, start=1):
        sid_dec = sid.decode()
        name = r.hget(f"studio:{sid_dec}", "name").decode()
        results.append({
            "rank": idx,
            "studio_id": sid_dec,
            "studio_name": name,
            "developer_count": int(score)
        })

    return results


# MAIN DEMO
if __name__ == "__main__":
    r = get_redis_connection()

    # Example for Task 1
    day_keys = {
        "activity:2025-11-06": 1,
        "activity:2025-11-05": 1,
        "activity:2025-11-04": 1,
        "activity:2025-11-03": 1,
        "activity:2025-11-02": 1,
        "activity:2025-11-01": 1,
        "activity:2025-10-31": 1,
    }

    top10_users = get_top10_active_users_last7days(r, day_keys)
    for u in top10_users:
        print(f"{u['rank']:2d}. User id: {u['user_id']}, Username: {u['username']} → {u['total_playtime']} h")

    print("\n")

    # Example for Task 10
    game_id = 42
    top10_achievers = get_top_users_for_game(r, game_id)
    for rank, user in enumerate(top10_achievers, 1):
        print(f"{rank:2d}. User id: {user['user_id']}, Username: {user['username']} → {user['achievements_unlocked']} achievements")

    print("\n")

    # Example for Task 11
    top10_studios = get_top10_studios_by_size(r)
    for s in top10_studios:
        print(f"{s['rank']:2d}. Studio id: {s['studio_id']}, Studio name: {s['studio_name']} → {s['developer_count']} developers")
