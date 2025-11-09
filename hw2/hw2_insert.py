import json
import redis

# Redis Connection
def get_redis_connection():
    return redis.Redis(host="localhost", port=6395, password="11PpiDXDWo", db=0)

# TASK 1 - Load Users and Activities
def load_users_and_activities(r, users_file="data/users.jsonl", activities_file="data/activities.jsonl"):
    print("Task 1: Loading Users and Activities")

    # Load users.jsonl
    with open(users_file, "r", encoding="utf8") as f:
        for line in f:
            user = json.loads(line.strip())
            user_id = user["user_id"]
            username = user["username"]

            # Save users to Redis as hashsets
            r.hset(f"user:{user_id}", mapping={"username": username})

    print("Users loaded successfully.")

    # Load activities.jsonl
    with open(activities_file, "r", encoding="utf8") as f:
        for line in f:
            act = json.loads(line.strip())
            act_id = act["activity_id"]
            user_id = act["user_id"]
            num_of_hours = act["number_of_hours"]
            timestamp = act["timestamp"]

            # Avoid duplicates using control set
            if r.sadd("events:loaded", act_id):
                r.zincrby(f"activity:{timestamp}", num_of_hours, user_id)
            else:
                print(f"Duplicate activity skipped: {act_id}")

    print("Activities loaded successfully.")


# TASK 10 - Load Achievements per Game
def load_achievements(r, achievements_file="data/achievements.jsonl"):
    print("\nTask 10: Loading Achievements per Game")

    with open(achievements_file, "r", encoding="utf8") as f:
        for line in f:
            achievement = json.loads(line)
            game_id = achievement["game_id"]
            users = achievement.get("users", [])

            if not users:
                continue

            for user_id in users:
                user_ach_key = f"user:{user_id}:game:{game_id}:achievements"

                # Avoid double-counting
                added = r.sadd(user_ach_key, f"{game_id}-{user_id}")

                if added:
                    r.zincrby(f"achievements_count:{game_id}", 1, user_id)

    print("Achievements loaded successfully.")


# TASK 11 - Load Game Studios and Members
def load_studios(r, studios_file="data/studios.json"):
    print("\nTask 11: Loading Studios and Members")

    with open(studios_file, "r", encoding="utf8") as f:
        studios = json.load(f)

    for studio in studios:
        raw_id = studio["game_studio_id"]
        sid = f"s{raw_id}"
        name = studio["name"]
        users = studio["users"]

        # Store studio info
        r.hset(f"studio:{sid}", mapping={"name": name})

        for uid in users:
            uid = str(uid)
            old_sid = r.get(f"user:studio:{uid}")

            if old_sid == sid:
                r.sadd(f"studio:members:{sid}", uid)
                continue

            if old_sid:
                r.srem(f"studio:members:{old_sid}", uid)
                r.zincrby("studio:counts", -1, old_sid)

            added = r.sadd(f"studio:members:{sid}", uid)
            if added == 1:
                r.zincrby("studio:counts", 1, sid)

            r.set(f"user:studio:{uid}", sid)

    print("Studios loaded successfully.")

def main():
    r = get_redis_connection()
    print("Connected to Redis.")

    # Call each task loader
    load_users_and_activities(r)
    load_achievements(r)
    load_studios(r)

    print("\n=== All Data Loaded Successfully ===")
    print(f"Redis now contains {r.dbsize()} keys.")

if __name__ == "__main__":
    main()
