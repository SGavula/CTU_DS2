import json
import time
import redis
from pymongo import MongoClient
from datetime import datetime, timedelta, UTC
from bson.json_util import dumps, loads

# MONGODB CONFIGURATION
USERNAME = "f25_gavulsim"
PASSWORD = "11PpiDXDWo"
HOST = "nosql.felk.cvut.cz"
PORT = 42222
DATABASE = USERNAME

# Connection for redis and mongodb
def init_connections():
    r = redis.Redis(
        host="localhost",
        port=6395,
        password="11PpiDXDWo",
        db=0
    )

    mongo_uri = f"mongodb://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client[DATABASE]

    return r, db

def save_to_redis(r, cache_key, result, ttl):
    r.setex(cache_key, ttl, dumps(result))

def get_from_redis(r, cache_key):
    cached = r.get(cache_key)
    if cached is None:
        return None
    return loads(cached)

def get_average_num_games(r, collection, ttl):
    cache_key = "stats:avg_num_games"

    # Try cache
    cached = get_from_redis(r, cache_key)
    if cached:
        return cached, True   # from cache

    # MongoDB aggregation
    pipeline = [
        {
            "$addFields": {
                "num_of_games": {"$size": "$games"}
            }
        },
        {
            "$group": {
                "_id": None,
                "average_num_of_game": {"$avg": "$num_of_games"}
            }
        }
    ]

    result = list(collection.aggregate(pipeline))

    # Save to Redis
    save_to_redis(r, cache_key, result, ttl)

    return result, False

def get_top_rated_games_per_genre(r, collection, ttl):
    cache_key = "stats:top_rated_games_per_genre:30d"

    # Try cache
    cached = get_from_redis(r, cache_key)
    if cached:
        return cached, True

    # MongoDB aggregation
    pipeline = [
        {
            "$addFields": {
                "createdDate": {
                    "$dateFromString": { "dateString": "$created" }
                }
            }
        },
        {
            "$match": {
                "createdDate": {
                    "$gte": datetime.now(UTC) - timedelta(days=30)
                }
            }
        }, 
        {
            "$group": {
                "_id": "$game_id",
                "average_score": { "$avg": "$value" }
            }
        },
        {
            "$lookup": {
                "from": "games",
                "localField": "_id",
                "foreignField": "game_id",
                "as": "game"
            }
        },
        {
            "$unwind": "$game"
        },
        {
            "$unwind": "$game.genres"
        },
        {
            "$sort": {
                "average_score": -1
            }
        },
        {
            "$group": {
                "_id": "$game.genres",
                "game_id": { "$first": "$game.game_id" },
                "game_title": { "$first": "$game.name" },
                "genre": { "$first": "$game.genres" },
                "average_score": { "$first": "$average_score" }
            }
        },
        {
            "$sort": {
                "average_score": -1
            }
        }
    ]

    result = list(collection.aggregate(pipeline))

    # Save to Redis
    save_to_redis(r, cache_key, result, ttl)

    return result, False

def get_average_score_per_publisher(r, collection, ttl):
    cache_key = "stats:average_score_per_publisher"

    # Try cache
    cached = get_from_redis(r, cache_key)
    if cached:
        return cached, True

    # MongoDB aggregation
    pipeline = [
        {
            "$group": {
                "_id": "$game_id",
                "average_score": {"$avg": "$value"}
            }
        },
        {
            "$lookup": {
                "from": "games",
                "localField": "_id",
                "foreignField": "game_id",
                "as": "game"
            }
        },
        {
            "$unwind": "$game"
        },
        {
            "$lookup": {
                "from": "publishers",
                "localField": "game.publishing_studio_id",
                "foreignField": "publishing_studio_id",
                "as": "publisher"
            }
        },
        {
            "$unwind": "$publisher"
        },
        {
            "$group": {
                "_id": "$publisher.publishing_studio_id",
                "publisher_name": {"$first": "$publisher.name"},
                "average_score": {"$avg": "$average_score"}
            }
        },
        {
            "$sort": {
                "average_score": -1
            }
        }
    ]

    result = list(collection.aggregate(pipeline))

    # Save to Redis
    save_to_redis(r, cache_key, result, ttl)

    return result, False

def get_top_tags(r, collection, ttl):
    cache_key = "games:top_tags:50"

    # Try Redis
    cached = get_from_redis(r, cache_key)
    if cached:
        return cached, True

    # MongoDB aggregation
    pipeline = [
        {"$unwind": "$tags"},
        {
            "$group": {
                "_id": "$tags",
                "count": {"$sum": 1}
            }
        },
        {"$sort": {"count": -1}},
        {"$limit": 50},
        {
            "$project": {
                "_id": 0,
                "tag": "$_id",
                "occurrences": "$count"
            }
        }
    ]

    result = list(collection.aggregate(pipeline))

    save_to_redis(r, cache_key, result, ttl)

    return result, False


def get_top_expensive_games_last_year(r, collection, ttl):
    cache_key = "games:top_expensive:12m"

    # Try Redis
    cached = get_from_redis(r, cache_key)
    if cached:
        return cached, True

    # MongoDB aggregation
    pipeline = [
        {
            "$addFields": {
                "release_date_dt": {"$toDate": "$release_date"}
            }
        },
        {
            "$match": {
                "release_date_dt": {
                    "$gte": datetime.now(UTC) - timedelta(days=365)
                }
            }
        },
        {"$sort": {"price": -1}},
        {"$limit": 20},
        {
            "$project": {
                "_id": 0,
                "game_id": 1,
                "name": 1,
                "price": 1,
                "release_date": 1
            }
        }
    ]

    result = list(collection.aggregate(pipeline))

    save_to_redis(r, cache_key, result, ttl)

    return result, False


def get_top_helpful_reviews_last_week(r, collection, game_id, ttl):
    # Deterministic cache key
    cache_key = f"reviews:top_helpful:game:{game_id}:7d"

    # Try Redis
    cached = get_from_redis(r, cache_key)
    if cached:
        return cached, True

    # MongoDB aggregation: last 7 days
    pipeline = [
        {
            "$match": {
                "game_id": game_id,
                "created": {
                    "$gte": datetime.now(UTC) - timedelta(days=7)
                }
            }
        },
        {"$sort": {"helpful_votes": -1}},
        {"$limit": 10},
        {
            "$project": {
                "_id": 0,
                "review_id": 1,
                "content": 1,
                "helpful_votes": 1
            }
        }
    ]

    result = list(collection.aggregate(pipeline))

    # Save to Redis
    save_to_redis(r, cache_key, result, ttl)

    return result, False

# Clearing redis
def purge_cache(redis_client, key=None, key_prefix=None):
    if key:
        redis_client.delete(key)
        return

    if key_prefix or key_prefix == "":
        pattern = f"{key_prefix}*"
        print("Remove keys based on pattern: ", pattern)
        for k in redis_client.scan_iter(pattern):
            redis_client.delete(k)


# Function for testing speed of query
def measure_time(func, *args):
    start = time.time()
    result, from_cache = func(*args)
    duration = (time.time() - start) * 1000  # ms
    return result, from_cache, duration


def DoTest():

    r, db = init_connections()

    # Clear all keys before demo
    purge_cache(r, key_prefix="")

    tests = [
        ("Task 2 - Calculate average number of purchased games by users",
         get_average_num_games,
         db["users"],
         3600,
         None),

         ("Task 4 - Find top-rated games by score per genre in the last month",
         get_top_rated_games_per_genre,
         db["scores"],
         3600,
         None),

         ("Task 4 - Find top-rated games by score per genre in the last month",
         get_average_score_per_publisher,
         db["scores"],
         3600,
         None),

        ("Task 13 – Top expensive games (last year)",
         get_top_expensive_games_last_year,
         db["games"],
         21600,
         None),

        ("Task 17 – Top 50 tags",
         get_top_tags,
         db["games"],
         43200,
         None),

        ("Task 19 – Top helpful reviews (last week)",
         get_top_helpful_reviews_last_week,
         db["reviews"],
         3600,
         [4381]),
        
        # (
        #     "User – Get user by name",
        #     get_user_by_name,
        #     db["users"],
        #     3600,  # 1h
        #     ["someUserName"]  # username param
        # ),

        # (
        #     "Stats – Average number of games per user",
        #     get_average_num_games,
        #     db["users"],
        #     86400,  # 24h cache
        #     None
        # ),

        # (
        #     "Stats – Top rated games per genre (last 30 days)",
        #     get_top_rated_games_per_genre,
        #     db["ratings"],
        #     86400,  # 24h cache
        #     None
        # ),
    ]

    all_results = {}

    for test_name, func, collection, ttl, params in tests:
        print(f"\nRunning test: {test_name}")
        runs_results = run_cache_multiple(test_name, func, r, collection, ttl, params, runs=5)
        all_results[test_name] = runs_results



def run_cache_multiple(test_name, func, r, collection, ttl, params=None, runs=5):
    """
    Runs a cached query multiple times to get more accurate timing.
    Returns a list of dicts with run results.
    """
    results = []

    for i in range(1, runs + 1):
        args = [r, collection] if 'r' in func.__code__.co_varnames else [collection]
        if params:
            args.extend(params)
        if 'ttl' in func.__code__.co_varnames:
            args.append(ttl)

        if i == 3:
            purge_cache(r, key_prefix="stats")

        result, from_cache, ms = measure_time(func, *args)
        results.append({
            "run": i,
            "cache_hit": from_cache,
            "time_ms": ms
        })
        print(f"{test_name} — Run {i}: Cache hit={from_cache}, Time={ms:.2f} ms")

    return results



if __name__ == "__main__":
    DoTest()


