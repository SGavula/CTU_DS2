import json
import csv

# with open("./out_dataset/users_minimal.jsonl") as f, open("./out_dataset/users.csv", "w", newline="") as out:
#     writer = csv.writer(out)
#     writer.writerow(["user_id", "username", "game_id"])
#     for line in f:
#         row = json.loads(line)
#         user_id = row["user_id"]
#         username = row["username"]
#         for game_id in row["games"]:
#             writer.writerow([user_id, username, game_id])

with open("./out_dataset/games_minimal.jsonl") as f, open("./out_dataset/games.csv", "w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow(["game_id", "name"])
    for line in f:
        row = json.loads(line)
        game_id = row["game_id"]
        name = row["name"]
        writer.writerow([game_id, name])
