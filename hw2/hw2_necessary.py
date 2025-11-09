import json

input_file_1 = "../out_dataset/users.jsonl"
output_file_1 = "../out_dataset/users_minimal.jsonl"

with open(input_file_1, "r", encoding="utf8") as infile, open(output_file_1, "w", encoding="utf8") as outfile:
    for line in infile:
        user = json.loads(line)
        user_id = user["user_id"]
        username = user["username"]
        minimal = {
            "user_id": user_id,
            "username": username
        }
        outfile.write(json.dumps(minimal, ensure_ascii=False) + "\n")

print(f"Minimal achievements JSON written to {output_file_1}")

input_file_2 = "../out_dataset/activities.jsonl"
output_file_2 = "../out_dataset/activities_minimal.jsonl"

with open(input_file_2, "r", encoding="utf8") as infile, open(output_file_2, "w", encoding="utf8") as outfile:
    for line in infile:
        act = json.loads(line)
        act_id = act["activity_id"]
        user_id = act["user_id"]
        num_of_hours = act["number_of_hours"]
        timestamp = act["timestamp"]
        
        minimal = {
            "activity_id": act_id,
            "user_id": user_id,
            "number_of_hours": num_of_hours,
            "timestamp": timestamp
        }
        outfile.write(json.dumps(minimal, ensure_ascii=False) + "\n")

print(f"Minimal achievements JSON written to {output_file_1}")