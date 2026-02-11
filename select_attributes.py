import json

input_file = "./out_dataset/games.jsonl"
output_file = "./out_dataset/games_minimal.jsonl"

with open(input_file, "r", encoding="utf8") as infile, open(output_file, "w", encoding="utf8") as outfile:
    for line in infile:
        record = json.loads(line)
        minimal = {
            "game_id": record["game_id"],
            "name": record["name"]
        }
        outfile.write(json.dumps(minimal, ensure_ascii=False) + "\n")

print(f"Minimal achievements JSON written to {output_file}")

# minimal_records = []

# with open(input_file, "r", encoding="utf8") as infile:
#     # Load the whole JSON array
#     records = json.load(infile)

# with open(output_file, "w", encoding="utf8") as outfile:
#     for record in records:
#         minimal = {
#             "game_studio_id": record["game_studio_id"],
#             "name": record["name"],
#         }
#         minimal_records.append(minimal)
        
# with open(output_file, "w", encoding="utf8") as outfile:
#     json.dump(minimal_records, outfile, ensure_ascii=False, indent=2)
    
# print(f"Minimal achievements JSON written to {output_file}")