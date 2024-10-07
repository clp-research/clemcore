import json
with open("games/codenames/in/instances.json", "r") as file:
    instances = json.load(file)

with open("games/codenames/in/strict_instances.json", "r") as second_file:
    strict_instances = json.load(second_file)

for i in range(len(instances["experiments"])):
    if not str(instances["experiments"][i]["game_instances"]) == str(strict_instances["experiments"][i]["game_instances"]):
        print(i, instances["experiments"][i]["variable"], instances["experiments"][i]["name"])