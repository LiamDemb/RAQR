import json

with open("data/raw/nq_final.jsonl", "r") as f:
    found = False
    for line in f:
        data = json.loads(line)  # parse one JSON object

        if (
            data.get("question_id")["text"]
            == "who was the first president to get his picture taken"
        ):
            found = True
            print(data)
    if found:
        print("Found!!")
    else:
        print("Not found!!")
