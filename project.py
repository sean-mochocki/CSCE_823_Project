import json
import os
import pandas as pd

# Create an empty list to store the data from each JSON file
data_list = []

# Directory where JSON files are located
json_directory = "/remote_home/AML/Avolve_data/cipub"

# Loop through each file in the directory
for filename in os.listdir(json_directory):
    # Check if the file is a JSON file
    if filename.endswith(".json"):
        file_path = os.path.join(json_directory, filename)
        with open(file_path, "r") as file:
            json_data = json.load(file)
            title = json_data.get("title", "")
            description = json_data.get("description", "")

            topics = []
            topics_data = json_data.get("topics", {})
            for topic_id, topic_list in topics_data.items():
                for topic in topic_list:
                    topics.append(topic)

            # Extract keywords
            keywords = json_data.get("keywords", [])

            if description:
                data_list.append({"title": title, "description": description, "topics": topics, "keywords": keywords})

# Create a pandas DataFrame from the data list
df = pd.DataFrame(data_list)
print(df.head)
