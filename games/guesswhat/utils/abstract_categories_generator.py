import json
import nltk
from nltk.corpus import wordnet as wn

nltk.download('wordnet')
nltk.download('omw-1.4')

file_path = 'abstract_dataset.json'  
with open(file_path, 'r') as file:
    data = json.load(file)

categories = {}

# List of categories to exclude
excluded_categories = []  

# Function to get the hypernym of a word
def get_hypernym(word):
    synsets = wn.synsets(word)

    if not synsets:
        return None
    hypernym = synsets[0].hypernyms()

    if not hypernym:
        return None
    return hypernym[0]

# Categories and subcategories with members
for group in data:
    input_category = group.get("category", "unknown")  # Get the input category

    if input_category in excluded_categories:
        continue  # Skip this category if it's in the exclusion list

    if input_category not in categories:
        categories[input_category] = {}

    for member in group["members"]:
        hypernym = get_hypernym(member)
        if hypernym:
            hypernym_name = hypernym.name()
            if hypernym_name not in categories[input_category]:
                categories[input_category][hypernym_name] = []
            categories[input_category][hypernym_name].append(member)

# Convert synsets to readable names and include members
categories_readable = []

for input_category, subcategories in categories.items():
    subcategories_readable = [{"Subcategory": sub.replace('_', ' '), "Members": members} for sub, members in subcategories.items()]
    categories_readable.append({"Category": input_category, "Subcategories": subcategories_readable})


output = {"Categories": categories_readable}

output_file_path = 'abstract_categories_subcategories.json' 
with open(output_file_path, 'w') as output_file:
    json.dump(output, output_file, indent=4)

print(f"Data has been saved to {output_file_path}")
