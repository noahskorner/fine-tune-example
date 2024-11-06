import json
import re
import string

# Load the JSON data from the file with UTF-8 encoding
with open("data/train.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Define a regex pattern to capture names at the start of lines or after \r\n
name_pattern = r"(^|\n)(\w+):"

# Process each entry in the dataset
for entry in data:
    dialogue = entry["dialogue"]

    # Find all unique names in the dialogue
    matches = re.findall(name_pattern, dialogue)
    unique_names = set(match[1] for match in matches)

    # Create labels "Character A", "Character B", etc.
    labels = [f"Character {letter}" for letter in string.ascii_uppercase]

    # Map each unique name to a "Character X" label
    name_replacements = {}
    for i, name in enumerate(unique_names):
        name_replacements[name] = labels[i]

    # Replace each name in the dialogue with its assigned "Character X" label
    for name, replacement in name_replacements.items():
        dialogue = re.sub(fr"(^|\n){name}:", fr"\1{replacement}:", dialogue)

    # Update the dialogue in the entry
    entry["dialogue"] = dialogue

# Save the modified data back to a new JSON file
with open("data/train_modified.json", "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=2)

print("Names in dialogues have been replaced with 'Character X' labels and saved to test_modified.json")
