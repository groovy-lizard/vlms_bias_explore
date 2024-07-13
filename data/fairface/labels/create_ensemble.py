"""Create age race gender ensemble prompts"""
import json

TEMPLATE = "a photo of a{age}{race}{gender}"

age_labels = [" young", " middle-aged", " old", ""]

race_labels = [" black", " indian", " latino hispanic",
               " middle eastern", " southeast asian", " east asian", " white",
               ""]

gender_labels = [" woman", " man"]


age_prompts = [TEMPLATE.format(
    age=label, race="{race}", gender="{gender}") for label in age_labels]

race_prompts = []
for ap in age_prompts:
    for rl in race_labels:
        race_prompts.append(ap.format(race=rl, gender="{gender}"))

final_prompts = []
for rp in race_prompts:
    for gl in gender_labels:
        final_prompts.append(rp.format(gender=gl))

with open("age_race_gender.json", "w", encoding="utf-8") as final:
    json.dump(final_prompts, final)
