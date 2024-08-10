"""Create label prompts"""
import json


def create_ensemble_prompts():
    "Create ours RAGP labels"
    template = "a photo of a{age}{race}{gender}"

    age_labels = [" young", " middle-aged", " old", ""]

    race_labels = [" black", " indian", " latino hispanic", " white",
                   " middle eastern", " southeast asian", " east asian",
                   ""]

    gender_labels = [" woman", " man"]

    age_prompts = [template.format(
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


def create_raw_race_prompts():
    """Create baseline raw race labels"""
    template = "a photo of a{race} person"
    race_labels = [" black", " indian", " latino hispanic", " white",
                   " middle eastern", " southeast asian", " east asian"]
    final_prompts = [template.format(race=label) for label in race_labels]
    with open("raw_race_labels.json", "w", encoding="utf-8") as final:
        json.dump(final_prompts, final)


if __name__ == "__main__":
    create_raw_race_prompts()
