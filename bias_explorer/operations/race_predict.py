"""Race label predictor"""
import pandas as pd
from ..utils import dataloader
from ..utils import system


def new_race_dict():
    """Generate an empty race score dictionary"""
    rd = {
        "White": 0,
        "Black": 0,
        "Indian": 0,
        "Latino Hispanic": 0,
        "East Asian": 0,
        "Southeast Asian": 0,
        "Middle Eastern": 0
    }

    return rd


def new_race_list():
    """Generate an empty race list dictionary"""
    rd = {
        "White": [],
        "Black": [],
        "Indian": [],
        "Latino Hispanic": [],
        "East Asian": [],
        "Southeast Asian": [],
        "Middle Eastern": []
    }

    return rd


def get_race_from_prompt(prompt):
    """Return the race label from given prompt"""
    race_dict = new_race_dict()
    for race in race_dict:
        if race.lower() in prompt:
            return race
    print(f"error prompt: {prompt}")
    return False


def get_k_score(label_list, k):
    """Get final score of k sum"""
    scores = sorted(label_list, reverse=True)
    return sum(scores[:k])


def get_avg_preds(sims_dict):
    """Generate the average sum predictions"""
    files = sims_dict.keys()
    preds = []

    for val in sims_dict.values():
        race_dict = new_race_dict()
        for prompt, sim_score in val.items():
            race_label = get_race_from_prompt(prompt)
            race_dict[race_label] += sim_score
        preds.append(max(race_dict, key=race_dict.get))

    avg_dict = {'file': files, 'race_preds': preds}
    return pd.DataFrame(data=avg_dict)


def get_top_k_winner(sims_dict, k):
    """Generate dataframe with the winning label of top k method"""
    files = sims_dict.keys()
    preds = []
    for val in sims_dict.values():
        race_list = new_race_list()
        race_scores = new_race_dict()
        for prompt, sim_score in val.items():
            race_label = get_race_from_prompt(prompt)
            race_list[race_label].append(sim_score)
        for race in race_scores:
            race_scores[race] = get_k_score(race_list[race], k)
        preds.append(max(race_scores, key=race_scores.get))
    top_k_dict = {'file': files, 'race_preds': preds}
    return pd.DataFrame(data=top_k_dict)


def filter_sims_dict(sims_dict):
    """Filter similarity dictionary removing race neutral prompts"""
    race_neutral_prompts = [
        "a photo of a man",
        "a photo of a woman",
        "a photo of a young man",
        "a photo of a old man",
        "a photo of a middle-aged man",
        "a photo of a young woman",
        "a photo of a old woman",
        "a photo of a middle-aged woman"]
    filtered_dict = sims_dict.copy()

    for sims_val in filtered_dict.values():
        for rn_prompt in race_neutral_prompts:
            sims_val.pop(rn_prompt, None)

    return filtered_dict


def run(conf):
    """Run the race predictor module"""
    print("Initializing race predictor...")
    print("Prepping output folders...")
    embs_path = system.concat_out_path(conf, 'Embeddings')
    preds_path = system.concat_out_path(conf, 'Predictions')
    sims_path = f"{embs_path}/similarities.json"
    system.prep_folders(preds_path)

    print("Loading data...")
    fface_df = dataloader.load_df(conf['Baseline'])
    sims_dict = dataloader.load_json(sims_path)
    sims_dict = filter_sims_dict(sims_dict)
    print("Starting predictions...")
    avg_df = get_avg_preds(sims_dict)
    final_avg_df = dataloader.generate_final_df(fface_df, avg_df)
    dataloader.save_df(
        df=final_avg_df, out=f"{preds_path}/race_avg_preds.csv")

    if conf['Flags']['multiple-k']:
        max_k = conf['Top K']
        for k in range(1, max_k+1):
            top_df = get_top_k_winner(sims_dict, k)
            final_bin_top_df = dataloader.generate_final_df(fface_df, top_df)
            dataloader.save_df(df=final_bin_top_df,
                               out=f"{preds_path}/" +
                               f"race_top_{str.zfill(str(k), 2)}_synms.csv")
    else:
        k = conf['Top K']
        top_df = get_top_k_winner(sims_dict, k)
        final_bin_top_df = dataloader.generate_final_df(fface_df, top_df)
        dataloader.save_df(df=final_bin_top_df,
                           out=f"{preds_path}/race_top_{k}_synms.csv")
