"""Evaluation module for comparisons between text and image embeddings"""
import json
import torch
import numpy as np
import pandas as pd
from ..utils import dataloader
from ..utils import system


def get_similarities(model, img_embeddings, txt_embeddings):
    """Grab similarity between text and image embeddings

    :param model: ict containing preprocessing, device and model objects
    :type model: dict[obj]
    :param img_embeddings: image embeddings
    :type img_embeddings: torch.tensor
    :param txt_embeddings: text embeddings
    :type txt_embeddings: torch.tensor
    :return: similarity between image and text embeddings
    :rtype: float
    """
    image_features = torch.from_numpy(img_embeddings).to(model["Device"])
    similarity = 100.0 * image_features @ txt_embeddings.T

    return similarity


def get_sims_dict(model, img_embeddings, txt_prompts, txt_embeddings):
    """Generate dictionary with filename and similarities
    scores between text prompts

    :param model: dict containig preprocessing, device and model objects
    :type model: dict[obj]
    :param img_embeddings: image embeddings
    :type img_embeddings: torch.tensor
    :param txt_prompts: textual prompts
    :type txt_prompts: json
    :param txt_embeddings: text embeddings
    :type txt_embeddings: torch.tensor
    :return: similarities dictionary
    :rtype: dict
    """
    final_dict = {}
    for _, emb in img_embeddings.iterrows():
        name = emb['file']
        img_features = emb['embeddings']
        img_sims = get_similarities(model, img_features, txt_embeddings)
        s_dict = {}
        for label, score in zip(txt_prompts, img_sims[0]):
            s_dict[label] = score.cpu().numpy().item()
        final_dict[name] = s_dict
    return final_dict


def save_sims_dict(sims_dict, dest):
    """Save similarity dictionary

    :param sims_dict: similarity dictionary to be saved
    :type sims_dict: dict
    :param dest: destination folder path
    :type dest: str
    """
    print('Saving similarities dictionary to ' + dest)
    with open(dest, 'w', encoding='utf-8') as ff:
        json.dump(obj=sims_dict, fp=ff, indent=4)
    print('Done!')


def split_man_woman(label_dict):
    """Split dictionary into man and woman tuples

    :param label_dict: dictionary of similarities
    :type label_dict: dict
    :return: tuple of woman and man list
    :rtype: tuple(list, list)
    """
    woman_list = []
    man_list = []
    for label in label_dict.items():
        key = label[0]
        if "woman" in key:
            woman_list.append(label)
        else:
            man_list.append(label)
    return woman_list, man_list


def get_k_score(tuple_list, k):
    """Get final score of k sum

    :param tuple_list: list of tuples with (label, similarity)
    :type tuple_list: list(tuple)
    :param k: k hyperparameter
    :type k: int
    :return: the sum of the top k similarities
    :rtype: float
    """
    scores = []
    for tup in tuple_list:
        scores.append(tup[1])
    scores = sorted(scores, reverse=True)
    return sum(scores[:k])


def get_top_k_winner(sims_dict, k):
    """Generate dataframe with the winning label of top k method

    :param sims_dict: similarities dictionary
    :type sims_dict: dict
    :param k: k hyperparameter
    :type k: int
    :return: dataframe with filename and final prediciton label
    :rtype: pd.DataFrame
    """
    files = sims_dict.keys()
    wins = []
    for val in sims_dict.values():
        woman_list, man_list = split_man_woman(val)
        w_score = get_k_score(woman_list, k)
        m_score = get_k_score(man_list, k)
        if w_score > m_score:
            wins.append("Female")
        else:
            wins.append("Male")
    top_k_dict = {'file': files, 'gender_preds': wins}
    return pd.DataFrame(data=top_k_dict)


def get_top_synm(sims_dict):
    """Grab most similar synonym

    :param sims_dict: similarities dictionary
    :type sims_dict: dict
    :return: dataframe with filename and gender prediction
    :rtype: pd.DataFrame
    """
    files = sims_dict.keys()
    wins = []
    for val in sims_dict.values():
        scores_list = list(val.values())
        label_list = list(val.keys())
        np_scores = np.asarray(scores_list)
        windex = np.where(np_scores == np_scores.max())[0][0]
        wins.append(label_list[windex])

    top_synm_dict = {'file': files, 'gender_preds': wins}
    return pd.DataFrame(data=top_synm_dict)


def get_man_prompts(prompts):
    """get a list of man prompts

    :param prompts: list of all textual prompts
    :type prompts: list
    :return: a filtered list of man prompts
    :rtype: list
    """
    man_p = []
    for prompt in prompts:
        if "woman" not in prompt:
            man_p.append(prompt)
    return man_p


def synm_to_gender(synm, man_prompts):
    """Mapper function to eval between Male and Female synonyms

    :param synm: the synonym to be evaluated
    :type synm: str
    :param man_prompts: list of male prompts
    :type man_prompts: list
    :return: synonym corresponding gender
    :rtype: str
    """
    if synm in man_prompts:
        return 'Male'
    return 'Female'


def map_synm_to_gender(df, man_prompts):
    """Evaluate and map synms to male or female by checking man prompts

    :param df: dataframe to be evaluated
    :type df: pd.DataFrame
    :param man_prompts: list of man prompts
    :type man_prompts: list
    :return: new dataframe with synonyms and binary gender preds
    :rtype: pd.DataFrame
    """
    new_df = df.copy()
    new_df['synm'] = new_df['gender_preds']
    new_df['gender_preds'] = df['gender_preds'].map(
        lambda x: synm_to_gender(x, man_prompts)
    )
    return new_df


def get_sum_synms(sims_dict, man_prompts):
    """Ensemble over avg sum of similarities
    between male and female synms

    :param sims_dict: similarities dictionary
    :type sims_dict: dict
    :param man_prompts: slice of prompts with only male ones
    :type man_prompts: list
    :return: dataframe with filename and gender prediciton
    :rtype: pd.DataFrame
    """
    files = sims_dict.keys()
    preds = []

    for _, val in sims_dict.items():
        man_score = 0
        woman_score = 0
        for prompt, sim_score in val.items():
            if prompt in man_prompts:
                man_score += sim_score
            else:
                woman_score += sim_score
        preds.append('Male' if man_score > woman_score else 'Female')

    sum_dict = {'file': files, 'gender_preds': preds}
    return pd.DataFrame(data=sum_dict)


def generate_final_df(fface_df, scores_df):
    """Join the winning class df with the original dataset df

    :param fface_df: fairFace dataset df
    :type fface_df: pd.DataFrame
    :param scores_df: predictions df
    :type scores_df: pd.DataFrame
    :return: fairFace dataset df with new 'prediction' column
    :rtype: pd.DataFrame
    """
    new_df = fface_df.set_index('file').join(scores_df.set_index('file'))
    new_df.drop(columns=['service_test'], inplace=True)
    return new_df


def save_df(df, out):
    """Save final dataframe to out folder as csv

    :param df: final fairface dataframe
    :type df: pd.DataFrame
    :param out: output destination folder
    :type out: str
    """
    print(f'Saving dataframe to {out}')
    df.to_csv(out)
    print('Done!')


def run(conf):
    """Run the Evaluator module

    :param conf: config file
    :type conf: dict
    :param model: model utilities object
    :type model: dict[obj]
    """
    print("Initializing predictor...")
    print("Prepping output folders...")
    embs_path = system.concat_out_path(conf, 'Embeddings')
    preds_path = system.concat_out_path(conf, 'Predictions')
    sims_path = f"{embs_path}/similarities.json"
    system.prep_folders(preds_path)

    print("Loading data...")
    prompts, _ = dataloader.load_txts(conf['Labels'])
    fface_df = dataloader.load_df(conf['Baseline'])

    print("Starting predictions...")
    sims_dict = dataloader.load_json(sims_path)

    sum_df = get_sum_synms(sims_dict, get_man_prompts(prompts))
    final_sum_df = generate_final_df(fface_df, sum_df)
    save_df(df=final_sum_df, out=f"{preds_path}/sum_synms.csv")

    if conf['Flags']['multiple-k']:
        max_k = conf['Top K']
        for k in range(1, max_k+1):
            top_df = get_top_k_winner(sims_dict, k)
            final_bin_top_df = generate_final_df(fface_df, top_df)
            save_df(df=final_bin_top_df,
                    out=f"{preds_path}/top_{k}_synms.csv")
    else:
        k = conf['Top K']
        top_df = get_top_k_winner(sims_dict, k)
        final_bin_top_df = generate_final_df(fface_df, top_df)
        save_df(df=final_bin_top_df,
                out=f"{preds_path}/top_{k}_synms.csv")

    print("Predictions finished")
