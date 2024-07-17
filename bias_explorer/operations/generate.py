"""Generate embeddings for text or image using the provided model"""
import json
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torch
from ..utils import dataloader
from ..utils import system


def generate_image_embeddings(model, img_list, outf):
    """Generate image embeddings from a list of images
    and returns a pandas dataframe with {name of file: img_emb}

    :param model: contains preprocessing, device and model object
    :type model: dict[obj]
    :param img_list: img paths
    :type img_list: list[str]
    :param outf: output folder
    :type outf: str
    :return: df with filenames (str) and embeddings (torch.tensors)
    :rtype: pd.DataFrame
    """

    files = []
    embs = []

    print("Generating image embeddings...")
    for file_name in tqdm(img_list):
        img = Image.open(file_name)
        img_input = model["Preprocessing"](
            img).unsqueeze(0).to(model["Device"])

        with torch.no_grad():
            image_features = model["Model"].encode_image(img_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        files.append(system.grab_filename(file_name))
        embs.append(image_features.cpu().numpy())
    d = {'file': files, 'embeddings': embs}

    df_out = pd.DataFrame(data=d)
    df_out.to_pickle(outf)
    print(f"Done! Saved pickle file to {outf}")
    return df_out


def generate_text_embeddings(model, txt_list, outf):
    """Generate text embeddings based on a text list

    :param model: dict containing preprocessing, device and model objects
    :type model: dict[obj]
    :param txt_list: a list of text labels to be encoded
    :type txt_list: list[str]
    :param outf: output folder
    :type outf: str
    :return: the embedded text feature vector
    :rtype: torch.tensor
    """
    print("Generating text embeddings...")
    text_inputs = torch.cat(
        [model["Tokenizer"](c) for c in txt_list]).to(model["Device"])

    with torch.no_grad():
        text_features = model["Model"].encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    torch.save(text_features, outf)
    print(f"Done! Saved torch tensor to {outf}")
    return text_features


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


def run(conf):
    """Run the generator

    :param conf: conf file loaded from main
    :type conf: dict
    :param model: model object loaded from main
    :type model: dict[obj]
    """
    print("Initializing generator...")

    flags = conf['Flags']
    model = dataloader.load_model(conf)
    label_name = system.grab_label_name(conf['Labels'])

    prompts = dataloader.load_json(conf['Labels'])
    img_list = dataloader.load_imgs(conf['Images'])
    root_path = system.concat_out_path(conf, 'Embeddings')
    sims_path = f"{root_path}/{label_name}_similarities.json"

    system.prep_folders(root_path)
    img_out = root_path + '/generated_img_embs.pkl'
    txt_out = root_path + f'/{label_name}_generated_txt_embs.pt'
    generate_text_embeddings(model, prompts, txt_out)
    if not flags['text-only']:
        generate_image_embeddings(model, img_list, img_out)
    else:
        print("Skipping image embeddings generation...")
    print("generating similarity scores")
    img_embs, txt_embs = dataloader.load_embs(img_out, txt_out)
    sims_dict = get_sims_dict(model, img_embs, prompts, txt_embs)
    save_sims_dict(sims_dict, dest=sims_path)
