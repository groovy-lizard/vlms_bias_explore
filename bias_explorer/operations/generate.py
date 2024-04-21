"""Generate embeddings for text or image using the provided model"""

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


def run(conf):
    """Run the generator

    :param conf: conf file loaded from main
    :type conf: dict
    :param model: model object loaded from main
    :type model: dict[obj]
    """
    model = dataloader.load_model(conf)
    print("Initializing generator...")
    prompts, _ = dataloader.load_txts(conf['Labels'])
    img_list = dataloader.load_imgs(conf['Images'])
    root_path = system.make_out_path(conf, 'Embeddings')
    system.prep_folders(root_path)
    img_out = root_path + '/generated_img_embs.pkl'
    txt_out = root_path + '/generated_txt_embs.pt'

    generate_text_embeddings(model, prompts, txt_out)
    generate_image_embeddings(model, img_list, img_out)
