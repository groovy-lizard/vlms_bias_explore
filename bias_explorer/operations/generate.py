"""Generate embeddings for text or image using the provided model"""

from tqdm import tqdm
import pandas as pd
from PIL import Image
import torch


def generate_image_embeddings(model, img_list):
    """Generate image embeddings from a list of images
    and returns a pandas dataframe with {name of file: img_emb}

    :param model: contains preprocessing, device and model object
    :type model: dict[obj]
    :param img_list: img paths
    :type img_list: list[str]
    :return: df with filenames (str) and embeddings (torch.tensors)
    :rtype: pd.DataFrame
    """

    files = []
    embs = []

    print("Generating embeddings...")
    for file_name in tqdm(img_list):
        img = Image.open(file_name)
        img_input = model["Preprocessing"](
            img).unsqueeze(0).to(model["Device"])

        with torch.nograd():
            image_features = model["Model"].encode_image(img_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        files.append(file_name)
        embs.append(image_features.cpu().numpy())
    d = {'file': files, 'embeddings': embs}

    df_out = pd.DataFrame(data=d)
    return df_out


def generate_text_embeddings(model, txt_list):
    """Generate text embeddings based on a text list

    :param model: dict containing preprocessing, device and model objects
    :type model: dict[obj]
    :param txt_list: a list of text labels to be encoded
    :type txt_list: list[str]
    :return: the embedded text feature vector
    :rtype: torch.tensor
    """
    text_inputs = torch.cat(
        [model["Tokenizer"](c) for c in txt_list]).to(model["Device"])

    with torch.no_grad():
        text_features = model["Model"].encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features
