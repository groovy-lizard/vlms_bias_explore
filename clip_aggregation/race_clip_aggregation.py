"""Original CLIP aggregation prediction"""
import json
import gc
import torch
from tqdm import tqdm
import pandas as pd
import open_clip


def load_json(path):
    """Load json from path and returns its data

    :param path: json filepath
    :type path: str
    :return: json file from path
    :rtype: dict
    """
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data


def grab_label_name(label_filename):
    """Grab the label name from the full filename

    :param label_filename: the label filename from conf
    :type label_filename: str
    :return: the label name split from filename
    :rtype: str
    """
    label_name = label_filename.split('/')[-1].split('.')[0]
    return label_name


def create_final_path(conf, root_path, filename):
    """Create the final path to save the file based on root_path"""
    label_name = grab_label_name(conf['Labels'])
    backbone = conf['Backbone']
    data_source = conf['DataSource']
    model_name = conf['Model']
    final_path = f"{root_path}/{model_name}/{backbone}/{data_source}/"
    final_path = final_path + f"{conf['Target']}_{label_name}/"
    final_path = final_path + filename
    print(final_path)
    return final_path


def zeroshot_classifier(model, classnames, templates):
    """Generate zeroshot embeddings using CLIP aggregation"""
    with torch.no_grad():
        zeroshot_weights = []
        prompts = []
        for classname in tqdm(classnames):
            texts = [template.format(classname)
                     for template in templates]  # format with class
            prompts.append(texts)
            texts = open_clip.tokenize(texts).cuda()  # tokenize
            # embed with text encoder
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def run():
    """Run the CLIP aggregation predictions"""
    conf = load_json("./conf.json")
    arch_dict = load_json("./archs_and_datasources.json")
    label_name = grab_label_name(conf['Labels'])
    model_name = conf['Model']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fface_path = conf['Baseline']
    fface_df = pd.read_csv(fface_path)

    temp_dict = {
        "age_race_gender": [
            'a photo of a {} man',
            'a photo of a {} woman',
            'a photo of a young {} man',
            'a photo of a young {} woman',
            'a photo of a middle-aged {} man',
            'a photo of a middle-aged {} woman',
            'a photo of an old {} man',
            'a photo of an old {} woman'
        ],
        "original_clip_labels": [
            'a photo of a {} man',
            'a photo of a {} woman'
        ],
        "raw_race_labels": [
            'a photo of a {} person'
        ],
    }

    race_classes = ["black", "indian", "latino", "white",
                    "middle eastern", "southeast asian", "east asian",]
    preds_classes = ['Black', 'Indian', 'Latino_Hispanic',  'White',
                     'Middle Eastern', 'Southeast Asian', 'East Asian']

    for backbone, datasources in arch_dict.items():
        temp_conf = conf.copy()
        temp_conf['Backbone'] = backbone
        for datasource in datasources:
            model, _, _ = open_clip.create_model_and_transforms(
                backbone, pretrained=datasource, device=device)
            temp_conf['DataSource'] = datasource
            preds_path = create_final_path(
                temp_conf, temp_conf['Predictions'],
                "race_clip_aggregation.csv")
            embs_path = f"{temp_conf['Embeddings']}/{model_name}"
            img_path = f"{embs_path}/{backbone}/{datasource}/"
            img_path = img_path + "generated_img_embs.pkl"
            img_embs = pd.read_pickle(img_path)

            weights = zeroshot_classifier(
                model, race_classes, temp_dict[label_name])
            del model
            gc.collect()
            torch.cuda.empty_cache()

            with torch.no_grad():
                preds_dict = {}
                fnames = []
                preds = []
                for _, emb in img_embs.iterrows():
                    name = emb['file']
                    img_features = emb['embeddings']
                    image_features = torch.from_numpy(img_features).to(device)
                    logits = 100. * image_features @ weights
                    text_probs = logits.softmax(dim=-1)
                    _, top_labels = text_probs.cpu().topk(1, dim=-1)
                    pindex = top_labels.cpu().numpy().item()
                    fnames.append(name)
                    preds.append(preds_classes[pindex])
                preds_dict['file'] = fnames
                preds_dict['race_preds'] = preds

            preds_df = pd.DataFrame(data=preds_dict)
            new_df = fface_df.set_index('file').join(
                preds_df.set_index('file'))
            new_df.drop(columns=['service_test'], inplace=True)

            new_df.to_csv(preds_path)


if __name__ == "__main__":
    run()
