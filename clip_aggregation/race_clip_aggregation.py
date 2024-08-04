import torch
from tqdm import tqdm
import pandas as pd
import open_clip

model_name = "ViT-L-14-336"
data_source = "openai"
preds_path = f"clip_aggregation_preds_{model_name}_{data_source}.csv"
report_path = f"clip_aggregation_report_{model_name}_{data_source}.csv"
embs_path = "./data/fairface/embeddings/openCLIP"
img_path = f"{embs_path}/{model_name}/{data_source}/generated_img_embs.pkl"
fface_path = "./data/fairface/fface_val.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"
img_embs = pd.read_pickle(img_path)
fface_df = pd.read_csv(fface_path)
model, _, preprocessing = open_clip.create_model_and_transforms(
    model_name, pretrained=data_source, device=device)

ours_race_clip_templates = [
    'a photo of a {} man',
    'a photo of a {} woman',
    'a photo of a young {} man',
    'a photo of a young {} woman',
    'a photo of a middle-aged {} man',
    'a photo of a middle-aged {} woman',
    'a photo of an old {} man',
    'a photo of an old {} woman'
]

original_race_clip_templates = [
    'a photo of a {} man',
    'a photo of a {} woman'
]

race_classes = ["black", "indian", "latino",
                "middle eastern", "southeast asian", "east asian", "white"]
preds_classes = ['Black', 'Indian', 'Latino_Hispanic',
                 'Middle Eastern', 'Southeast Asian', 'East Asian', 'White']


def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        prompts = []
        for classname in tqdm(classnames):
            texts = [template.format(classname)
                     for template in templates]  # format with class
            prompts.append(texts)
            texts = open_clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(
                texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights, prompts


zeroshot_weights, prompts = zeroshot_classifier(race_classes,
                                                original_race_clip_templates)

with torch.no_grad():
    preds_dict = {}
    fnames = []
    preds = []
    for _, emb in img_embs.iterrows():
        name = emb['file']
        img_features = emb['embeddings']
        image_features = torch.from_numpy(img_features).to(device)
        logits = 100. * image_features @ zeroshot_weights
        text_probs = logits.softmax(dim=-1)
        top_probs, top_labels = text_probs.cpu().topk(1, dim=-1)
        pindex = top_labels.cpu().numpy().item()
        fnames.append(name)
        preds.append(preds_classes[pindex])
    preds_dict['file'] = fnames
    preds_dict['race_preds'] = preds

preds_df = pd.DataFrame(data=preds_dict)
new_df = fface_df.set_index('file').join(preds_df.set_index('file'))
new_df.drop(columns=['service_test'], inplace=True)

new_df.to_csv(f"{model_name}_{data_source}_full_race_clip_aggregation.csv")
