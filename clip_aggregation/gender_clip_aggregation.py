import torch
from tqdm import tqdm
import pandas as pd
import open_clip

model_name = "ViT-L-14"
data_source = "openai"
preds_path = f"clip_aggregation_preds_{model_name}_{data_source}.csv"
report_path = f"clip_aggregation_report_{model_name}_{data_source}.csv"
embs_path = "./data/fairface/embeddings/openCLIP"
img_path = f"{embs_path}/{model_name}/{data_source}/generated_img_embs.pkl"
val_path = "./data/fairface/fface_val.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"
img_embs = pd.read_pickle(img_path)
fface_df = pd.read_csv(val_path)
model, _, preprocessing = open_clip.create_model_and_transforms(
    model_name, pretrained=data_source, device=device)

clip_templates = [
    'a photo of a {}',
    'a photo of a white {}',
    'a photo of a young white {}',
    'a photo of an old white {}',
    'a photo of a black {}',
    'a photo of a young black {}',
    'a photo of an old black {}',
    'a photo of a latino {}',
    'a photo of a young latino {}',
    'a photo of an old latino {}',
    'a photo of a east asian {}',
    'a photo of a young east asian {}',
    'a photo of an old east asian {}',
    'a photo of a southeast asian {}',
    'a photo of a young southeast asian {}',
    'a photo of an old southeast asian {}',
    'a photo of an indian {}',
    'a photo of a young indian {}',
    'a photo of an old indian {}',
    'a photo of a middle eastern {}',
    'a photo of a young middle eastern {}',
    'a photo of an old middle eastern {}'
]

gender_classes = ['man', 'woman']
preds_classes = ['Male', 'Female']


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


zeroshot_weights, prompts = zeroshot_classifier(gender_classes, clip_templates)
print(prompts)

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
    preds_dict['gender_preds'] = preds

preds_df = pd.DataFrame(data=preds_dict)
new_df = fface_df.set_index('file').join(preds_df.set_index('file'))
new_df.drop(columns=['service_test'], inplace=True)

new_df.to_csv(f"{model_name}_{data_source}_gender_clip_aggregation.csv")
