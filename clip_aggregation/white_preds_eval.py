import pandas as pd
from sklearn.metrics import accuracy_score

model_name = "ViT-L-14-336"
data_source = "openai"
preds_path = "./data/fairface/predictions/CLIP"

# read predictions csv file
preds_df = pd.read_csv(
    "./ViT-L-14-336_openai_race_clip_aggregation.csv")

# dictionary with all races and their replacements
replace_dict = {
    "East Asian": "Non-White",
    "White": "White",
    "Latino_Hispanic": "Non-White",
    "Southeast Asian": "Non-White",
    "Black": "Non-White",
    "Indian": "Non-White",
    "Middle Eastern": "Non-White"
}

# binarize 'race' column
preds_df = preds_df.replace({"race": replace_dict, "race_preds": replace_dict})

# create filtered dataframes with only white and only non-white images
white_df = preds_df[preds_df['race'] == 'White']
non_white_df = preds_df[preds_df['race'] != 'White']

# measure accuracy for white and non-white images
white_acc = round(accuracy_score(white_df['race'], white_df['race_preds']), 4)
non_white_acc = round(accuracy_score(
    non_white_df['race'], non_white_df['race_preds']), 4)

print("7 races prediction")
print(
    "Percent accuracy on Race classification of images in category 'White':")
print(white_acc)

print(
    "Percent accuracy on Race classification of images in categories grouped as 'Non-White':")
print(non_white_acc)

# read binary predictions csv file
binary_preds = pd.read_csv(
    "./ViT-L-14-336_openai_binary_race_clip_aggregation.csv"
)

# binarize 'race' column
binary_preds = binary_preds.replace({"race": replace_dict})

# create filtered dataframes with only white and only non-white images
binary_white_df = binary_preds[binary_preds['race'] == 'White']
binary_non_white_df = binary_preds[binary_preds['race'] != 'White']

# measure accuracy for white and non-white dataframes
binary_white_acc = round(accuracy_score(
    binary_white_df['race'], binary_white_df['race_preds']), 4)
binary_non_white_acc = round(accuracy_score(
    binary_non_white_df['race'], binary_non_white_df['race_preds']), 4)

print("2 races prediction (White, Non-White)")
print(
    "Percent accuracy on Race classification of images in category 'White':")
print(binary_white_acc)
print(
    "Percent accuracy on Race classification of images in categories grouped as 'Non-White':")
print(binary_non_white_acc)
