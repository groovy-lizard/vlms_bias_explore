"""MLFoundations openCLIP model handler
"""
import sys
import torch
import open_clip


def model_setup(model_name, data_source):
    """Initialize a pretrained openCLIP model

    :param model_name: Name of the backbone
    :type model_name: str
    :param data_source: Name of the data source where model was trained
    :type data_source: str
    :return: model dictionary with model, preprocess, device and tokenizer
    :rtype: dict[obj]
    """
    available_models = open_clip.list_pretrained()

    if (model_name, data_source) in available_models:
        print(f"Loading model: ({model_name}, {data_source})")
    else:
        print(
            f"ERROR: ({model_name}, {data_source}) not found in ")
        print("open_clip.list_pretrained().")
        sys.exit(-1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocessing = open_clip.create_model_and_transforms(
        model_name, pretrained=data_source, device=device)
    print(f"Done! ({model_name}, {data_source}) loaded to {device} device")
    model_dict = {}
    model_dict = {"Model": model,
                  "Preprocessing": preprocessing,
                  "Device": device,
                  "Tokenizer": open_clip.tokenizer.tokenize}
    return model_dict
