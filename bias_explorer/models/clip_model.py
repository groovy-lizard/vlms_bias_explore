"""OpenAI original CLIP model handler"""
import sys
import torch
import clip


def model_setup(model_name):
    """Initial loading of CLIP model

    :param model_name: Backbone name. (See list at clip.available_models())
    :type model_name: str
    :return: model, preprocessing and device objects
    :rtype: dict
    """
    available_models = clip.available_models()

    if model_name in available_models:
        print(f"Loading model: {model_name}")
    else:
        print(f"ERROR: {model_name} not found in clip.available_models().")
        sys.exit(-1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocessing = clip.load(model_name, device=device, jit=False)
    print(f"Done! {model_name} model loaded to {device} device")
    model_dict = {"Model": model,
                  "Preprocessing": preprocessing,
                  "Device": device}
    return model_dict
