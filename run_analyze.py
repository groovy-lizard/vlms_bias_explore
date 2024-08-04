from bias_explorer.utils import dataloader
from bias_explorer.operations import analyze

conf = dataloader.load_json("./conf.json")
analyze.run(conf)
