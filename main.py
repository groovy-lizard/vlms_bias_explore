"""Main script for runing the bias_explorer package
"""

from bias_explorer.utils import dataloader


def run_all_models(conf, ops):
    """Run operations on all target backbones"""
    arch_dict = dataloader.load_json("./archs_and_datasources.json")
    for backbone, datasources in arch_dict.items():
        temp_conf = conf.copy()
        temp_conf['Backbone'] = backbone
        for datasource in datasources:
            temp_conf['DataSource'] = datasource
            for operation in conf['Operations'].items():
                if operation[1]:
                    ops[operation[0]].run(temp_conf)
    if conf['Flags']['analyze']:
        for backbone, datasources in arch_dict.items():
            temp_conf = conf.copy()
            temp_conf['Backbone'] = backbone
            for datasource in datasources:
                temp_conf['DataSource'] = datasource
                ops['Analyze'].run(temp_conf)


def run_all_sources(conf, ops):
    """Run operations on single model but all target datasources"""
    arch_dict = dataloader.load_json("./archs_and_datasources.json")
    for datasource in arch_dict[conf['Backbone']]:
        temp_conf = conf.copy()
        temp_conf['DataSource'] = datasource
        for operation in conf['Operations'].items():
            if operation[1]:
                ops[operation[0]].run(temp_conf)


def run_openai_clip(conf, ops):
    """Run operations using CLIP model"""
    conf['DataSource'] = 'openai'
    for operation in conf['Operations'].items():
        if operation[1]:
            ops[operation[0]].run(conf)


def run_open_clip(conf, ops):
    """Run operations using openCLIP model"""
    for datasource in conf['DataSource']:
        temp_conf = conf.copy()
        temp_conf['DataSource'] = datasource
        for operation in conf['Operations'].items():
            if operation[1]:
                ops[operation[0]].run(temp_conf)


def main():
    """Reads config file and call the appropriate operations"""
    conf = dataloader.load_json("./conf.json")
    ops = dataloader.load_operations()
    if conf['Model'] == 'openCLIP':
        if conf['Flags']['all-models']:
            run_all_models(conf, ops)
        elif conf['Flags']['all-sources']:
            run_all_sources(conf, ops)
        else:
            run_open_clip(conf, ops)
    else:
        run_openai_clip(conf, ops)


if __name__ == "__main__":
    main()
