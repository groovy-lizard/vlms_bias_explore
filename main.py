"""Main script for runing the bias_explorer package
"""

from bias_explorer.utils import dataloader


def main():
    """Reads config file and call the appropriate operations
    """
    conf = dataloader.load_config()
    ops = dataloader.load_operations()
    if conf['Model'] == 'openCLIP':
        for datasource in conf['DataSource']:
            temp_conf = conf.copy()
            temp_conf['DataSource'] = datasource
            for operation in conf['Operations'].items():
                if operation[1]:
                    ops[operation[0]].run(temp_conf)
    else:
        conf['DataSource'] = 'openai'
        for operation in conf['Operations'].items():
            if operation[1]:
                ops[operation[0]].run(conf)
    if conf['Flags']['concatenate']:
        ops['Concatenate'].run(conf)


if __name__ == "__main__":
    main()
