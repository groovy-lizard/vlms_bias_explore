"""Main script for runing the bias_explorer package
"""

from bias_explorer.utils import dataloader


def main():
    """Reads config file and call the appropriate operations
    """
    conf = dataloader.load_config()
    ops = dataloader.load_operations()
    for datasource in conf['DataSource']:
        temp_conf = conf.copy()
        temp_conf['DataSource'] = datasource
        for operation in conf['Operations'].items():
            if "Concatenate" not in operation[0]:
                if operation[1]:
                    ops[operation[0]].run(temp_conf)
    if conf['Operations']['Concatenate']:
        ops['Concatenate'].run(conf)


if __name__ == "__main__":
    main()
