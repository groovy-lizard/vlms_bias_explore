"""Main script for runing the bias_explorer package
"""

from bias_explorer.utils import dataloader


def main():
    """Reads config file and call the appropriate operations
    """
    conf = dataloader.load_config()
    ops = dataloader.load_operations()
    for operation in conf['Operations'].items():
        if operation[1]:
            for datasource in conf['DataSource']:
                temp_conf = conf.copy()
                temp_conf['DataSource'] = datasource
                ops[operation[0]].run(temp_conf)


if __name__ == "__main__":
    main()
