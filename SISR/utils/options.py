
import yaml


def parse_options(path):
    with open(path, mode='r') as opt_file:
        options = yaml.safe_load(opt_file)
    return options

