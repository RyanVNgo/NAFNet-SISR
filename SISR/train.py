
import os

from utils import base_utils
from model import base_model
from data import data_loader


def main():
    print(f"Running {os.path.basename(__file__)}")
    base_utils.base_utils_test()

    model = base_model.BaseModel()
    print(model.call_test())

    data = data_loader.load_data()
    print(data)


if __name__ == "__main__":
    main()

