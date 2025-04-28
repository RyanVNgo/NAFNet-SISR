
import os


def model_save_path(dir_path, stemname):
    dir_path = os.path.abspath(dir_path)
    if os.path.exists(dir_path) == False:
        dir_path = os.path.abspath(os.getcwd())

    filename = stemname + '.pth'
    save_path = os.path.join(dir_path, filename)

    index = 1;
    while os.path.exists(save_path):
        filename = f'{stemname}_({index}).pth'
        save_path = os.path.join(dir_path, filename)
        index += 1

    return save_path



def config_save_path(model_save_path):
    dir = os.path.dirname(model_save_path)
    filename = os.path.basename(model_save_path)
    filename = os.path.splitext(filename)[0] + '.yaml'
    config_save_path = os.path.join(dir, filename)
    return config_save_path


