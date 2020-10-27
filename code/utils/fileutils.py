import os, joblib, errno

def check_path(path):
    try:
        os.makedirs(path)  # Support multi-level
        print(path + ' created')
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
        # print(path, ' exists')
    return path
