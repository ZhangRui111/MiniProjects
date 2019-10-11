import os
import shutil


def exist_delete_or_create_folder(path_name):
    """
    Check whether a path exists,
    if so, delete it;
    if not, then create this path.
    """
    flag = False
    pure_path = os.path.dirname(path_name)
    if os.path.exists(pure_path):
        try:
            shutil.rmtree(pure_path)
        except OSError:
            raise OSError("Error while delete original folder!")
    try:
        os.makedirs(pure_path)
        flag = True
    except OSError:
        pass
    return flag
