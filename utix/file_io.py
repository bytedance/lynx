# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0 

import logging
import shutil
import subprocess
from pathlib import Path
import os

from .logger import Logger

logger = Logger(__name__)


def to_path(in_path):
    """
    if input is a string, then turn it into a Path;
    if input is already a Path, then do nothing.
    otherwise, log error and return original object.
    """

    def to_path_local(in_path):
        if isinstance(in_path, str):
            in_path = Path(in_path)
            return in_path
        elif isinstance(in_path, Path):
            return in_path
        else:
            logger.warning(
                f"to_path(): unsupported input type <{type(in_path)}>, return as it is."
            )
            return in_path

    if isinstance(in_path, list):
        return [to_path_local(x) for x in in_path]
    
    else:
        return to_path_local(in_path)


def check_path(in_path):
    """
    when you call this method, you expect 'in_path' to exist
    therefore when it's not, will log error and return None

    in case you don't want to print anything even 'in_path' is missing,
    you might consider calling "path_exists"
    """
    if in_path is None or in_path == '':
        logging.error(f"check_path: Empty string '' or None input!")
        return None

    if isinstance(in_path, str):
        in_path = Path(in_path)

    if in_path.exists():
        return 'OK'
    else:
        logging.error(f"check_path: Path not exists! <{in_path}>")
        return None


def path_exists(in_path, var_name=""):
    """
    Just checking if the path exists or not, not very critical
    return False and log debug info when path doesn't exist
    """
    if in_path is None or in_path == '':
        return None

    if isinstance(in_path, str):
        in_path = Path(in_path)

    if in_path.exists():
        return True
    else:
        logging.debug(f"path '{var_name}' not exists!")
        logging.debug(f"<{in_path}>")
        return False


DEFAULT_IMAGE_SUFFIXES = \
    ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.tif', '.TIF', '.webp', '.WEBP',
     '.heic', '.HEIC']


def find_files_by_extensions(in_dir, extensions=None, recursive=False):
    """
    Find files by extension (will ignore hidden files)
    searches recursively in the given 'in_dir'

    'extensions' can take single or multiple suffix, will return sorted results
    e.g. ['.JPG', '.jpg']
    e.g. '.png'

    Hidden files are excluded

    glob: Glob the given relative pattern in the directory represented by this path
    rglob: The “**” pattern means “this directory and all subdirectories, recursively"
    """
    if extensions is None:
        extensions = DEFAULT_IMAGE_SUFFIXES.copy()

    if in_dir is None:
        logging.warning(f"No valid files found in target dir <{in_dir}>!")
        return None

    if isinstance(in_dir, str):
        in_dir = Path(in_dir)

    file_list = []
    ext_list = []

    if isinstance(extensions, str):
        # if it's a single string, don't treat 'extensions' as a list
        ext_list.append(extensions)
    else:
        ext_list = extensions

    # print(ext_list)

    for ext in ext_list:
        if recursive:
            files = [x for x in in_dir.rglob(f'*{ext}') if
                     not is_hidden_file(x)]
        else:
            files = [x for x in in_dir.glob(f'*{ext}') if
                     not is_hidden_file(x)]

        file_list.extend(files)

    if not file_list:
        logging.warning(f"No valid files found in target dir <{in_dir}>!")
        return None

    return sorted(file_list)


def rename_files(in_dir, prefix=''):
    """
    This function renames a folder of files into format "prefix0123.jpg"
    *Notice* It does in place renaming, thus we need to put files in tmp folder first,
    so that it can handle the case where 'dst' file name already exists in 'in_dir'
    for example, the 'in_dir' already contains 'prefix0123.jpg'
    it's possible that first rename 'test.jpg' -> 'prefix0123.jpg', then 'prefix0123.jpg' is gone
    file name extensions will be kept as original
    """

    if check_path(in_dir) is None:
        return False

    # create a temp folder for holding renamed files
    tmp_dir = in_dir / "~tmp"
    create_dir(tmp_dir)

    # Iterate through all files from 'in_dir'
    # Move files from 'in_dir' into 'tmp_dir', and rename the files
    index = 0
    for src_file_path in find_all_files(in_dir):
        # print(f"debug:{src_file_path}")
        dst_file_name = generate_filename(prefix, index, src_file_path.suffix)
        dst_file_path = tmp_dir / dst_file_name
        src_file_path.rename(dst_file_path)
        index += 1

    # move files from 'tmp_dir' back into 'in_dir'
    for tmp_file_path in find_all_files(tmp_dir):
        dst_file_path = in_dir / tmp_file_path.name
        tmp_file_path.rename(dst_file_path)

    tmp_dir.rmdir()


def create_dir(the_dir):
    """
    create top directory only
    :param the_dir:
    :return:
    """
    Path(the_dir).mkdir(mode=0o777, parents=False, exist_ok=True)


def create_dirs(the_dir):
    """
    Recursively create folders
    :param the_dir:
    :return:
    """
    if the_dir is None:
        logging.warning(f"cannot create directories: {the_dir}")
        return

    Path(the_dir).mkdir(mode=0o777, parents=True, exist_ok=True)


def is_empty_folder(the_dir):
    """
    if it's an empty folder, return True
    if the folder has content, return False
    if it's a path, also return False
    """
    if isinstance(the_dir, str):
        the_dir = Path(the_dir)

    if the_dir.is_file():
        return False

    is_empty = not any(the_dir.iterdir())
    return is_empty


def delete_dir(the_dir):
    """
    Recursively delete the content of an non-empty folder
    :param the_dir:
    :return:
    """
    logger.warning("Attempting to remove directory {the_dir}", the_dir=the_dir)

    if not path_exists(the_dir):
        # no need to delete
        return

    for sub in the_dir.iterdir():
        if sub.is_dir():
            delete_dir(sub)
        else:
            sub.unlink()

    the_dir.rmdir()  # if you just want to delete dir content, remove this line


def find_all_folders(root_dir, return_str=False):
    """
    return a list of folder names inside 'root_dir'
    :param root_dir:
    :return:
    """
    folder_list = Path(root_dir).glob('*')
    
    if return_str:
        folders = [str(x) for x in folder_list if x.is_dir() and not is_hidden_file(x)]
    else:
        # return Path instances
        folders = [x for x in folder_list if x.is_dir() and not is_hidden_file(x)]

    return sorted(folders)


def find_all_files(in_dir):
    """
    Get all files, while excluding hidden files which starts with '.' (for Mac)
    """

    in_dir = to_path(in_dir)
    file_list = in_dir.glob('**/*')
    files = [x for x in file_list if x.is_file() and not is_hidden_file(x)]

    return sorted(files)


def generate_filename(prefix, index, suffix, fill_width=4):
    """
    input example: prefix = 'night', index = 3, suffix = '.jpg', fill_width = 4
    output: 'night_0003.jpg'
    """
    idx_str = str(index).zfill(fill_width)
    return f"{prefix}_{idx_str}{suffix}"


def is_hidden_file(filepath):
    if any([p for p in filepath.resolve().parts if p.startswith(".")]):
        return True
    else:
        return False


def copy_file(src_file_path, dst_file_path):
    if check_path(src_file_path) is None:
        return False

    create_dirs(Path(dst_file_path).parent)

    shutil.copy(str(src_file_path), str(dst_file_path))


def copy_files(src_dir, dst_dir):
    """
    will create 'dst_dir' folder if it doesn't already exist
    """
    logging.info(f"Copying files from {src_dir} to {dst_dir}")

    src_dir = to_path(src_dir)
    dst_dir = to_path(dst_dir)

    if check_path(src_dir) is None:
        logging.warning("failed to copy files!")
        return False

    # create output directory if not already exist
    create_dirs(dst_dir)

    if src_dir.samefile(dst_dir):
        logging.error(f"'src_dir' equals 'dst_dir': {src_dir}")
        return False

    for filename in src_dir.iterdir():
        if filename.is_file():
            src_path = src_dir / filename.name
            dst_path = dst_dir / filename.name
            # move files
            shutil.copy(str(src_path), str(dst_path))

    return True


def find_file_and_get_relative_path(file_name, middle_dirs=[], outer_searching_level=2):
    """
    Given file_name, return relative path from your working dir (where you execute your program)
    . In additionally, increase the accuracy by providing
    the middle dir name you sure it will passed in case there's some files have the
    same time.
    """

    middle_dirs_list = []
    if isinstance(middle_dirs, str):
        middle_dirs_list.append(middle_dirs)
    else:
        middle_dirs_list = middle_dirs

    working_dir = Path.cwd()
    searching_dir = working_dir
    for i in range(outer_searching_level):
        searching_dir = working_dir.parent
    name_match_files = find_files_by_extensions(searching_dir, str(file_name),
                                                recursive=True)
    middle_dir_match_files = []
    for path in name_match_files:
        for m in middle_dirs:
            if m in str(path):
                path = os.path.relpath(str(path), str(working_dir))
                middle_dir_match_files.append(path)

    try:
        ## TODO: should sort the result base on how many middle dir it passed
        return middle_dir_match_files[0]
    except:
        logging.warning(f"not matching files found")
        return ""


def get_relative_path(in_path, outer_searching_level=2):
    """
    Given file_name, return relative path from your working dir (where you execute your program)
    """
    working_dir = Path.cwd()
    searching_dir = working_dir
    for i in range(outer_searching_level):
        searching_dir = working_dir.parent

    name_match_files = find_files_by_extensions(searching_dir, str(in_path),
                                                recursive=True)
    path = os.path.relpath(str(name_match_files[0]), str(working_dir))
    return path


def move_files(src_dir, dst_dir):
    """
    Move all files (non-hidden) from 'src_dir' into 'dst_dir'
    will create 'dst_dir' folder if it doesn't already exist
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    # create output directory if not already exist
    create_dirs(dst_dir)

    index = 0
    for src_file_path in find_all_files(src_dir):
        # print(f"debug:{src_file_path}")
        dst_file_path = dst_dir / src_file_path.name
        src_file_path.rename(dst_file_path)
        index += 1


def move_file(src_file_path, dst_file_path):
    """
    """
    if check_path(src_file_path) is None:
        logging.warning(f"failed to move file: {src_file_path}!")
        return False

    src_file_path.rename(dst_file_path)
    return True
