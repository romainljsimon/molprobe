import os 

def fast_scandir(path, dirname):
    subfolders= [f.path for f in os.scandir(path) if (f.is_dir() and dirname in f.path)]
    return subfolders