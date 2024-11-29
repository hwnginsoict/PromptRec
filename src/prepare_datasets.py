import zipfile
import collections
import os
import random
import sys

from models.utils import frozen


SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42

def split_dataset(target, train_rate=250, valid_rate=50):
    frozen(SEED)
    with open(target + "/full.tsv", encoding="utf8") as f:
        for size, _ in enumerate(f, 1):
            pass
    if isinstance(train_rate, float):
        train_rate = int(train_rate * size)
    if isinstance(valid_rate, float):
        valid_rate = int(valid_rate * size)
    assert train_rate + valid_rate <= size
    recordlist = list(range(size))
    random.shuffle(recordlist)
    validset = set(recordlist[train_rate:train_rate + valid_rate])
    testset = set(recordlist[train_rate + valid_rate:])
    with open(target + "/full.tsv", encoding="utf8") as full,\
         open(target + "/train.tsv", "w", encoding="utf8") as train,\
         open(target + "/valid.tsv", "w", encoding="utf8") as valid,\
         open(target + "/test.tsv", "w", encoding="utf8") as test:
        #full.readline() # skip the column line
        for idx, row in enumerate(full):
            if idx in validset:
                valid.write(row)
            elif idx in testset:
                test.write(row)
            else:
                train.write(row)


def time_split_dataset(target, train_rate=250, valid_rate=50, seg="\t", pos=-1):
    with open(target + "/full.tsv", encoding="utf8") as f:
        recordlist = f.readlines()
    if isinstance(train_rate, float):
        train_rate = int(train_rate * size)
    if isinstance(valid_rate, float):
        valid_rate = int(valid_rate * size)
    assert train_rate + valid_rate <= len(recordlist)
    recordlist.sort(key=lambda x: int(x.split(seg)[pos].strip()))
    with open(target + "/train.tsv", "w", encoding="utf8") as train,\
         open(target + "/valid.tsv", "w", encoding="utf8") as valid,\
         open(target + "/test.tsv", "w", encoding="utf8") as test:
        for idx, row in enumerate(recordlist):
            if idx < train_rate:
                train.write(row)
            elif idx < train_rate + valid_rate:
                valid.write(row)
            else:
                test.write(row)



def shuffle_dataset(data):
    frozen(SEED)
    with open(data, encoding="utf8") as f:
        records = f.readlines()
    random.shuffle(records)
    with open(data, 'w', encoding="utf8") as f:
        for row in records:
            f.write(row)


def clean_dataset_dir(target):
    valid = {"full.tsv", "train.tsv", "test.tsv", "valid.tsv",
              "item_idx.txt", "item_meta.txt", "user_idx.txt", "user_meta.txt"}
    for fpath in os.listdir(target):
        if fpath not in valid:
            os.remove(target + "/" + fpath)   
                

def prepare_movielen100k(source, target="./"):
    target = os.path.abspath(target).replace("\\", "/")
    with zipfile.ZipFile(source) as f:
        for old, new in zip(["u.user", "u.item", "ua.base", "ua.test"],
                        ["user_meta.txt", "item_meta.txt", "train.tsv", "test.tsv"]):
            f.extract("ml-100k/%s" % old, target)
            with open(target + "/ml-100k/%s" % old, encoding="ISO-8859-1") as fin,\
                 open(target + "/ml-100k/%s" % new, "w", encoding="utf8") as fout:
                for row in fin:
                    if "meta" in new:
                        row = row.replace("|", "\t")
                    fout.write(row)
    for node in ["user", "item"]:
        with open(target + "/ml-100k/%s_meta.txt" % node, encoding="utf8") as f,\
             open(target + "/ml-100k/%s_idx.txt" % node, "w", encoding="utf8") as t:
            for size, row in enumerate(f, 1):
                t.write(str(size) + "\n")

    with open(target + "/ml-100k/item_meta.txt", encoding="utf8") as src,\
         open(target + "/ml-100k/item_meta.txt.tmp", "w", encoding="utf8") as tgt:
        categories = ["action", "adventure", "animated", "kid", "comedy", "criminal", "documentary", "dramatic",
                      "fantasy", "dark", "horrible", "musical", "mystical", "romatic", "scientific", "horrible",
                      "war", "western"]
        for row in src:
            features = row[:-1].split("\t")
            movie_cate = [c for f, c in zip(features, categories) if f == "1"]
            for_child = 0
            if "kid" in movie_cate:
                del movie_cate[movie_cate.index("kid")]
                for_child = 1
            movie_cate = "|".join(movie_cate)
            tgt.write("\t".join(features[:3] + [movie_cate, str(for_child)]) + '\n')
    os.remove(target + "/ml-100k/item_meta.txt")
    os.rename(target + "/ml-100k/item_meta.txt.tmp", target + "/ml-100k/item_meta.txt")

    with open(target + "/ml-100k/full.tsv", "w", encoding="ISO-8859-1") as fout:
        for subset in ["train", "test"]:
            with open(target + "/ml-100k/%s.tsv" % subset, encoding="utf8") as fin:
                for row in fin:
                    fout.write(row)
    split_dataset(target + "/ml-100k/")
    clean_dataset_dir(target + "/ml-100k")
    shuffle_dataset(target + "/ml-100k/train.tsv")
    
if __name__ == "__main__":
    prepare_movielen100k("../datasets/downstream_tasks/ml-100k.zip",
                         "../datasets/downstream_tasks/")

