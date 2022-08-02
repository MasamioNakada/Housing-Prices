def txt_list(path):
    with open(path, "r") as f:
        features = " ".join([l.rstrip("\n") for l in f]) 
        return features.split()

