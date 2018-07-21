import numpy as np


def preprocess(x):
    return ((x/255)-0.5)*2

def softmax(list):
    e1 = np.exp(list)
    d2 = np.sum(e1)
    result = e1/np.sum(d2)
    return result

def load_json(path):
    import json
    with open(path,encoding='UTF-8') as config_buffer:
        config = json.loads(config_buffer.read())
        return config
if __name__ == "__main__":
    a = np.array([2,4,5,7,9])
    b = softmax(a)
    if np.max(b)>0.5:
        index = np.argmax(b)
    pass