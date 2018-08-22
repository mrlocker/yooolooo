import numpy as np
import os

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
def get_dir_filelist_by_extension(dir, ext):
    r = os.listdir(dir)
    file_list = []
    for item in r:
        if item.split('.')[-1] == ext:
            file_list.append(dir + '/' + item)
    return file_list

def load_anno_xml(xml_path):
    with open(xml_path) as xmlbuffer:
        xml_content = xmlbuffer.read()
if __name__ == "__main__":
    a = np.array([2,4,5,7,9])
    b = softmax(a)
    if np.max(b)>0.5:
        index = np.argmax(b)
    pass