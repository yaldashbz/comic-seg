import json


def open_json(file_name):
    f = open(file_name, 'r')
    json_obj = json.load(f)
    f.close()
    return json_obj


def save_json(file_name, json_obj):
    f = open(file_name, 'w')
    json.dump(json_obj, f)
    f.close()
