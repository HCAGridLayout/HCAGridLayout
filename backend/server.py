from flask import Flask, jsonify, session, g, request
from application.data.port import Port
from application.utils.pickle import *
from flask_cors import *
import os
app = Flask(__name__)
CORS(app, supports_credentials=True)

port = Port(400, {"method": "qap"})
port.load_dataset('imagenet1k_animals') #'MNIST''cifar'

@app.route("/api/metadata", methods=["POST"])
def get_metadata():
    dataset = request.json["dataset"]
    # print("Load {}.".format(dataset))
    ret = {}
    ret["msg"] = "Success"
    return jsonify(ret)


@app.route("/api/gridlayout", methods=["POST"])
def gridlayout():
    import time
    start = time.time()
    node_id = request.json["node_id"]
    gridlayout = {}
    if node_id == -1:
        # fetch top grid layout
        # gridlayout, _ = port.test()
        gridlayout = port.top_gridlayout()
        # gridlayout, _ = port.load_gridlayout(0)
    else:
        # fetch layer grid layout
        samples = request.json["samples"]
        zoom_without_expand = False
        zoom_balance = False
        if "zoom_without_expand" in request.json:
            zoom_without_expand = request.json["zoom_without_expand"]
        if "zoom_balance" in request.json:
            zoom_balance = request.json["zoom_balance"]
        if samples == -1: # zoom out
            gridlayout = port.zoom_out_gridlayout(node_id)
        else:
            gridlayout = port.layer_gridlayout(node_id, samples, zoom_without_expand=zoom_without_expand, zoom_balance=zoom_balance)

    import numpy as np
    for key in gridlayout:
        if isinstance(gridlayout[key], np.ndarray):
            gridlayout[key] = gridlayout[key].tolist()
        elif isinstance(gridlayout[key], dict):
            gridlayout[key] = gridlayout[key].copy()
            for key2 in gridlayout[key]:
                if isinstance(gridlayout[key][key2], np.ndarray):
                    gridlayout[key][key2] = gridlayout[key][key2].tolist()
                elif isinstance(gridlayout[key][key2], dict):
                    gridlayout[key][key2] = gridlayout[key][key2].copy()
                    for key3 in gridlayout[key][key2]:
                        if isinstance(gridlayout[key][key2][key3], np.ndarray):
                            gridlayout[key][key2][key3] = gridlayout[key][key2][key3].tolist()
    if "info_before" in gridlayout:
        del gridlayout["info_before"]
    if "feature" in gridlayout:
        del gridlayout["feature"]

    ret = {}
    ret["msg"] = "Success"
    ret.update(gridlayout)
    ret["dataset_name"] = port.data_set
    ret["ambiguity_threshold"] = port.data_ctrler.ambiguity_threshold
    ret["info_dict"] = port.info_dict
    print("full backend time: ", time.time()-start)
    return jsonify(ret)

@app.route('/api/instances', methods=["POST"])
def instances():
    node_id = request.json["node_id"]
    if "chosen_names" in request.json:
        chosen_names = request.json["chosen_names"]
        return jsonify(port.get_image_instances(node_id, chosen_names))
    return jsonify(port.get_image_instances(node_id))

@app.route('/api/colormap', methods=["POST"])
def colormap():
    # used only for top layer colors
    cache_path = port.data_ctrler.cache_path
    colormap_path = os.path.join(cache_path, "top_colors.pkl")
    
    if request.json["save"]:
        # save_pickle(request.json["colormap"], colormap_path)
        return jsonify({'msg': 'Colormap saved.', 'res': True})
    else:
        if not os.path.exists(colormap_path):
            return jsonify({'msg': 'No colormap found.', 'res': False})
        else:
            return jsonify({
                'msg': 'Colormap loaded.',
                'res': True,
                'map': load_pickle(colormap_path)})

@app.route("/api/reset-gridlayout", methods=["POST"])
def reset_gridlayout():
    import time
    start = time.time()

    mode = request.json["mode"]
    value = request.json["value"]
    # print(mode, value)

    if mode == 0:
        dataset_name = value[0]
        update_info = value[1]
        port.update_setting(info_dict=update_info)
        port.load_dataset(dataset_name)
        gridlayout = port.top_gridlayout()

    if mode == 1:
        gridlayout = port.re_gridlayout(ambiguity_threshold=value)

    import numpy as np
    for key in gridlayout:
        if isinstance(gridlayout[key], np.ndarray):
            gridlayout[key] = gridlayout[key].tolist()
        elif isinstance(gridlayout[key], dict):
            gridlayout[key] = gridlayout[key].copy()
            for key2 in gridlayout[key]:
                if isinstance(gridlayout[key][key2], np.ndarray):
                    gridlayout[key][key2] = gridlayout[key][key2].tolist()
                elif isinstance(gridlayout[key][key2], dict):
                    gridlayout[key][key2] = gridlayout[key][key2].copy()
                    for key3 in gridlayout[key][key2]:
                        if isinstance(gridlayout[key][key2][key3], np.ndarray):
                            gridlayout[key][key2][key3] = gridlayout[key][key2][key3].tolist()
    if "info_before" in gridlayout:
        del gridlayout["info_before"]
    if "feature" in gridlayout:
        del gridlayout["feature"]

    ret = {}
    ret["msg"] = "Success"
    ret.update(gridlayout)
    ret["dataset_name"] = port.data_set
    ret["ambiguity_threshold"] = port.data_ctrler.ambiguity_threshold
    ret["info_dict"] = port.info_dict
    print("full backend time: ", time.time()-start)
    return jsonify(ret)

@app.route("/api/get-setting", methods=["POST"])
def get_setting():
    dataset_name = port.data_set
    ambiguity_threshold = port.data_ctrler.ambiguity_threshold
    info_dict = port.info_dict
    ret = {}
    ret["msg"] = "Success"
    ret["dataset_name"] = dataset_name
    ret["ambiguity_threshold"] = ambiguity_threshold
    ret["info_dict"] = info_dict
    return jsonify(ret)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=12121)
