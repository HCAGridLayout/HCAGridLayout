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
    print("Load {}.".format(dataset))
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
        if "zoom_without_expand" in request.json:
            zoom_without_expand = request.json["zoom_without_expand"]
        if samples == -1: # zoom out
            gridlayout = port.zoom_out_gridlayout(node_id)
        else:
            gridlayout = port.layer_gridlayout(node_id, samples, zoom_without_expand=zoom_without_expand)

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

    ret = {}
    ret["msg"] = "Success"
    ret.update(gridlayout)
    ret["dataset_name"] = port.data_set
    ret["ambiguity_threshold"] = port.data_ctrler.ambiguity_threshold
    ret["info_dict"] = port.info_dict
    print("backend time", time.time()-start)
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

import math
import numpy as np
## The method has been abandoned.
# at least 4 * 4, more than 4 cells
def random_zoom_in(grid, labels):
    grid_size = len(grid)
    grid_size = int(math.sqrt(grid_size))
    grid_num = len(labels)
    selected = []
    while len(selected) < 4:
        start_x = np.random.randint(0, grid_size - 3)
        start_y = np.random.randint(0, grid_size - 3)
        len_x = np.random.randint(3, grid_size - start_x)
        len_y = np.random.randint(3, grid_size  - start_y)
        selected = []
        for i in range(start_x, start_x + len_x):
            for j in range(start_y, start_y + len_y):
                cell = grid[i * grid_size + j]
                if cell < grid_num:
                    selected.append(cell)
        with open("log.txt", "a") as f:
            f.write(str((start_x, start_y, len_x, len_y, len(selected))) + "\n")

    return selected

iter_num = 10
@app.route('/api/evaluate_script', methods=["POST"])
def evaluate_script():
    mode = request.json["mode"]
    print(mode)
    colors = []
    pcolors = []
    times = []
    if mode == "top":
        port.color_map.extend_mode = "tree"
        res, _ = port.load_gridlayout(0)
        colors.append(res["colors"])
        pcolors.append(res["pcolors"])
        times.append(0)

        port.color_map.extend_mode = "colorlib"
        port.color_map.colorlib_type = "d"
        for i in range(iter_num):
            res, _ = port.load_gridlayout(0)
            colors.append(res["colors"])
            pcolors.append(res["pcolors"])
        avg, std = port.color_map.getTime(iter_num)
        times.append([avg, std])
        port.color_map.colorlib_type = "s"
        for i in range(iter_num):
            res, _ = port.load_gridlayout(0)
            colors.append(res["colors"])
            pcolors.append(res["pcolors"])
        avg, std = port.color_map.getTime(iter_num)
        times.append([avg, std])

        port.color_map.extend_mode = "ramp"
        res, _ = port.load_gridlayout(0)
        colors.append(res["colors"])
        pcolors.append(res["pcolors"])
        times.append(port.color_map.ramp.total_time)

        port.color_map.extend_mode = "palettailor"
        for i in range(iter_num):
            port.color_map.seed = 5 * (i + 1)
            res, _ = port.load_gridlayout(0)
            colors.append(res["colors"])
            pcolors.append(res["pcolors"])
        avg, std = port.color_map.getTime(iter_num)
        times.append([avg, std])
        return jsonify({
            "colors": colors,
            "pcolors": pcolors,
            "times": times,
            # "example": res,
        })
    
    if mode == "gen":
        port.save_port_for_eval = True
        port.color_map.extend_mode = "ramp"
        res = port.clear_port_cache()
        res, extra = port.top_gridlayout()
        total_num = [0]
        # clear log
        with open("log.txt", "w") as f:
            f.write("")
        from IPython import embed
        embed()
        return

        def zoom_in(gridlayout, extra_info, total_num, layer_num=0):
            if len(total_num) <= layer_num:
                total_num.append(0)
            total_num[layer_num] += 1

            top_labels = extra_info["top_labels"]
            if np.unique(top_labels).shape[0] == 1:
                return
            for label in np.unique(top_labels):
                samples = np.where(top_labels == label)[0]
                grid_res, grid_extra = port.layer_gridlayout(gridlayout["index"], samples)
                with open("log.txt", "a") as f:
                    f.write("Zoom in" + "\n")
                zoom_in(grid_res, grid_extra, total_num, layer_num + 1)
                port.zoom_out_gridlayout(grid_res["index"])
                with open("log.txt", "a") as f:
                    f.write("Zoom out" + "\n")

        zoom_in(res, extra, total_num)
        print(total_num)
        port.save_port_for_eval = False
        return jsonify({
            "msg": "Success",
        })
    if mode == "zoom":
        import time
        save_path = os.path.join(port.port_path, "result.pkl")
        if os.path.exists(save_path):
            import pickle
            with open(save_path, "rb") as f:
                answer = pickle.load(f)
        else:
            levels = []
            timeout = 10
            records = [[None], [None], [None], [None], [None]]
            cur_level = -1
            iszoomin = True
            for i in range(port.save_id):
                level = port.get_level(i)
                print(cur_level)
                if level > cur_level:
                    iszoomin = True
                else:
                    iszoomin = False
                    if len(records[0]) > 2:
                        for i in range(5):
                            records[i] = records[i][:2]
                cur_level = level

                port.color_map.extend_mode = "tree"
                res = port.load_gridlayout(i, records[0][-1])
                colors.append(res["colors"])
                pcolors.append(res["pcolors"])
                if iszoomin:
                    records[0].append(res["records"])
                times.append(0)

                port.color_map.extend_mode = "colorlib"
                port.color_map.colorlib_type = "d"
                res = port.load_gridlayout(i, records[1][-1])
                colors.append(res["colors"])
                pcolors.append(res["pcolors"])
                avg, std = port.color_map.getTime(1)
                times.append(avg)
                if iszoomin:
                    records[1].append(res["records"])

                port.color_map.colorlib_type = "s"
                res = port.load_gridlayout(i, records[2][-1])
                colors.append(res["colors"])
                pcolors.append(res["pcolors"])
                avg, std = port.color_map.getTime(1)
                times.append(avg)
                if iszoomin:
                    records[2].append(res["records"])

                port.color_map.extend_mode = "ramp"
                res = port.load_gridlayout(i, records[3][-1])
                colors.append(res["colors"])
                pcolors.append(res["pcolors"])
                times.append(port.color_map.ramp.total_time)
                if iszoomin:
                    records[3].append(res["records"])

                port.color_map.extend_mode = "palettailor"
                res = port.load_gridlayout(i, records[4][-1])
                colors.append(res["colors"])
                pcolors.append(res["pcolors"])
                avg, std = port.color_map.getTime(1)
                times.append(avg)
                if iszoomin:
                    records[4].append(res["records"])

                levels.append(cur_level)
                time.sleep(1)
                
            answer = {
                "colors": colors,
                "pcolors": pcolors,
                "times":times,
                "levels": levels
            }
            
            import pickle
            with open(save_path, "wb") as f:
                pickle.dump(answer, f)
        
        return jsonify(answer)

@app.route("/api/reset-gridlayout", methods=["POST"])
def reset_gridlayout():
    import time
    start = time.time()

    mode = request.json["mode"]
    value = request.json["value"]
    print(mode, value)

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

    ret = {}
    ret["msg"] = "Success"
    ret.update(gridlayout)
    ret["dataset_name"] = port.data_set
    ret["ambiguity_threshold"] = port.data_ctrler.ambiguity_threshold
    ret["info_dict"] = port.info_dict
    print("backend time", time.time()-start)
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
