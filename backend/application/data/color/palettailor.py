import subprocess
import os, json
import numpy as np

def call_nodejs_script(json_path):
    node_script_path = './application/data/color/nodejs/palettailor.js'
    command = ['node', node_script_path, json_path]
    try:
        output = subprocess.check_output(command, universal_newlines=True)
        
        return output.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None

def palettailor(colors, inside_labels, grid_size, grid, labels, seed):
    cur_path = './application/data/color/nodejs/'
    tmp_path = os.path.join(cur_path, 'tmp.json')

    res = {
        'colors': colors,
        'inside_labels': inside_labels,
        'size': grid_size,
        'grids': grid,
        'labels': labels,
        'seed': seed
    }
    with open(tmp_path, 'w') as f:
        json.dump(res, f)
    
    call_nodejs_script(tmp_path)

    output_path = os.path.join(cur_path, 'output.json')
    with open(output_path, 'r') as f:
        output = json.load(f)
   
    os.remove(tmp_path)
    os.remove(output_path)
    # print(output['hues'])
    # print(output['hue_deltas'])
    return output


# palettailor([[255,0,0], [0,255,0]],
#             [[0,1],[2,3]], [2,2], [0,1,2,3],
#             [0,1,2,3], 0)

