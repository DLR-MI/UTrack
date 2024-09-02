from tqdm import tqdm
from pathlib import Path
import itertools
import json

import numpy as np
from omegaconf import OmegaConf, DictConfig, ListConfig


HP_SEARCH_PATH = Path(__file__).parent


def extract_values(data, combinations_list):
    # Check if the input is a dictionary
    if isinstance(data, (DictConfig, dict)):
        # Check if "type" key is present
        if "type" in data:
            if data["type"] == "continuous":
                min_val = data['min']
                max_val = data['max']
                step = data['step']
                n = int((max_val - min_val) / step)
                num_points = n + 1
                values_array = list(np.linspace(min_val, min_val + n * step, num_points))
                # Append the combination array to the combinations_list list for 'continuous' variables
                combinations_list.append((values_array, "continuous"))
            # Append the "values" to the combinations_list list for 'static' and 'discrete' variables
            elif "values" in data:
                combinations_list.append((data["values"], data["type"]))
        else:
            # Recursively call the function for values in the dictionary
            for key, value in data.items():
                extract_values(value, combinations_list)

    else:
        raise KeyError("Check configuration file structure, must be of type dictionary.")

    return combinations_list


def get_total_combinations(input_list):
    # Find the indices of lists in the input_list
    list_indices = [i for i, item in enumerate(input_list) if isinstance(item[0], (list, ListConfig))]

    # Calculate the total number of combinations
    total_combinations = 1
    for i, item in enumerate(input_list):
        if i in list_indices and item[1] != "static":
            total_combinations *= len(item[0])

    return total_combinations


def generate_combinations(input_list):
    # Find the indices of lists in the input_list
    list_indices = [
        i for i, item in enumerate(input_list) if isinstance(item[0], (list, ListConfig))
        and not item[1] == "static"
    ]

    # Generate all combinations of the list elements at list_indices
    combinations = itertools.product(*[
        input_list[i][0] if i in list_indices else [input_list[i][0]] for i in range(len(input_list))
    ])

    # Convert combinations to a list of lists
    result = [list(combo) for combo in tqdm(combinations)]

    return result


def generate_yaml(data, combinations_list):
    if isinstance(data, (dict, DictConfig)):
        if "type" in data:
            if isinstance(combinations_list[0], np.float64):
                data = float(combinations_list[0])
            elif isinstance(combinations_list[0], np.int64):
                data = int(combinations_list[0])

            else:
                data = combinations_list[0]
            combinations_list.pop(0)

        else:
            for key, value in data.items():
                data[key] = generate_yaml(value, combinations_list)
    else:
        raise KeyError("Check configuration file structure, must be of type dictionary.")
                
    return data


def generate_config(association='byte', gpu_id=0, template='hp_search'):
    config_data = OmegaConf.load(HP_SEARCH_PATH / f'{template}.yaml')
    if association is not None:
        config_data.tracker.association['values'] = association
    config_data.yolo.device['values'] = gpu_id
    possibilities_list = extract_values(config_data, [])
    total_combinations = get_total_combinations(possibilities_list)
    
    estimated_runtime = total_combinations * 6 / 60  # Assuming ~6 minutes per run
    print(f'Estimated runtime: {estimated_runtime:.1f} hours or {estimated_runtime/24:.1f} days')
    
    print('Generating combinations...')
    combinations_list = generate_combinations(possibilities_list)

    for combination in tqdm(combinations_list):
        # Create a copy of the template configuration
        config = OmegaConf.create(config_data)
        yaml_data = generate_yaml(config, combination) 
        yield json.dumps(OmegaConf.to_container(yaml_data))  


if __name__ == '__main__':
    for config in tqdm(generate_config()):
        continue