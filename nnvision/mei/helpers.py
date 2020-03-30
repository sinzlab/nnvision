from featurevis.integration import load_pickled_data


def get_neuron_mappings_sessions(dataset_config, key, load_func=None):
    if load_func is None:
        load_func = load_pickled_data
    entities = []
    for i, datafile_path in enumerate(dataset_config["neuronal_data_files"]):
        data = load_func(datafile_path)
        print(data["session_id"])
        if "unit_indices" in data:
            n_neurons = len(data["unit_indices"])
        else:
            n_neurons = data["testing_responses"].shape[1]
        for neuron_pos in range(n_neurons):
            entities.append(dict(key, neuron_id=i*50 + neuron_pos, neuron_position=neuron_pos, session_id=int(data["session_id"])))
    return entities


def get_real_mappings(dataset_config, key, load_func=None):
    if load_func is None:
        load_func = load_pickled_data
    entities = []
    for datafile_path in dataset_config["neuronal_data_files"]:
        data = load_func(datafile_path)
        for neuron_pos, neuron_id in enumerate(data["unit_indices"]):
            entities.append(dict(key, neuron_id=neuron_id, neuron_position=neuron_pos, session_id=data["session_id"]))
    return entities
