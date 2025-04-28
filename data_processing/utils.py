import pickle

import obonet

def save_pickle(obj, filepath):
    try:
        with open(filepath, 'wb') as file:
            pickle.dump(obj, file)
        print(f"Object successfully saved to {filepath}.")
    except Exception as e:
        print(f"Error saving object: {e}")

def load_pickle(filepath):
    try:
        with open(filepath, 'rb') as file:
            obj = pickle.load(file)
        print(f"Object successfully loaded from {filepath}.")
        return obj
    except Exception as e:
        print(f"Error loading object: {e}")
        return None
    
    
def load_graph():
    go_graph = obonet.read_obo(open("/home/fbqc9/Workspace/MCLLM_DATA/obo/go.obo", 'r'))

    accepted_edges = set()
    unaccepted_edges = set()

    for edge in go_graph.edges:
        if edge[2] == 'is_a' or edge[2] == 'part_of':
            accepted_edges.add(edge)
        else:
            unaccepted_edges.add(edge)
    go_graph.remove_edges_from(unaccepted_edges)
    return go_graph