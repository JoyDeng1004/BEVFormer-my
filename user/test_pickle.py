import pickle
import pprint

file_path = '../data/nuscenes/nuscenes_infos_temporal_val.pkl'  # Adjust the path accordingly

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print("Pickle file is loaded successfully!")
    print("Number of entries:", len(data))
except Exception as e:
    print("Error loading pickle file:", e)

pprint.pprint(data)