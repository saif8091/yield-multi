import os

meteo_list = ['gdd','gdd_harvest','evap','evap_harvest']
#Initialize an empty dictionary to store the features
features_dict = {}

# List of files (assuming 'files' is defined elsewhere in your code)
files = os.listdir('feature_filter/filtered_features/')

for file in files:
    # Skip directories
    if os.path.isdir('feature_filter/filtered_features/' + file):
        continue

    # Read the selected features from the file
    with open('feature_filter/filtered_features/' + file, 'r') as f:
        selected_feat = [line.strip() for line in f]
    
    # Remove anything after '.' from the file name to use as the key
    key = file.split('.')[0]
    
    # Add the selected features to the dictionary with the modified file name as the key
    features_dict[key] = selected_feat