import os
import pickle

import scipy.io


# Function to load and prepare the data
def load_and_prepare_data(file_name):
    # Load the .mat file
    data_file = get_file_name(file_name, ext='mat')
    mat = scipy.io.loadmat(data_file)

    list = ['tra_X_tr', 'tra_Y_tr', 'tra_X_te', 'tra_Y_te', 'tra_adj_mat'];
    for data_item in list:
        # Extract training and testing data
        data = mat[data_item]

        # Save the data using pickle for later use
        with open(get_file_name(data_item), 'wb') as _:
            pickle.dump(data, _)

    print("Data preparation complete. Data saved as pickle files.")


def get_file_name(filename, ext='pkl'):
    main_path = os.path.dirname(__file__)
    file_path = os.path.join(main_path, f'data/{filename}.{ext}')
    return file_path


if __name__ == '__main__':
    # Load and prepare the data (You need to replace the file path with the actual path to your .mat file)
    file_name = 'traffic_dataset'
    load_and_prepare_data(file_name)
