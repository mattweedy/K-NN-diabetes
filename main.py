import pandas as pd
import numpy as np

data_file_path = "diabetes.csv"
considered_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

class KNN:
    def __init__(self, training_input, training_output, k):
        # k = num nearest neighbours
        self.training_input = training_input
        self.training_output = training_output
        self.k = k

    def predict(self, test_input):
        predictions = [self._predict(x) for x in test_input]
        return np.array(predictions)

    def _predict(self, x):
        # get distances between x and all examples in the training set
        dist_to_all_points = [calc_euclid_dist(training_input, x) for training_input in self.training_input]
        # get indices of smallest k dists
        indices_nearest_neighbours = np.argpartition(dist_to_all_points, self.k)[:self.k]
        # get the labels of the k nearest neighbour training samples
        label_nearest_neighbours = [self.training_output[i] for i in indices_nearest_neighbours]
        # find most freq label
        most_freq_label = max(set(label_nearest_neighbours), key=label_nearest_neighbours.count)

        return most_freq_label

# load and clean data
def load_and_clean_data(data_file):
    data = pd.read_csv(data_file)
    # replace any 0's with NAN, only in considered cols
    data[considered_cols] = data[considered_cols].replace(0, np.nan)
    # fill blanks with median
    data.fillna(data.median(), inplace=True)

    return data

# calculate euclidean distance
def calc_euclid_dist(row1, row2):
    diff = row1 - row2
    squared_diff = diff ** 2
    sum_squared_diff = np.sum(squared_diff)
    dist = np.sqrt(sum_squared_diff)

    return dist
    
def main():
    data = load_and_clean_data(data_file_path)

    input_data = data.drop("Outcome", axis=1).values
    output_data = data['Outcome'].values

    # split data into 80% for training 20% for testing
    data_training = int(0.80 * len(input_data))
    training_input, test_input = input_data[:data_training], input_data[data_training:]
    training_output, test_output = output_data[:data_training], output_data[data_training:]

    # to store best k value and corresponding accuracy
    best_k = 0
    best_accuracy = 0

    # for testing best output
    for i in range (3, 11):
        # instantiate knn classifier
        knn = KNN(training_input, training_output, i)
        # grab predictions
        preds = knn.predict(test_input)
        # calc accuracy
        accuracy = sum(preds == test_output) / len(test_output)
        print(f"k value  : {i}")
        print(f"accuracy : {accuracy * 100:.5f}%")
        
        # set max accuracy and best k
        best_k, best_accuracy = (i, accuracy) if accuracy > best_accuracy else (best_k, best_accuracy)

    # display best k and max acc
    print("")
    print(f"best k value  : {best_k}")
    print(f"best accuracy : {best_accuracy * 100:.5f}%")

if __name__ == "__main__":
    main()