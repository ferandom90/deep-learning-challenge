# deep-learning-challenge

The code starts by importing necessary libraries: pandas, sklearn, and tensorflow.
The dataset "charity_data.csv" is loaded into a Pandas DataFrame called application_df.
Initial Data Exploration:

The .head() function is used to display the first few rows of the dataset.
.nunique() is called to determine the number of unique values in each column of the DataFrame.
Drop Non-Beneficial Columns:
The "EIN" column is dropped from the DataFrame since it likely won't contribute to the prediction.
Further Data Exploration:
.nunique() is used again to check the number of unique values in each column after dropping the "EIN" column.
Application Type Binning:
The code examines the distribution of values in the "APPLICATION_TYPE" column using value_counts().
A cutoff value of 160 is chosen. Application types with counts below this cutoff are replaced with "Other".
Classification Binning:
Similar to the previous step, the code analyzes the distribution of values in the "CLASSIFICATION" column.
Application classifications with counts above 1 and below a cutoff value of 1800 are replaced with "Other".
One-Hot Encoding:
Categorical data in the DataFrame is converted into numeric format using pd.get_dummies().
Data Splitting:
The dataset is split into features (X) and target (y) arrays.
The data is further split into training and testing sets using train_test_split().
Standard Scaling:
A StandardScaler is created and fitted to the training data.
The training and testing data are then scaled using the fitted scaler.
Neural Network Model Definition:
A sequential neural network model is defined using TensorFlow's Keras API.
The model consists of an input layer, two hidden layers with 160 units each and 'sigmoid' activation function, and an output layer with 1 unit and 'sigmoid' activation function.
Model Compilation:
The model is compiled with an Adam optimizer, 'binary_crossentropy' loss function, and 'accuracy' metric.
Model Training:
The model is trained on the scaled training data using the fit() function.
Training is performed for 5 epochs with validation on the scaled testing data.
Model Evaluation:
The trained model is evaluated using the scaled testing data.
The loss and accuracy are printed to the console.
Model Saving:
The trained model is saved to an HDF5 file named "AlphabetSoupCharity_Optimization.h5".

Overall, this code performs data preprocessing by handling categorical variables through one-hot encoding, implements feature engineering by binning less frequent application types and classifications, builds a neural network model using TensorFlow/Keras, trains and evaluates the model, and finally saves the trained model to a file for future use.
