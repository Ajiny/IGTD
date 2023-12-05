import pandas as pd
import os
from IGTD_Functions import min_max_transform, table_to_image, select_features_by_variation


# 99 columns
num_row = 4    # Number of pixel rows in image representation
num_col = 4    # Number of pixel columns in image representation
num = num_row * num_col # Number of features to be included for analysis, which is also the total number of pixels in image representation
save_image_size = 3 # Size of pictures (in inches) saved during the execution of IGTD algorithm.
max_step = 10000    # The maximum number of iterations to run the IGTD algorithm, if it does not converge.
val_step = 50  # The number of iterations for determining algorithm convergence. If the error reduction rate
                # is smaller than a pre-set threshold for val_step itertions, the algorithm converges.

# Import the example data and linearly scale each feature so that its minimum and maximum values are 0 and 1, respectively.
data = pd.read_csv('../MyData/cardio_train.csv', nrows=2000, index_col=0)

# adjust age to years
data['age'] =  data['age'] / 365.24

# make your blood pressure values are within range and that ap_hi is higher than ap_lo
data = data[(data['ap_lo'] <= 370) & (data['ap_lo'] > 0)].reset_index(drop=True)
data = data[(data['ap_hi'] <= 370) & (data['ap_hi'] > 0)].reset_index(drop=True)
data = data[data['ap_hi'] >= data['ap_lo']].reset_index(drop=True)

# split positive and negative data
positive_data = data[data['cardio'] == 1]
positive_data = positive_data.drop(['cardio'], axis=1).copy()

negative_data = data[data['cardio'] == 0]
negative_data = negative_data.drop(['cardio'], axis=1).copy()

positive_data.to_csv('cvd_positive.csv', index=False)
negative_data.to_csv('cvd_negative.csv', index=False)

#positive_data.info()
#negative_data.info()
#data.info()

# Select features with large variations across samples
id_p = select_features_by_variation(positive_data, variation_measure='var', num=num)
positive_data = positive_data.iloc[:, id_p]

id_n = select_features_by_variation(negative_data, variation_measure='var', num=num)
negative_data = negative_data.iloc[:, id_n]

# Perform min-max transformation so that the maximum and minimum values of every feature become 1 and 0, respectively.
norm_positive_data = min_max_transform(positive_data.values)
norm_positive_data = pd.DataFrame(norm_positive_data, columns=positive_data.columns, index=positive_data.index)

norm_negative_data = min_max_transform(negative_data.values)
norm_negative_data = pd.DataFrame(norm_negative_data, columns=negative_data.columns, index=negative_data.index)

# Run the IGTD algorithm using (1) the Euclidean distance for calculating pairwise feature distances and pariwise pixel
# distances and (2) the absolute function for evaluating the difference between the feature distance ranking matrix and
# the pixel distance ranking matrix. Save the result in Test_1 folder.
fea_dist_method = 'Euclidean'
image_dist_method = 'Euclidean'
error = 'abs'

result_dir_p = '../CVDResults/Table_To_Image_Conversion/cvd_positive/Test_1'
result_dir_n = '../CVDResults/Table_To_Image_Conversion/cvd_negative/Test_1'

os.makedirs(name=result_dir_p, exist_ok=True)
os.makedirs(name=result_dir_n, exist_ok=True)
table_to_image(norm_positive_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
               max_step, val_step, result_dir_p, error)
table_to_image(norm_negative_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
               max_step, val_step, result_dir_n, error)

# Run the IGTD algorithm using (1) the Pearson correlation coefficient for calculating pairwise feature distances,
# (2) the Manhattan distance for calculating pariwise pixel distances, and (3) the square function for evaluating
# the difference between the feature distance ranking matrix and the pixel distance ranking matrix.
# Save the result in Test_2 folder.
fea_dist_method = 'Pearson'
image_dist_method = 'Manhattan'
error = 'squared'
norm_positive_data = norm_positive_data.iloc[:, :800]
norm_negative_data = norm_negative_data.iloc[:, :800]
result_dir_p = '../CVDResults/Table_To_Image_Conversion/cvd_positive/Test_2'
result_dir_n = '../CVDResults/Table_To_Image_Conversion/cvd_negative/Test_2'

os.makedirs(name=result_dir_p, exist_ok=True)
os.makedirs(name=result_dir_n, exist_ok=True)
table_to_image(norm_positive_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
              max_step, val_step, result_dir_p, error)
table_to_image(norm_negative_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
              max_step, val_step, result_dir_n, error)
