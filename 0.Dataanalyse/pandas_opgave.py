import numpy as np
import pandas as pd


# Create and populate a 5x2 NumPy array.
my_data = np.array([[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]])

# Create a Python list that holds the names of the two columns.
my_column_names = ['temperature', 'activity']

# Create a DataFrame.
my_dataframe = pd.DataFrame(data=my_data, columns=my_column_names)

# Print the entire DataFrame
print(my_dataframe)

# Create a new column named adjusted.
my_dataframe["adjusted"] = my_dataframe["activity"] + 2

# Print the entire DataFrame
print(my_dataframe)

print("Rows #0, #1, and #2:")
print(my_dataframe.head(3), '\n')

print("Row #2:")
print(my_dataframe.iloc[[2]], '\n')

print("Rows #1, #2, and #3:")
print(my_dataframe[1:4], '\n')

print("Column 'temperature':")
print(my_dataframe['temperature'])

naming_is_hard = np.random.randint(low=0, high=101, size=(3,4))

print(naming_is_hard)
my_data = np.array([[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]])


column_names = ["Eleanor", "Chidi", "Tahani", "Jason"]

dataframe = pd.DataFrame(data=naming_is_hard, columns=column_names)

print(dataframe)


print("Experiment with a reference:")
reference_to_dataframe = dataframe

# Print the starting value of a particular cell.
print("  Starting value of dataframe: %d" % dataframe['Jason'][1])
print("  Starting value of reference_to_dataframe: %d\n" % reference_to_dataframe['Jason'][1])

# Modify a cell in dataframe.
dataframe.at[1, 'Jason'] = dataframe['Jason'][1] + 5
print("  Updated dataframe: %d" % dataframe['Jason'][1])
print("  Updated reference_to_dataframe: %d\n\n" % reference_to_dataframe['Jason'][1])

# Create a true copy of my_dataframe
print("Experiment with a true copy:")
copy_of_my_dataframe = my_dataframe.copy()

# Print the starting value of a particular cell.
print("  Starting value of my_dataframe: %d" % my_dataframe['activity'][1])
print("  Starting value of copy_of_my_dataframe: %d\n" % copy_of_my_dataframe['activity'][1])

# Modify a cell in dataframe.
my_dataframe.at[1, 'activity'] = my_dataframe['activity'][1] + 3
print("  Updated my_dataframe: %d" % my_dataframe['activity'][1])
print("  copy_of_my_dataframe does not get updated: %d" % copy_of_my_dataframe['activity'][1])