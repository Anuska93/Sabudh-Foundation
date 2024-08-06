Task 1.  LISTS
--------------


1.Write a Python program to multiply all the items in a list.
def multiply_list(items):
    result = 1
    for item in items:
        result *= item
    return result

2.Write a Python program to get the largest number from a list.
def get_largest_number(items):
    if not items:
        return None  # Return None if the list is empty
    largest = items[0]
    for item in items:
        if item > largest:
            largest = item
    return largest

3.Write a Python program to get the smallest number from a list.
def get_smallest_number(items):
    if not items:
        return None  # Return None if the list is empty
    smallest = items[0]
    for item in items:
        if item < smallest:
            smallest = item
    return smallest

4.Write a Python program to get a list, sorted in increasing order by the last element in each tuple from a given list of non-empty tuples.

def sort_by_last_element(tuples):
    return sorted(tuples, key=lambda x: x[-1])

5.Write a Python program to remove duplicates from a list.
def remove_duplicates(items):
    return list(set(items))

6.Write a Python program to check if a list is empty or not.
def is_list_empty(items):
    return len(items) == 0

7.Write a Python program to count the lowercase letters in a given list of word
def count_lowercase_letters(words):
    count = 0
    for word in words:
        for char in word:
            if char.islower():
                count += 1
    return count

8.Write a Python program to extract specified number of elements from a given list, which follows each other continuously.
○ Original list:  [1, 1, 3, 4, 4, 5, 6, 7]

○ Extract 2 number of elements from the said list which follows each other continuously: [1, 4]

○ Original list: [0, 1, 2, 3, 4, 4, 4, 4, 5, 7]

○ Extract 4 number of elements from the said list which follows each other continuously: [4]

def extract_continuous_elements(lst, n):
    result = []
    current_count = 1

    for i in range(1, len(lst)):
        if lst[i] == lst[i - 1]:
            current_count += 1
        else:
            if current_count == n:
                result.append(lst[i - 1])
            current_count = 1

    if current_count == n:
        result.append(lst[-1])

    return result

9.  Write a Python program to find the largest odd number in a given list of integers.

○ Sample Data:  ([0, 9, 2, 4, 5, 6]) -> 9

                           ([-4, 0, 6, 1, 0, 2]) -> 1

                          ([1, 2, 3]) -> 3

                           ([-4, 0, 5, 1, 0, 1]) -> 5

def find_largest_odd(numbers):
    largest_odd = None
    for number in numbers:
        if number % 2 != 0:  # Check if the number is odd
            if largest_odd is None or number > largest_odd:
                largest_odd = number
    return largest_odd

10. Write a Python program to print a specified list after removing the 0th, 4th and 5th elements.

○ Sample List : [A, B, C, D, E, F]

○ Expected Output : [A, B, F]

 def remove_elements(lst):
    indices_to_remove = {0, 4, 5}
    return [item for i, item in enumerate(lst) if i not in indices_to_remove]



Task 2. TUPLES
----------------


1. Write a Python program to create a tuple with different data types.
my_tuple = (1, "hello", 3.14, True)
print(f"Tuple with different data types: {my_tuple}")

2.Write a Python program to create a tuple of numbers and print one item.

numbers_tuple = (10, 20, 30, 40, 50)
print(f"Second item in the tuple: {numbers_tuple[1]}")

 3.Write a Python program to add an item to a tuple.

original_tuple = (1, 2, 3)
new_item = 4
new_tuple = original_tuple + (new_item,)
print(f"Tuple after adding an item: {new_tuple}")

4.Write a Python program to get the 4th element from the last element of a Tuple.

sample_tuple = (10, 20, 30, 40, 50, 60, 70)
fourth_from_last = sample_tuple[-4]
print(f"4th element from the last: {fourth_from_last}")

5.Write a Python program to convert a tuple to a dictionary.
tuple_of_pairs = (('a', 1), ('b', 2), ('c', 3))
dict_from_tuple = dict(tuple_of_pairs)
print(f"Dictionary from tuple: {dict_from_tuple}")

6.Write a Python program to replace the last value of tuples in a list.
Sample list: [(10, 20, 40), (40, 50, 60), (70, 80, 90)]
Expected Output: [(10, 20, 100), (40, 50, 100), (70, 80, 100)]

list_of_tuples = [(10, 20, 40), (40, 50, 60), (70, 80, 90)]
new_value = 100
modified_list = [t[:-1] + (new_value,) for t in list_of_tuples]
print(f"List after replacing the last value of tuples: {modified_list}")

 
Task 3. DICTIONARY
---------------------


1  Write a Python script to sort (ascending and descending) a dictionary by value.
my_dict = {'a': 3, 'b': 1, 'c': 2}
sorted_dict_asc = dict(sorted(my_dict.items(), key=lambda item: item[1]))
print(f"Dictionary sorted by value (ascending): {sorted_dict_asc}")
sorted_dict_desc = dict(sorted(my_dict.items(), key=lambda item: item[1], reverse=True))
print(f"Dictionary sorted by value (descending): {sorted_dict_desc}")

2. Write a Python program to iterate over dictionaries using for loops.
my_dict = {'a': 1, 'b': 2, 'c': 3}
for key, value in my_dict.items():
    print(f"Key: {key}, Value: {value}")

3. Write a Python script to merge two Python dictionaries.
dict1 = {'a': 1, 'b': 2}
dict2 = {'c': 3, 'd': 4}
merged_dict = {**dict1, **dict2}
print(f"Merged dictionary: {merged_dict}")

4. Write a Python program to sum all the items in a dictionary.
my_dict = {'a': 10, 'b': 20, 'c': 30}
total_sum = sum(my_dict.values())
print(f"Sum of all items: {total_sum}")

5. Write a Python program to multiply all the items in a dictionary.
my_dict = {'a': 2, 'b': 3, 'c': 4}
from functools import reduce
from operator import mul
product = reduce(mul, my_dict.values(), 1)
print(f"Product of all items: {product}")

6. Write a Python program to sort a given dictionary by key.
my_dict = {'b': 2, 'a': 1, 'c': 3}
sorted_dict_by_key = dict(sorted(my_dict.items()))
print(f"Dictionary sorted by key: {sorted_dict_by_key}")

7. Write a Python program to remove duplicates from the dictionary.
my_dict = {'a': 1, 'b': 2, 'c': 1, 'd': 3}
seen_values = set()
unique_dict = {}
for key, value in my_dict.items():
    if value not in seen_values:
        unique_dict[key] = value
        seen_values.add(value)
print(f"Dictionary with duplicates removed: {unique_dict}")



Task 4. Numpy
----------------
 

1: Numpy array creation and manipulation
1.Create a 1D Numpy array “a” containing 10 random integers between 0 and 99.

import numpy as np
a = np.random.randint(0, 100, size=10)
print(f"Array a: {a}")

2.Create a 2D Numpy array “b” of shape (3, 4) containing random integers between -10 and 10.

b = np.random.randint(-10, 11, size=(3, 4))
print(f"Array b:\n{b}")

3.Reshape “b” into a 1D Numpy array “b_flat”.

b_flat = b.flatten()
print(f"Flattened array b_flat: {b_flat}")

4.Create a copy of “a” called “a_copy”, and set the first element of “a_copy” to -1.

a_copy = np.copy(a)
a_copy[0] = -1
print(f"Array a_copy after modification: {a_copy}")

5.Create a 1D Numpy array “c” containing every second element of “a”.

c = a[::2]
print(f"Array c (every second element of a): {c}")


2: Numpy array indexing and slicing

Print the third element of “a”.
import numpy as np
a = np.random.randint(0, 100, size=10)
print(f"Third element of a: {a[2]}")

Print the last element of “b”.
b = np.random.randint(-10, 11, size=(3, 4))
print(f"Last element of b: {b[-1, -1]}")

Print the first two rows and last two columns of “b”.
print(f"First two rows and last two columns of b:\n{b[:2, -2:]}")

Assign the second row of “b” to a variable called “b_row”.
b_row = b[1]
print(f"Second row of b: {b_row}")

Assign the first column of “b” to a variable called “b_col”.
b_col = b[:, 0]
print(f"First column of b: {b_col}")


 3: Numpy array operations

Create a 1D Numpy array “d” containing the integers from 1 to 10.
import numpy as np
d = np.arange(1, 11)
print(f"Array d: {d}")

Add “a” and “d” element-wise to create a new Numpy array “e”.
a = np.random.randint(0, 100, size=10)
d = np.arange(1, 11)
e = a + d[:10]  # Adjusting d to match the length of a
print(f"Array e (element-wise addition of a and d): {e}")

Multiply “b” by 2 to create a new Numpy array “b_double”.
b = np.random.randint(-10, 11, size=(3, 4))
b_double = b * 2
print(f"Array b_double (b multiplied by 2):\n{b_double}")

Calculate the dot product of “b” and “b_double” to create a new Numpy array “f”.
f = np.dot(b, b_double.T)  # Transpose b_double to align dimensions
print(f"Dot product of b and b_double:\n{f}")

Calculate the mean of “a”,” b”, and “b_double” to create a new Numpy array “g”.
mean_a = np.mean(a)
mean_b = np.mean(b)
mean_b_double = np.mean(b_double)
g = np.array([mean_a, mean_b, mean_b_double])
print(f"Array g (mean of a, b, and b_double): {g}")


 

4: Numpy array aggregation
-----------------------------
 

Find the sum of every element in “a” and assign it to a variable “a_sum”.
import numpy as np
a = np.random.randint(0, 100, size=10)
a_sum = np.sum(a)
print(f"Sum of every element in a: {a_sum}")

Find the minimum element in “b” and assign it to a variable “b_min”.
b = np.random.randint(-10, 11, size=(3, 4))
b_min = np.min(b)
print(f"Minimum element in b: {b_min}")

Find the maximum element in “b_double” and assign it to a variable “b_double_max”.
b_double = b * 2
b_double_max = np.max(b_double)
print(f"Maximum element in b_double: {b_double_max}")


Task 5 : Pandas
----------------------
 

Dataset : https://www.kaggle.com/datasets/rkiattisak/sports-car-prices-dataset
Load the dataset into a Pandas DataFrame and display the first 5 rows to get an idea of the data.
import pandas as pd
url = "https://www.kaggle.com/datasets/rkiattisak/sports-car-prices-dataset"
df = pd.read_csv(url)
print(df.head())

Use Pandas to clean the dataset by removing any missing or duplicate values, and converting any non-numeric data to numeric data where appropriate.
df = df.dropna()
df = df.drop_duplicates()
print(df.head())

Use Pandas to explore the dataset by computing summary statistics for each column, such as mean, median, mode, standard deviation, and range.
summary_stats = df.describe(include='all')  # include='all' for all columns
print(summary_stats)
median = df.median()
mode = df.mode().iloc[0]  # mode can return multiple values
std_dev = df.std()
data_range = df.max() - df.min()
print(f"Median:\n{median}")
print(f"Mode:\n{mode}")
print(f"Standard Deviation:\n{std_dev}")
print(f"Range:\n{data_range}")

Use Pandas to group the dataset by car make and compute the average price for each make.
avg_price_by_make = df.groupby('Make')['Price'].mean()
print(avg_price_by_make)

Use Pandas to group the dataset by year and compute the average horsepower for each year.
avg_hp_by_year = df.groupby('Year')['Horsepower'].mean()
print(avg_hp_by_year)

Use Pandas to create a scatter plot of price versus horsepower, and add a linear regression line to the plot.
import matplotlib.pyplot as plt
import seaborn as sns
sns.lmplot(x='Horsepower', y='Price', data=df)
plt.title('Price vs Horsepower')
plt.xlabel('Horsepower')
plt.ylabel('Price')
plt.show()

Use Pandas to create a histogram of the 0-60 MPH times in the dataset, with bins of size 0.5 seconds.
plt.hist(df['0-60 MPH'], bins=range(int(df['0-60 MPH'].min()), int(df['0-60 MPH'].max()) + 1, 0.5))
plt.title('Histogram of 0-60 MPH Times')
plt.xlabel('0-60 MPH Time (seconds)')
plt.ylabel('Frequency')
plt.show()

Use Pandas to filter the dataset to only include cars with a price greater than $500,000, and then sort the resulting dataset by horsepower in descending order.
filtered_sorted_df = df[df['Price'] > 500000].sort_values(by='Horsepower', ascending=False)
print(filtered_sorted_df)

Use Pandas to export the cleaned and transformed dataset to a new CSV file.
df.to_csv('cleaned_sports_car_prices.csv', index=False)

 
