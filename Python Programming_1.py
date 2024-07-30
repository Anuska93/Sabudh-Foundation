1.
def print_squares(N):
      squares = [i**2 for i in range(N)]
    print(squares)

2.
def remove_duplicates(s):
    seen = set()      
    result = []        
    
    for char in s:
        if char not in seen:
            seen.add(char)
            result.append(char)
    
    return ''.join(result) 

3.
def filter_numbers(numbers):
    result = []
    
    for num in numbers:
        if num > 500:
            break
        if num > 150:
            continue
        if num % 5 == 0:
            result.append(num)
    
    return result

4.
def count_digits(number):
   
    if number == 0:
        return 1

    count = 0
    while number > 0:
        number 
        count += 1
    
    return count

5.
def sum_of_series(n):
    total_sum = 0
    current_term = 0
    
    for i in range(1, n + 1):
        current_term = current_term * 10 + 2  
        total_sum += current_term  
    
    return total_sum

6.
def reverse_number(number):
    reversed_number = 0
    
    while number > 0:
        last_digit = number % 10
        reversed_number = reversed_number * 10 + last_digit
        number 
    
    return reversed_number

7.
def elements_at_odd_indices(lst):
    result = []
    for index in range(len(lst)):
        if index % 2 != 0: 
            result.append(lst[index])
    
    return result

8.
def find_median_of_three(a, b, c):
    numbers = [a, b, c]
    numbers.sort()
    return numbers[1]
    a = int(input("Input first number: "))
    b = int(input("Input second number: "))
    c = int(input("Input third number: "))
    print(find_median_of_three(a, b, c))

9.
def factorial(n):
    if n == 0:
        return 1
    result = 1
    for i in range(1, n + 1):
        result *= i
      return result

10.
def count_vowels_and_consonants(word):
    vowels = "aeiouAEIOU"
    vowel_count = 0
    consonant_count = 0
    for char in word:
        if char.isalpha():
            if char in vowels:
                vowel_count += 1
            else:
                consonant_count += 1
     return vowel_count, consonant_count
