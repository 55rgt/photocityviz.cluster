import random
array = [{'name': 'Tom', 'age': 24},{'name': 'Mike', 'age': 42}, {'name': 'Kate', 'age': 14}]
random.shuffle(array)
print(array)


a = []

b = [1, 2, 3, 4]
c = [5, 6, 7, 8]

a.extend(b)
a.extend(c)

print(a)
