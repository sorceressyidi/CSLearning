vowels = ['a', 'e', 'i', 'o', 'u','A','E','I','O','U']
words = input("Input: ")
for word in words:
    if word in vowels:
        continue
    print(word, end="")