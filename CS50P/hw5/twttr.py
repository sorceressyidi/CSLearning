def main():
    print("Output: ",shorten(input("Input: ")))

def shorten(words):
    vowels = ['a', 'e', 'i', 'o', 'u','A','E','I','O','U']
    return "".join([ch for ch in words if ch.upper() not in vowels])
if __name__ == "__main__":
    main()
