def main():
    s = input("Greeting: ")
    print(f"${value(s)}")


def value(greeting):
    
    greet = greeting.strip().lower()
    if greet.startswith("hello"):
        return 0
    elif greet.startswith("h"):
        return 20
    else:
        return 100
if __name__ == "__main__":
    main()
