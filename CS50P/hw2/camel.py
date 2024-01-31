camel = input("camelCase: ").strip()
for ch in camel:
    if ch.isupper():
        print("_", end="")
    print(ch.lower(), end="")
    
'''
snake = "".join(["_" + ch.lower() if ch.isupper() else ch for ch in camel])
print("snake_case:", snake)
'''
