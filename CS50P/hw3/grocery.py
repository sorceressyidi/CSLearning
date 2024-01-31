groceries = {}
while True:
    try:
        grocery = input("").upper()
        if grocery in groceries:
            groceries[grocery] += 1
        else:
            groceries[grocery] = 1
    except EOFError:
        groceries = dict(sorted(groceries.items()))
        for i in groceries:
            print(groceries[i],i)
        break
    