due = int(50)
insert = 0
while(due > 0):
    print(f"Amount Due: {due}")
    insert = int(input("Insert Coin: "))
    if insert in [5,10,25]:
        due -= int(insert)
due = -due
print(f"Change Owed: {due}")