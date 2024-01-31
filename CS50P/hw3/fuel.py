while True:
    try:
        x,y = map(int, input("Fraction:").split("/")) #if x or y not int, ValueError
        if y == 0:
            raise ZeroDivisionError
        if x > y:
            raise ValueError
        break
    except (ValueError, ZeroDivisionError):
        pass
        
result = round(x/y*100)
if result <= 1 :
    print("E")
elif result >= 99:
    print("F")
else:
    print(f"{result}%")
