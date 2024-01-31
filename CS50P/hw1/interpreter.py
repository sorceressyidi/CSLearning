x,y,z = input("Expression: ").split(" ")
match y:
    case "+":
        print(f"{float(x) + float(z):.1f}")
    case "-":
        print(f"{float(x) - float(z):.1f}")
    case "*":
        print(f"{float(x) * float(z):.1f}")
    case "/":
        print(f"{float(x) / float(z):.1f}")
    
    
'''
x, y, z = input("Expression: ").split()
x, z = int(x), int(z)
ans = eval(f"{x} {y} {z}")
print(f"{ans:.1f}")
'''