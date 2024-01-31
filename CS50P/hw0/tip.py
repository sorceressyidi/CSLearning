def main():
    dollars = dollars_to_float(input("How much was the meal? "))
    percent = percent_to_float(input("What percentage would you like to tip? "))
    tip = dollars * percent
    print(f"Leave ${tip:.2f}")
def dollars_to_float(d):
    # d in the form of $price(in num)
    return float(d[1:])

def percent_to_float(p):
    #p in the form of 15%
    return float(p[:-1])/100

main()