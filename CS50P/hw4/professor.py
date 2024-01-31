import random
def main():
    level = get_level()
    score = 0
    for n in range (10):
        x = generate_integer(level)
        y = generate_integer(level)
        score += check_answer(x, y)
    print(f"Score: {score}")
def get_level():
    while True:
        try:
            level = int(input("Level: "))
            if level in range(1,4):
                return level
        except ValueError:
               pass
def generate_integer(level):
    if level==1:
        return random.randint(0, 9)
    return random.randint(10**(level-1), 10 ** level-1)
def check_answer(x, y):
    answer = x+y
    for n in range(3):
        try:
            user_answer = int(input(f"{x} + {y} = "))
            if user_answer == answer:
                return 1
            else:
                print("EEE")
        except ValueError:
            pass
    print(f"{x} + {y} = {answer}")
    return 0
if __name__ == "__main__":
    main()