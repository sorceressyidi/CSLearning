import random
def main():
    n = get_level()
    rand = random.randint(1, n)
    while True:
        try :
            guess = int(input("Guess: "))
            if guess < 0:
                raise ValueError
            if guess == rand:
                print("Just right!")
                break
            if guess < rand:
                print("Too small!")
            if guess > rand:
                print("Too large!")
        except ValueError:
            pass
    
def get_level():
   while True:
       try:
           level = int(input("Level: "))
           if level > 0:
               return level
       except ValueError:
              pass
if __name__ == "__main__":
    main()