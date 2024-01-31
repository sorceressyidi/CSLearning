class Jar:
    def __init__(self, capacity=12):
        if capacity < 0 or not type(capacity) == int:
            raise ValueError("Invalid Number")
        self._capacity = capacity
        self.cookies = 0

    def __str__(self):
        return "🍪" * self.cookies

    def deposit(self, n):
        if n < 0 or not type(n) == int or self.cookies + n > self.capacity:
            raise ValueError("Invalid Number")
        self.cookies += n

    def withdraw(self, n):
        if n < 0 or not type(n) == int or self.cookies - n < 0:
            raise ValueError("Invalid Number")
        self.cookies -= n

    @property
    def capacity(self):
        return self._capacity
    @property
    def size(self):
        return self.cookies 
    # size(capacity) has no setter,so we use size instead of cookies to reach "Obeject Oriented Programming"
    
def main():
    jar = Jar(10)
    print(jar)
    jar.deposit(10)
    jar.withdraw(2)
    print(jar)
    jar.withdraw(8)
    print(jar)
if __name__ == "__main__":
    main()