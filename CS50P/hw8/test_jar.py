from jar import Jar
import pytest
def test_init():
    with pytest.raises(ValueError):
        jar = Jar(-1)
    jar = Jar(0)
    assert jar.capacity == 0
    assert jar.size == 0
    jar = Jar(10)
    assert jar.capacity == 10
    assert jar.size == 0


def test_str():
    jar = Jar()
    assert str(jar) == ""
    jar.deposit(3)
    assert str(jar) == "🍪🍪🍪"
    jar.deposit(1)
    assert str(jar) == "🍪🍪🍪🍪"


def test_deposit():
    jar = Jar(10)
    with pytest.raises(ValueError):
        jar.deposit(-1)
    with pytest.raises(ValueError):
        jar.deposit(11)
    jar.deposit(3)
    assert jar.size == 3
    jar.deposit(7)
    assert jar.size == 10


def test_withdraw():
    jar = Jar(10)
    with pytest.raises(ValueError):
        jar.withdraw(-1)
    with pytest.raises(ValueError):
        jar.withdraw(1)
    jar.deposit(5)
    jar.withdraw(2)
    assert jar.size == 3
    with pytest.raises(ValueError):
        jar.withdraw(4)
    jar.withdraw(3)
    assert jar.size == 0