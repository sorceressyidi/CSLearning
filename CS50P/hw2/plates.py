def main():
    plate = input("Plate: ")
    if is_valid(plate):
        print("Valid")
    else:
        print("Invalid")


def is_valid(s):
    if len(s)>6 or len(s)<2:
        return False
    if not (s[0].isalpha() and s[1].isalpha()):
        return False
    if not all(ch.isalnum() for ch in s):
        return False
    flag = 0
    for ch in s :
        if ch.isdigit():
            flag = 1
        if ch.isalpha() and flag:
            return False
    for ch in s:
        if ch.isdigit():
            return ch != '0'
    return True
    
main()