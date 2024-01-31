def main():
    time= convert(input("What time is it? ".strip()))
    if 7 <= time <= 8:
        print("Breakfast time")
    if 12 <= time <= 13:
        print("Lunch time")
    if 18 <= time <= 19:
        print("Dinner time")
def convert(time):
    hour,minute = time.replace("a.m.","").replace("p.m.","").split(":")
    hour,minute = list(map(int,(hour,minute)))
    if "p.m." in time and hour != 12:
        hour += 12
    if "a.m." in time and hour == 12:
        hour = 0
    return hour + minute/60

if __name__ == "__main__":
    main()

