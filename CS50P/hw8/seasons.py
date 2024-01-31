from datetime import date
import sys
import inflect

def main():
    try:
        born_date = date.fromisoformat(input("Date of Birth: "))
    except ValueError:
        sys.exit("Invalid Input")
    season = Season.from_date(born_date, date.today())
    print(season.get_season())


class Season:
    def __init__(self,born_date,now_date):
        self.born_date = born_date
        self.now_date = now_date
    @classmethod
    def from_date(cls, born_date, now_date):
        return cls(born_date, now_date)
    def get_season(self):
        time = (self.now_date - self.born_date).days*24*60
        p = inflect.engine()
        return  f"{p.number_to_words(time, andword='').capitalize()} {p.plural('minute', time)}"

        
if __name__ == "__main__":
    main()