months = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]
while True:
    try:
        date = input("Date: ")
        if len(date.split("/"))==3:
            date = date.split("/")
            month,day,year = map(int, date)
            if 1<=month<=12 and 1<=day<=31 :
                print(f"{year:04d}-{month:02d}-{day:02d}")
                break
            else:
                raise ValueError
        else:
            if len(date.split(","))!=2:
                raise ValueError
            month,day,year = date.strip().replace(",","").split(" ")  
            day = int(day)
            year = int(year)       
            if month in months and 1 <= day <= 31:
                month = months.index(month)+1
                print(f"{year:04d}-{month:02d}-{day:02d}")
                break
            else:
                raise ValueError
    except ValueError:
        pass

    