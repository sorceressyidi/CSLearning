import csv
import sys
if len(sys.argv)!=3 or not sys.argv[1].endswith(".csv"):
    if len(sys.argv)<2:
        sys.exit("Too few command-line arguments.")
    elif len(sys.argv)>3:
        sys.exit("Too many command-line arguments.")
    else:
        sys.exit("Invalid arguments.")
try:
    with open(sys.argv[1]) as input, open(sys.argv[2], "w") as output:
        reader = csv.DictReader(input)
        writer = csv.DictWriter(output, fieldnames=["first","last","house"])
        writer.writeheader()
        for _ in reader:
            last,first = _["name"].split(", ")
            writer.writerow({"first":first,"last":last,"house":_["house"]})
           
except FileNotFoundError:
     sys.exit(f"Could not read {sys.argv[1]}")