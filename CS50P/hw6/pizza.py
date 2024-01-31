import csv
import sys
from tabulate import tabulate
if len(sys.argv)!=2 or not sys.argv[1].endswith(".csv"):
    sys.exit("Invalid arguments.")
try:
    with open(sys.argv[1]) as f:
         print(tabulate(csv.DictReader(f), headers="keys", tablefmt="grid"))
except FileNotFoundError:
    sys.exit("File does not exist.")