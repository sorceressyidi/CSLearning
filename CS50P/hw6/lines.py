import sys
if len(sys.argv) != 2:
    sys.exit("Invalid argument")
if not sys.argv[1].endswith('.py'):
    sys.exit("Not a Python file")
try:
    with open(sys.argv[1], 'r') as f:
        print(sum(1 for line in f if line.strip() and not line.lstrip().startswith('#')))     
except FileNotFoundError:
    sys.exit("File does not exist.")
    