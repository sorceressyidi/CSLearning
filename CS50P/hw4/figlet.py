from pyfiglet import Figlet
import random
import sys
figlet = Figlet().getFonts()
if len(sys.argv)==1:
    font = random.choice(figlet)
elif len(sys.argv)==3:
    if sys.argv[1] in ['-font','-f'] and sys.argv[2] in figlet:
        font = sys.argv[2]
    else :
        sys.exit("Invalid arguments")
else:
    sys.exit("Invalid number of arguments.")
    
text = input("Input: ")
print ("Output: ",Figlet(font=font).renderText(text))
