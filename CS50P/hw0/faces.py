def convert(text):
    return text.replace(":)", "🙂").replace(":(", "🙁")
def main():
    print(convert(input("What's the text?")))
if __name__ == "__main__":
    main()


