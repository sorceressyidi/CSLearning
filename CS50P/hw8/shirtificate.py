from fpdf import FPDF
class shirtificate:
    def __init__(self,name):
        self.name = name
        self.generate()
    @classmethod
    def get_name(cls):
        name = input("Name: ").strip()
        return cls(name)
    def generate(self):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", "B", 60)
        pdf.cell(0, 10, 'CS50 Shirtificate', new_x="LMARGIN", new_y="NEXT", align='C')
        pdf.image("shirtificate.png", x=13, y=50, w=180)
        pdf.set_font("Helvetica", "B", size=30)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 180, align="C", text=f"{self.name} took CS50")
        pdf.output("shirtificate.pdf")
def main():
    shirtificate.get_name()


if __name__ == "__main__":
    main()