from fpdf import FPDF

class Shirtificate:
    def __init__(self, name):
        self.name = name
        self.generate()

    @classmethod
    def get_name(cls):
        name = input("Name: ").strip()
        return cls(name)

    def generate(self):
        pdf = FPDF()
        pdf.add_page()

        # 绘制背景矩形并设置填充颜色为淡蓝色
        pdf.set_fill_color(173, 216, 230)
        pdf.rect(0, 0, 210, 297, "F")

        # 绘制边框矩形并设置边框颜色为深蓝色
        pdf.set_draw_color(0, 0, 128)
        pdf.rect(5, 5, 200, 287)

        # 添加花纹
        for x in range(0, 210, 10):
            for y in range(0, 297, 10):
                pdf.set_fill_color(255, 255, 255)  # 设置填充颜色为白色
                pdf.ellipse(x, y, 5, 5, "F")  # 绘制白色椭圆

        # 设置标题字体和大小
        pdf.set_font("helvetica", "B", 60)

        # 在页面中间添加标题
        pdf.cell(0, 10, 'CS50 Shirtificate', ln=True, align='C')

        # 插入图片
        pdf.image("shirtificate.png", x=13, y=50, w=180)

        # 设置正文字体和大小
        pdf.set_font("Helvetica", "B", size=30)

        # 设置文本颜色为白色
        pdf.set_text_color(255, 255, 255)

        # 在底部添加文本
        pdf.cell(0, 180, align="C", text=f"{self.name} took CS50")

        # 输出 PDF 文件
        pdf.output("shirtificate.pdf")

def main():
    Shirtificate.get_name()

if __name__ == "__main__":
    main()
