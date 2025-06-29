import tkinter as tk
from PIL import Image, ImageDraw

class InputCanvas(tk.Frame):
    def __init__(self, master=None, width=280, height=280, bg='white', pen_width=12):
        super().__init__(master)
        self.width = width
        self.height = height
        self.pen_width = pen_width

        self.canvas = tk.Canvas(self, width=self.width, height=self.height, bg=bg)
        self.canvas.pack()

        self.button_clear = tk.Button(self, text="Clear", command=self.clear_canvas)
        self.button_clear.pack(pady=5)

        # PIL Image（内部描画用）
        self.image = Image.new("L", (self.width, self.height), color=255)
        self.draw = ImageDraw.Draw(self.image)

        # マウス操作のバインド
        self.canvas.bind("<B1-Motion>", self.draw_event)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        self.last_x = None
        self.last_y = None

    def draw_event(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    width=self.pen_width, capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y],
                           fill=0, width=self.pen_width)  # PIL側にも同じ線を描く
        self.last_x = x
        self.last_y = y

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.width, self.height], fill=255)

    def export_image(self):
        """
        Returns:
            PIL.Image: 描画内容を含むモノクロ画像（背景：白、筆跡：黒）
        """
        return self.image.copy()