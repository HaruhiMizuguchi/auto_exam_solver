import tkinter as tk
from PIL import ImageGrab

class RegionCapture:
    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-alpha', 0.2)
        self.root.configure(bg='black')
        self.start_x = self.start_y = 0
        self.rect = None
        self.canvas = tk.Canvas(self.root, cursor="cross", bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.bbox = None

    def on_press(self, event):
        self.start_x, self.start_y = event.x, event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, width=2)

    def on_drag(self, event):
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        x1, y1, x2, y2 = self.canvas.coords(self.rect)
        self.bbox = (int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2)))
        self.root.destroy()

    def capture(self):
        self.root.mainloop()
        if not self.bbox:
            return None
        return ImageGrab.grab(bbox=self.bbox)


def capture_region_to_image():
    cap = RegionCapture()
    img = cap.capture()
    return img
