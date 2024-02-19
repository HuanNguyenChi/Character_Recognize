import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw
from tkinter import ttk
import cv2
from tkinter import filedialog
import os
from tkinter import messagebox
import predict


def browse_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        path_var.set(file_path)
        entry_path.delete(0, tk.END)  # Xóa nội dung hiện tại của Entry
        entry_path.insert(0, file_path)  # Gán đường dẫn vào Entry

def detect():
    # Thực hiện chức năng detect tại đây
    path_var = save_image()
    print(path_var)
    predictions = predict.sol(os.path.join(path_var))
    messagebox.showinfo("Detect", "This character is:" + predictions)
    pass

def clear():
    path_var.set('')
    entry_path.delete(0, tk.END)
    canvas.delete("all")


def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=15)


# Tạo cửa sổ chính
root = tk.Tk()
root.geometry("600x400+0+0")
root.title("RECOGNITION_VIETNAMESE")
root.configure(background="beige")
root.resizable(False, False)

# Tạo một biến StringVar để lưu trữ đường dẫn được chọn
path_var = tk.StringVar()

# Tạo một nhãn để hiển thị đường dẫn
label = tk.Label(root,
                 text="This is an app for recognition VietNamese. \n You must upload an image to detect.\n Try it.",
                 background="beige")
label.grid(row=0, column=1)

# Tạo một Entry để hiển thị đường dẫn đã chọn
label_path = tk.Label(root,
                      text="File :",
                      background="beige").grid(row=1, column=0)
entry_path = tk.Entry(root, textvariable=path_var, width=80)
entry_path.grid(row=1, column=1)

# Tạo một cửa sổ để
label_detect = tk.Label(root,
                        text="Detect :",
                        background="beige").grid(row=2, column=0)
text_widget = tk.Text(root, width=50, height=10, padx=1, pady=1, state="disabled", background="gray")
text_widget.grid(row=2, column=1, padx=70)

# Kích thước canvas
canvas_width = 150
canvas_height = 150

# Tạo canvas để vẽ
canvas = tk.Canvas(root, bg="white", width=150, height=150)
canvas.grid(row=2, column=1)
canvas.bind("<B1-Motion>", paint)


# Tạo nút "Save" để lưu hình ảnh
def save_image():
    if len(canvas.find_all()) >0:
        # Khởi tạo một hình ảnh mới với kích thước 400x400 pixel và màu nền trắng
        image = Image.new("RGB", (400, 400), "white")

        # Sử dụng ImageDraw để vẽ nội dung của canvas lên hình ảnh
        draw = ImageDraw.Draw(image)

        # Tính tỷ lệ để lấy hình ảnh vẽ trên canvas và thay đổi kích thước
        scale_factor_x = 400 / canvas_width
        scale_factor_y = 400 / canvas_height

        for item in canvas.find_all():
            coords = canvas.coords(item)
            x1 = (coords[0] - 5) * scale_factor_x
            y1 = (coords[1] - 5) * scale_factor_y
            x2 = (coords[2] + 5) * scale_factor_x
            y2 = (coords[3] + 5) * scale_factor_y
            draw.ellipse((x1, y1, x2, y2), fill="black")

        # Chọn vị trí để lưu hình ảnh trên ổ đĩa D
        save_path = "E:\CODE\BTL/drawn_image.png"

        # Lưu hình ảnh với định dạng PNG
        image.save(save_path, "PNG")
        return save_path
    return path_var.get()

# Tạo 3 nút upload-detect-clear
button_upload = tk.Button(root, text="Upload", background="white",
                          width=10, height=2, padx=50, pady=5,
                          command=browse_file)
button_upload.grid(row=3, column=0, padx=10)

button_detect = tk.Button(root, text="Detect", background="white", width=10,
                          height=2, padx=5, pady=5,
                          command=detect)
button_detect.grid(row=3, column=1)

button_clear = tk.Button(root, text="Clear", background="white",
                         width=10, height=2, padx=50, pady=5,
                         command=clear)
button_clear.grid(row=3, column=2, padx=10)

# Thiết lập lưới cho giao diện
root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=1)
root.rowconfigure(2, weight=1)
root.rowconfigure(3, weight=1)
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.columnconfigure(2, weight=1)

root.mainloop()
