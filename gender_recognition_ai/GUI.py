import tkinter
from tkinter import filedialog, CENTER, NW

from PIL import ImageTk, Image

from gender_recognition_ai import GenderNN

root = tkinter.Tk()
root.geometry("600x400+0+0")
root.title("Gender Recognition AI v1.0")
root.iconbitmap("gender.ico")


def open_picture(path="bg.png"):
    width = 600
    img = Image.open(path)
    w_percent = (width / float(img.size[0]))
    h_size = int((float(img.size[1]) * float(w_percent)))
    img = img.resize((width, h_size), Image.ANTIALIAS)
    root.geometry("%dx%d+0+0" % (width, h_size))
    return img


picture = ImageTk.PhotoImage(open_picture())
pic_label = tkinter.Label(image=picture)
pic_label.grid(row=0, column=0, columnspan=3)

gender_label = tkinter.Label(text="Gender: none")
gender_label.config(font=("Arial", 20))
gender_label.place(relx=0, rely=0, anchor=NW)


# On button press, prompts the user to select a file then predicts the gender
def predict_pressed():
    root.filename = filedialog.askopenfilename(
        title="Select an image", filetypes=(("all files", "*.*"), ("png files", "*.png"), ("jpg files", "*.jpg")))
    new_picture = ImageTk.PhotoImage(open_picture(root.filename))
    pic_label.configure(image=new_picture)
    pic_label.image = new_picture
    prediction = GenderNN.make_prediction(root.filename)
    gender_label["text"] = "Gender: %s" % prediction


button_predict = tkinter.Button(root, text="Open Picture", command=predict_pressed)
button_predict.config(font=("Arial", 14))
button_predict.place(relx=0.5, rely=0.94, anchor=CENTER)

root.mainloop()
