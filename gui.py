from tkinter import Tk 
import tkinter.messagebox
import customtkinter
from tensorflow.keras.models import load_model
import threading
import time
from tkinter.filedialog import askopenfilename
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib
from tkinter import filedialog
import numpy as np
import tensorflow as tf
import segmentation_models as sm
from sklearn.preprocessing import MinMaxScaler
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from Models.UNet import create_model
from Dataset_Load import datset_load

minmaxscaler = MinMaxScaler()
customtkinter.set_appearance_mode("Dark")  
customtkinter.set_default_color_theme("dark-blue")

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("AI Visualization")
        self.geometry(f"{1500}x{780}")
        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./icons")
        self.home_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "home_dark.png")),
                                                 dark_image=Image.open(os.path.join(image_path, "home_light.png")), size=(30, 30))
        self.chat_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "chat_dark.png")),
                                                 dark_image=Image.open(os.path.join(image_path, "chat_light.png")), size=(30, 30))
        self.add_user_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "add_user_dark.png")),
                                                     dark_image=Image.open(os.path.join(image_path, "add_user_light.png")), size=(30, 30))
        self.image_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "image_icon_light.png")),
                                                     dark_image=Image.open(os.path.join(image_path, "image_icon_light.png")), size=(30, 30))
        self.upload_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "upload.png")),
                                                     dark_image=Image.open(os.path.join(image_path, "upload.png")), size=(30, 30))
        self.prediction_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "prediction.png")),
                                                     dark_image=Image.open(os.path.join(image_path, "prediction.png")), size=(30, 30))
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)


        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(4, weight=1)
        
        self.home_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Home",
                                                   fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                   image=self.home_image, anchor="w", command=self.home_button_event)
        self.home_button.grid(row=1, column=0, sticky="ew")

        self.model_predict_frame_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Frame 2",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      image=self.chat_image, anchor="w", command=self.model_predict_frame_button_event)
        self.model_predict_frame_button.grid(row=2, column=0, sticky="ew")

        self.frame_3_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Frame 3",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      image=self.add_user_image, anchor="w", command=self.frame_3_button_event)
        self.frame_3_button.grid(row=3, column=0, sticky="ew")

        self.home_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.home_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.home_frame.grid_propagate(False)
        self.home_frame.grid_rowconfigure(3, weight=1)
       
        
        

        self.figure = Figure(figsize=(16, 12), dpi=100, facecolor='#1a1a1a')
        self.original_subplot = self.figure.add_subplot(131)
        self.original_subplot.set_title("Original Image", color="#519dd6")
        self.mask_subplot = self.figure.add_subplot(132)
        self.mask_subplot.set_title("Ground Truth", color="#519dd6")
        self.predicted_subplot = self.figure.add_subplot(133)
        self.predicted_subplot.set_title("Predicted Image", color="#519dd6")
        self.original_subplot.set_axis_off()
        self.mask_subplot.set_axis_off()
        self.predicted_subplot.set_axis_off()

    
        # create canvas to show the figure
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.home_frame)
        self.canvas.draw()

        

        self.title_label = customtkinter.CTkLabel(self.home_frame, text="IMAGE SELECTION AND PREDICTION", font=("Arial", 30),text_color="#d3d3d3")
        self.title_label.grid(row=0, column=0, columnspan=3, padx=10, pady=30)

        self.select_image_button = customtkinter.CTkButton(self.home_frame,image=self.image_image, text="Resim Seç", command=self.select_image,height=50,width=200)
        self.select_image_button.grid(row=1, column=0, padx=20, pady=20)

        self.model_load_button = customtkinter.CTkButton(self.home_frame,image=self.upload_image, text="Model Yükle", command=self.model_load_button_event,height=50,width=200)
        self.model_load_button.grid(row=1, column=1, padx=20, pady=20)

        self.model_predict = customtkinter.CTkButton(self.home_frame,image=self.prediction_image, text="Tahmin Yap", command=lambda: self.prediction(img=predict_image),height=50,width=200)
        self.model_predict.grid(row=1, column=2, padx=20, pady=20)

        self.progressbar = customtkinter.CTkProgressBar(self.home_frame)
        self.progressbar.set(0)
        self.progressbar.grid(row=2, column=0, columnspan=3, padx=(20, 10), pady=(10, 10), sticky="ew")

        self.canvas.get_tk_widget().grid(row=3, column=0, columnspan=3, padx=20, pady=10)

        self.accuracy_label = customtkinter.CTkLabel(self.home_frame, text="Accuracy : ", font=("Arial", 20),text_color="#d3d3d3")
        self.accuracy_label .grid(row=4, column=0, columnspan=3, padx=10, pady=30)
        

        self.model_predict_frame = customtkinter.CTkFrame(self, corner_radius=10, fg_color="transparent")
        # create third frame
        self.third_frame = customtkinter.CTkFrame(self, corner_radius=10, fg_color="transparent")
        self.target_size = (256, 256)
        self.train_dir = None
        self.test_dir = None
        self.epochs = 10
        self.batch_size = 32
        self.optimizer_var = customtkinter.StringVar(value="adam")
        self.pooling_var = customtkinter.StringVar(value="avg")
        self.loss_var = customtkinter.StringVar(value="categorical_crossentropy")
        self.dropout_var = customtkinter.BooleanVar(value=True)
        
        self.create_widgets()
        # select default frame
        self.select_frame_by_name("home")
        
    def create_widgets(self):
        # Veriseti seçim kısmı
        self.dataset_label = customtkinter.CTkLabel(self.model_predict_frame, text="Veriseti Tipini Seçin:")
        self.dataset_label.grid(row=0, column=0, sticky='w', padx=5, pady=5)

        self.dataset_combo = customtkinter.CTkComboBox(self.model_predict_frame, values=["Binary", "Categorical"])
        self.dataset_combo.set(0)
        self.dataset_combo.grid(row=0, column=1, sticky='w', padx=5, pady=5)

        # Epoch sayısı kısmı
        self.epochs_label = customtkinter.CTkLabel(self.model_predict_frame, text="Epoch Sayısı:")
        self.epochs_label.grid(row=1, column=0, sticky='w', padx=5, pady=5)

        self.epochs_entry = customtkinter.CTkEntry(self.model_predict_frame)
        self.epochs_entry.insert(0, "10")
        self.epochs_entry.grid(row=1, column=1, sticky='w', padx=5, pady=5)

        # Batch size kısmı
        self.batch_label = customtkinter.CTkLabel(self.model_predict_frame, text="Batch Size:")
        self.batch_label.grid(row=2, column=0, sticky='w', padx=5, pady=5)

        self.batch_entry = customtkinter.CTkEntry(self.model_predict_frame)
        self.batch_entry.insert(0, "32")
        self.batch_entry.grid(row=2, column=1, sticky='w', padx=5, pady=5)

        # Optimizer seçim kısmı
        self.optimizer_label = customtkinter.CTkLabel(self.model_predict_frame, text="Optimizer Seçin:")
        self.optimizer_label.grid(row=3, column=0, sticky='w', padx=5, pady=5)

        self.optimizer_combo = customtkinter.CTkComboBox(self.model_predict_frame, values=["Adam", "SGD", "RMSprop"])
        self.optimizer_combo.set(0)
        self.optimizer_combo.grid(row=3, column=1, sticky='w', padx=5, pady=5)

        # Pooling seçim kısmı
        self.pooling_label = customtkinter.CTkLabel(self.model_predict_frame, text="Pooling Seçin:")
        self.pooling_label.grid(row=4, column=0, sticky='w', padx=5, pady=5)

        self.pooling_combo = customtkinter.CTkComboBox(self.model_predict_frame, values=["Max", "Average"])
        self.pooling_combo.set(0)
        self.pooling_combo.grid(row=4, column=1, sticky='w', padx=5, pady=5)

        # Dropout kısmı
        self.dropout_label = customtkinter.CTkLabel(self.model_predict_frame, text="Dropout Kullan:")
        self.dropout_label.grid(row=5, column=0, sticky='w', padx=5, pady=5)

        self.dropout_combo = customtkinter.CTkComboBox(self.model_predict_frame, values=["Evet", "Hayır"])
        self.dropout_combo.set(0)
        self.dropout_combo.grid(row=5, column=1, sticky='w', padx=5, pady=5)

        # Loss fonksiyonu seçim kısmı
        self.loss_label = customtkinter.CTkLabel(self.model_predict_frame, text="Loss Fonksiyonu Seçin:")
        self.loss_label.grid(row=6, column=0, sticky='w', padx=5, pady=5)

        self.loss_combo = customtkinter.CTkComboBox(self.model_predict_frame, values=["Binary Crossentropy", "Categorical Crossentropy"])
        self.loss_combo.set(0)
        self.loss_combo.grid(row=6, column=1, sticky='w', padx=5, pady=5)

        self.train_button = customtkinter.CTkButton(self.model_predict_frame, text="Train", command=self.train_model)
        self.train_button.grid(row=12, column=0, padx=5, pady=5)

        # Status Label
        self.status_label = customtkinter.CTkLabel(self, text="", fg_color="green")
        self.status_label.grid(row=13, column=0, columnspan=2, padx=5, pady=5)

        # Progress Bar
        self.progress = customtkinter.CTkProgressBar(self.model_predict_frame, mode='determinate')
        self.progress.grid(row=14, column=0, columnspan=2, padx=5, pady=5)

        self.quit_button = customtkinter.CTkButton(self.model_predict_frame, text="Quit", command=self.quit)
        self.quit_button.grid(row=12, column=1, padx=5, pady=5)

        self.train_dir_select_button = customtkinter.CTkButton(self.model_predict_frame, text="Select Train Dir", command=self.select_train_dir)
        self.train_dir_select_button.grid(row=15, column=0, padx=5, pady=5)

        self.quit_button = customtkinter.CTkButton(self.model_predict_frame, text="Select Test Dir", command=self.select_test_dir)
        self.quit_button.grid(row=15, column=1, padx=5, pady=5)

        self.target_size_label = customtkinter.CTkLabel(self.model_predict_frame, text="Target Size:")
        self.target_size_label.grid(row=16, column=0, sticky='w', padx=5, pady=5)

        self.target_size_entry = customtkinter.CTkEntry(self.model_predict_frame)
        self.target_size_entry.insert(0, "256")
        self.target_size_entry.grid(row=16, column=1, sticky='w', padx=5, pady=5)
        return
        
    def select_train_dir(self):
        global train_dir
        train_dir = filedialog.askdirectory(title="Select Train Dataset Directory")
        return
    def select_test_dir(self):
        global test_dir 
        test_dir = filedialog.askdirectory(title="Select Test Dataset Directory")
        return
    def train_model(self):
        # Get user inputs
        
        target_size = int(self.target_size_entry.get())
        batch_size = int(self.batch_entry.get())
        epochs = int(self.epochs_entry.get())
        optimizer = self.optimizer_var.get()
        pool_type = self.pooling_combo.get()
        loss = self.loss_combo.get()
        dropout = self.dropout_combo.get()
        self.model = create_model()
        #Train model with user inputs
        self.model.train(train_dir=train_dir, 
                         test_dir=test_dir, 
                         target_size=target_size, 
                         batch_size=batch_size, 
                         epochs=epochs, 
                         optimizer=optimizer, 
                         pool_type=pool_type, 
                         loss=loss, 
                         dropout=dropout)
        print(train_dir,test_dir,target_size,batch_size,epochs,optimizer,pool_type,loss,dropout)
        return
        

    def select_frame_by_name(self, name):
        # set button color for selected button
        self.home_button.configure(fg_color=("gray75", "gray25") if name == "home" else "transparent")
        self.model_predict_frame_button.configure(fg_color=("gray75", "gray25") if name == "model_predict_frame" else "transparent")
        self.frame_3_button.configure(fg_color=("gray75", "gray25") if name == "frame_3" else "transparent")

        # show selected frame
        if name == "home":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.home_frame.grid_forget()
        if name == "model_predict_frame":
            self.model_predict_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.model_predict_frame.grid_forget()
        if name == "frame_3":
            self.third_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.third_frame.grid_forget()

    def home_button_event(self):
        self.select_frame_by_name("home") 

    def model_predict_frame_button_event(self):
        self.select_frame_by_name("model_predict_frame")

    def frame_3_button_event(self):
        self.select_frame_by_name("frame_3")   
 
    def prediction(self,img):
        global result_image
        with tf.device('/CPU:0'):
            result_image = model.predict(img)
            self.progressbar.set(1)
            result_image = np.argmax(result_image, axis=3)
            result_image = result_image[0,:,:]
        self.original_subplot.imshow(image)
        self.mask_subplot.imshow(mask)
        self.predicted_subplot.imshow(result_image)
        self.canvas.draw()
        return

    def select_image(self):
        global image,predict_image,mask
        # Dosya seçme penceresi aç
        filepath = filedialog.askopenfilename(
            title="Resim Dosyası Seç",
            filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*"))
        )
        maskpath = filepath.replace("X_test", "y_test")
        self.progressbar.set(0.33)
        # Dosya seçilmediyse fonksiyondan çık
        if not filepath:
            self.progressbar.set(0)
            return
 
        # Buraya modelin kullanacağı işlemler yazılacak
        # Örnek olarak seçilen dosyayı alalım
        image = plt.imread(filepath)
        mask = plt.imread(maskpath)
        predict_image = np.expand_dims(image, 0)

    def model_load_button_event(self):
        threading.Thread(target=self.model_load).start()
        

    def model_load(self):
        global model
        filepath = filedialog.askopenfilename(
                title="Modeli Seç",
                filetypes=[("HDF5 files", "*.h5")]
        )
        self.progressbar.set(0.66)
            
        if not filepath:
            self.progressbar.set(0.33)
            return
        # load model here
        model = load_model(filepath, compile=False)





   
if __name__ == "__main__":
    app = App()
    app.mainloop()
