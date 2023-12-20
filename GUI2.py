import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import cv2
import time
import datetime
import torch
import threading
from PIL import Image, ImageTk

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection App")
        self.model_path = ""
        self.video_path = ""
        self.size = 640
        self.selected_model = "YOLOv5"
        self.save_video = False

        # Left frame
        self.frame_left = ttk.Frame(self.root, style='Left.TFrame')
        self.frame_left.grid(row=0, column=0, padx=10, pady=5)

        # Create a style for text and buttons
        self.style = ttk.Style()
        font1 = ("Comic Sans MS", 12, "bold")
        font2 = ("Comic Sans MS", 10, "bold")
        self.style.configure("TLabel", font=font2, foreground='black')
        self.style.configure("TButton", font=font1, background="skyblue", foreground='black')
        self.style.configure("TRadiobutton", font=("Comic Sans MS", 12, "bold"))
        self.style.configure("TCheckbutton", font=("Comic Sans MS", 12, "bold"))
        self.style.configure("TCombobox", font=font1, background="skyblue", foreground='black')

        # Combobox for Machine
        self.machine_combobox = ttk.Combobox(self.frame_left, values=[], state="readonly", style="TCombobox")
        self.machine_combobox.set("Select Machine ID")
        self.machine_combobox.pack(fill=tk.X, padx=10, pady=5)

        # Combobox for Location
        self.location_combobox = ttk.Combobox(self.frame_left, values=[], state="readonly", style="TCombobox")
        self.location_combobox.set("Select Location")
        self.location_combobox.pack(fill=tk.X, padx=10, pady=5)
        
        # Load Models from Text File
        self.load_models_from_file()

        # Load Model Button
        self.load_model_button = ttk.Button(self.frame_left, text="Load Model", command=self.load_model)
        self.load_model_button.pack(fill=tk.X, padx=10, pady=5)

        # Model Selection Radio Buttons
        self.model_var = tk.StringVar()
        self.model_var.set("YOLOv5")
        self.yolov5_radio = ttk.Radiobutton(self.frame_left, text="YOLOv5", variable=self.model_var, value="YOLOv5")
        self.yolov5_radio.pack(fill=tk.X, padx=10, pady=5)
        self.yolov7_radio = ttk.Radiobutton(self.frame_left, text="YOLOv7", variable=self.model_var, value="YOLOv7")
        self.yolov7_radio.pack(fill=tk.X, padx=10, pady=5)

        # Inside the __init__ method, add these lines to create Entry widgets and labels
        self.position_y1_label = ttk.Label(self.frame_left, text="Position Y1:")
        self.position_y1_label.pack(fill=tk.X, padx=10, pady=5)
        self.position_y1_entry = ttk.Entry(self.frame_left)
        self.position_y1_entry.pack(fill=tk.X, padx=10, pady=5)

        self.x_min_label = ttk.Label(self.frame_left, text="X Min:")
        self.x_min_label.pack(fill=tk.X, padx=10, pady=5)
        self.x_min_entry = ttk.Entry(self.frame_left)
        self.x_min_entry.pack(fill=tk.X, padx=10, pady=5)

        self.x_max_label = ttk.Label(self.frame_left, text="X Max:")
        self.x_max_label.pack(fill=tk.X, padx=10, pady=5)
        self.x_max_entry = ttk.Entry(self.frame_left)
        self.x_max_entry.pack(fill=tk.X, padx=10, pady=5)


        # Save Video Checkbox
        self.save_video_var = tk.IntVar()
        self.save_video_checkbox = ttk.Checkbutton(self.frame_left, text="Simpan Video", variable=self.save_video_var, onvalue=1, offvalue=0)
        self.save_video_checkbox.pack(fill=tk.X, padx=10, pady=5)

        # Start Detection Button
        self.start_detection_button = ttk.Button(self.frame_left, text="Start Counting", command=self.start_detection)
        self.start_detection_button.pack(fill=tk.X, padx=10, pady=5)

        # Save Text Button
        self.save_txt_button = ttk.Button(self.frame_left, text="Done", command=self.restart_gui)
        self.save_txt_button.pack(fill=tk.X, padx=10, pady=5)

        # Counter Labels
        self.counters = {
            'b': 0,
            'c': 0,
            'm': 0,
            't': 0
        }
        self.counter_labels = {}
        self.counttotal = 0

        self.counter_label_total = ttk.Label(self.frame_left, text="Total: 0")
        self.counter_label_total.pack(fill=tk.X, padx=10, pady=10)

        for class_name in self.counters.keys():
            self.counter_labels[class_name] = ttk.Label(self.frame_left, text=f"{class_name.capitalize()}: 0")
            self.counter_labels[class_name].pack(fill=tk.X, padx=10, pady=10)

        # Right frame
        self.frame_right = ttk.Frame(self.root, style='Right.TFrame')
        self.frame_right.grid(row=0, column=1, padx=10, pady=5)

        # Video Display Canvas
        self.canvas = tk.Canvas(self.frame_right, width=640, height=640)
        self.canvas.pack()

        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.video_thread = None
        self.stopped = False
        
        # File .txt
        self.txt_filename = f"hasil{self.timestamp.replace(':', '-')}.txt"
        
    def load_models_from_file(self):
        # Lokasi file teks yang berisi daftar model
        machine_file_path = 'lokasi.txt'
        location_file_path = 'koordinat.txt'
        
        # Membaca data dari file teks
        machine_data = self.load_data_from_file(machine_file_path)
        location_data = self.load_data_from_file(location_file_path)
        # Mengisi Combobox dengan data model
        self.machine_combobox['values'] = machine_data
        self.location_combobox['values'] = location_data
        
    def load_data_from_file(self, file_path):
        try:
            with open(file_path, 'r') as file:
                data = file.read().splitlines()
            return data
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return []

    def load_model(self):
        self.model_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.pt")])
        messagebox.showinfo("Load Model", "Model Loaded")
        
    def start_detection(self):
        
        if not self.model_path:
            return
        
        self.save_video = bool(self.save_video_var.get())

        model_name = self.model_var.get()
        if model_name == "YOLOv5":
            self.model = self.get_yolov5_model(self.model_path)
        elif model_name == "YOLOv7":
            self.model = self.get_yolov7_model(self.model_path)
        else:
            return

        cap = cv2.VideoCapture(0)
        fps = cap.get(cv2.CAP_PROP_FPS)

        self.model.names[0] = 'b'
        self.model.names[1] = 'c'
        self.model.names[2] = 'm'
        self.model.names[3] = 't'
        
        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_filename = f"output_{self.timestamp.replace(':', '-')}.mp4"
            out = cv2.VideoWriter(self.video_filename, fourcc, fps, (640, 640))
        
        def update_video():
            count = 0
            offset = 6
            position_y1 = int(self.position_y1_entry.get())
            x_min = int(self.x_min_entry.get())
            x_max = int(self.x_max_entry.get())
            
            while not self.stopped:
                ret, img = cap.read()
                
                if not ret:
                    break  
                
                count += 1
                if count % 4 != 0:
                    continue
                img = cv2.resize(img, (640, 640))
                results = self.model(img, self.size)

                class_counts = {class_name: 0 for class_name in self.counters.keys()}
                cv2.line(img,(x_min,position_y1),(x_max,position_y1),(255,255,255),1)
                
                
                for index, row in results.pandas().xyxy[0].iterrows():
                    x1 = int(row['xmin'])
                    y1 = int(row['ymin'])
                    x2 = int(row['xmax'])
                    y2 = int(row['ymax'])
                    detect = row['class']

                    rectx1, recty1 = ((x1 + x2) / 2, (y1 + y2) / 2)
                    rectcenter = int(rectx1), int(recty1)
                    position_x = rectcenter[0]
                    position_y = rectcenter[1]
                    cv2.circle(img, (position_x, position_y), 3, (0, 0, 255), -1)

                    class_name = self.model.names[detect]
                    class_counts[class_name] += 1

                    if detect == 0:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                
                    elif detect == 1:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    elif detect == 2:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    else: 
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    if (position_y >= position_y1 - offset and position_y <= position_y1 + offset) and (position_x >= x_min and position_x <= x_max):
                        self.counttotal +=1
                        self.counters[class_name] += 1
                        self.update_counter_labels()
                        self.update_counter_total()
                        cv2.line(img,(x_min,position_y1),(x_max,position_y1),(255,255,255),1)
                        cv2.putText(img,str(self.counttotal),(x1,y1), cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
                        
                self.show_frame(img)
             
                if self.save_video:
                    out.write(img)
            
            if self.save_video:
                out.release()

        if self.video_thread is None or not self.video_thread.is_alive():
            self.video_thread = threading.Thread(target=update_video)
            self.video_thread.daemon = True
            self.video_thread.start()

    def get_yolov5_model(self, model_path):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        return model

    def get_yolov7_model(self, model_path):
        model = torch.hub.load('WongKinYiu/yolov7', 'custom', model_path)
        return model

    def show_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.canvas.photo = photo

    def update_counter_labels(self):
        for class_name, count in self.counters.items():
            self.counter_labels[class_name].config(text=f"{class_name.capitalize()}: {count}")
            
    def update_counter_total(self):
        self.counter_label_total.config(text="Total: " + str(self.counttotal))
        
        
    def restart_gui(self):
        self.stopped = True
        selected_machine = self.machine_combobox.get()
        selected_location = self.location_combobox.get()
        
        with open(self.txt_filename, 'w') as txt_file:
            txt_file.write(f"Hasil Perhitungan {selected_machine} {selected_location} {self.timestamp}\n")
            for class_name, count in self.counters.items():
                txt_file.write(f"{class_name.capitalize()}: {count}\n")
            txt_file.write(f"Total: {self.counttotal}\n")
        
        messagebox.showinfo("Restart GUI", "GUI Restarted")
        self.root.destroy()
        
        root = tk.Tk()
        root.title("Perhitungan Kendaraan")
        root.maxsize(900, 800)
        root.config(bg="skyblue")

        # Create left and right frames
        left_frame = ttk.Frame(root, width=200, height=800, style='Left.TFrame')
        left_frame.grid(row=0, column=0, padx=10, pady=5)

        right_frame = ttk.Frame(root, width=640, height=640, style='Right.TFrame')
        right_frame.grid(row=0, column=1, padx=10, pady=5)

        # Create and run ObjectDetectionApp
        app = ObjectDetectionApp(root)
        root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Perhitungan Kendaraan")
    root.maxsize(900, 800)
    root.config(bg="skyblue")

    # Create left and right frames
    left_frame = ttk.Frame(root, width=200, height=800, style='Left.TFrame')
    left_frame.grid(row=0, column=0, padx=10, pady=5)

    right_frame = ttk.Frame(root, width=640, height=640, style='Right.TFrame')
    right_frame.grid(row=0, column=1, padx=10, pady=5)

    # Create and run ObjectDetectionApp
    app = ObjectDetectionApp(root)
    root.mainloop()
