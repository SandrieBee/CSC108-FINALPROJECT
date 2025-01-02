import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
from detect import run, stop_flag
from pathlib import Path
import os
import sys
import cv2
import time

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    # If running as a PyInstaller bundle
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    # If running in a development environment
    return os.path.join(os.path.abspath("."), relative_path)

# Determine the base directory based on execution context
if getattr(sys, 'frozen', False):  # Check if running as an executable
    BASE_DIR = sys._MEIPASS  # PyInstaller's temporary directory
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Script's directory

# Define paths to resources
MODEL_PATH = resource_path("models/yolov5s.pt")
UTILS_PATH = resource_path("utils/general.py")

# Debugging: Print resolved paths
print(f"BASE_DIR: {BASE_DIR}")
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"UTILS_PATH: {UTILS_PATH}")

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}")
else:
    print("Model file found!")

# Check if the utils file exists
if not os.path.exists(UTILS_PATH):
    print(f"Utils file not found at {UTILS_PATH}")
else:
    print("Utils file found!")

# Global variables
stop_live_detection = threading.Event()
cancel_image_detection = threading.Event()
video_capture = cv2.VideoCapture(0)


class ImageObjectDetectionApp:
    def __init__(self, app):
        self.app = app
        self.cancel_button = None
        self.end_button = None
        self.result_label = None
        self.image_label = None

        self.setup_gui()

    def setup_gui(self):
        """Setup GUI components."""
        self.app.title("IMAGE AND OBJECT RECOGNITION")
        self.app.geometry("600x700")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        #header with modern font and padding
        header_label = ctk.CTkLabel(
            self.app, text="IMAGE AND OBJECT RECOGNITION", 
            font=ctk.CTkFont(size=26, weight="bold"), 
            text_color="#FFFFFF" 
        )
        header_label.pack(pady=(20, 15)) 

      
        # Button font with bold styling
        button_font = ctk.CTkFont(size=16, weight="bold")

      
        image_button = ctk.CTkButton(
            self.app,  
            text="Upload Image to Detect",
            command=self.detect_image,
            width=250,
            height=40,
            font=button_font,
            fg_color=("navy"),  
            hover_color="#0066CC",  
            text_color="white", 
            border_width=0,  
            border_color="white",  
            corner_radius=10,
        )
        image_button.pack(pady=20)

   
        live_button = ctk.CTkButton(
            self.app,
            text="Live Camera Detection",
            command=self.live_detection,
            width=250,
            height=45,
            font=button_font,
            fg_color=("navy"),  
            hover_color="#0066CC",
            text_color="white",
            corner_radius=10,
        )
        live_button.pack(pady=20)

       
        self.image_label = ctk.CTkLabel(self.app, text="", width=500, height=300, corner_radius=20, fg_color="white")
        self.image_label.pack(pady=20)

        # Result label for displaying the detection outcome
        self.result_label = ctk.CTkLabel(
            self.app, 
            text="Detection Result: None", 
            font=ctk.CTkFont(size=16, weight="normal"),  
            text_color="#A9A9A9" 
        )
        self.result_label.pack(pady=10)

    def detect_image(self):
        """handle image detection"""
    # Reset flags and ensure clean state
        cancel_image_detection.clear()
        stop_flag.clear()
        stop_live_detection.clear()

        def process_image_detection(file_path):
            try:
                # Debugging: Check file path
                print(f"Processing image: {file_path}")

                # Run YOLOv5 detection on the selected image
                run(
                    source=file_path,
                    weights=MODEL_PATH,  # Use dynamic model path
                    conf_thres=0.25,
                    iou_thres=0.45,
                    nosave=False,
                    save_txt=False,
                    save_conf=False,
                    view_img=False,
                    project=os.path.join(BASE_DIR, "runs", "detect"),
                    name="image_results",
                    exist_ok=True,
                )

                #if detection is canceled, stop further processing
                if cancel_image_detection.is_set():
                    self.update_result_label("Image detection canceled.")
                    return

                #Debug: Check result image path
                result_image_path = os.path.join(BASE_DIR, "runs", "detect", "image_results", Path(file_path).name)
                print(f"Result image path: {result_image_path}")

                #load & display the resulting image
                if os.path.exists(result_image_path):
                    img = Image.open(result_image_path).resize((600, 400))
                    img_tk = ImageTk.PhotoImage(img)
                    self.app.after(0, lambda: self.image_label.configure(image=img_tk))
                    self.app.after(0, lambda: setattr(self.image_label, "image", img_tk))
                    self.update_result_label("Detection complete. Results displayed.")
                else:
                    self.update_result_label("Error: Result image not found.")
            except Exception as e:
                self.update_result_label(f"Error during detection: {e}")
                print(f"Error during detection: {e}")
            finally:
                self.app.after(0, self.hide_cancel_button)

        #stop image detection
        def stop_image_detection():
            cancel_image_detection.set()
            self.hide_cancel_button()
            self.update_result_label("Image detection canceled.")

        #open file dialog to select img
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpeg;*.jpg;*.png")])
        if file_path:
            threading.Thread(target=process_image_detection, args=(file_path,)).start()
            self.show_cancel_button(stop_image_detection)
            self.update_result_label("Detecting image... Press 'Cancel' to stop.")
        else:
            self.update_result_label("No image selected.")

    def live_detection(self):
        """Handle live camera detection."""
        stop_live_detection.clear()

        def process_live_feed():
            try:
                run(
                    source=0,
                    weights=MODEL_PATH,  # Use dynamic path
                    view_img=True,
                    nosave=True,
                )
                while not stop_live_detection.is_set():
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                stop_flag.set()
            except Exception as e:
                self.update_result_label(f"Error during live detection: {e}")
            finally:
                self.hide_control_buttons()
                self.update_result_label("Live detection stopped.")

        def end_live_feed():
            """Stop live feed and clean up resources."""
            #signal to stop live detection
            try:
                stop_live_detection.set()
                stop_flag.set()
                print("Stop flags set successfully.")
            except NameError as e:
                print(f"Error setting stop flags: {e}")
                return

            #wait for threads to finish
            try:
                for t in threading.enumerate():
                    print(f"Active thread: {t.name}")
                    if t.name.startswith("Thread"):  # Flexible thread matching
                        print(f"Joining thread: {t.name}")
                        t.join(timeout=1)  # Adjust timeout if needed
                print("All threads joined.")
            except Exception as e:
                print(f"Error joining threads: {e}")

            #release/free the webcam
            try:
                if video_capture is not None:
                    video_capture.release()
                    time.sleep(1)  # Allow the OS to fully release the webcam
                    print("Webcam released successfully.")
                else:
                    print("No video capture object to release.")
            except Exception as e:
                print(f"Error releasing webcam: {e}")

            #close opencv windows
            try:
                if cv2.getWindowProperty("YOLO", cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyAllWindows()
                    print("OpenCV windows closed.")
                else:
                    print("No OpenCV windows to close.")
            except cv2.error as e:
                print(f"OpenCV window error: {e}")

            #update UI components
            try:
                if hasattr(self, "hide_control_buttons"):
                    self.hide_control_buttons()
                else:
                    print("hide_control_buttons method not found.")
                
                if hasattr(self, "update_result_label"):
                    self.update_result_label("Live detection ended.")
                else:
                    print("update_result_label method not found.")
            except AttributeError as e:
                print(f"Error updating UI components: {e}")

        threading.Thread(target=process_live_feed, daemon=True).start()

        #makes sure the stop button is displayed when live detection starts
        self.show_control_buttons(end_live_feed)

        self.update_result_label("Running live detection... Press 'End' to stop.")
    
    def show_cancel_button(self, command):
        print("Creating cancel button...")
        self.cancel_button = ctk.CTkButton(
            self.app, text="Cancel", command=command, width=100, height=30, fg_color="red"
        )
        self.cancel_button.pack(pady=10)

    def hide_cancel_button(self):
        if self.cancel_button and self.cancel_button.winfo_exists():
            print("Destroying cancel button...")
            self.cancel_button.destroy()
            self.cancel_button = None
        else:
            print("Cancel button already destroyed or not initialized.")

    def show_control_buttons(self, end_command):
        """display control buttons for live detection"""
        self.end_button = ctk.CTkButton(
            self.app,
            text="End Live Detection",
            command=end_command,
            width=150,
            height=30,
            fg_color="red",
            hover_color="#C82333",
            text_color="white",
        )
        self.end_button.pack(pady=10)

    def hide_control_buttons(self):
        """hide control buttons"""
        if self.end_button:
            self.end_button.destroy()
            self.end_button = None

    def update_result_label(self, text):
        """update result label"""
        if self.result_label and self.result_label.winfo_exists():
            self.result_label.configure(text=text)
        else:
            print("Result label does not exist.")
            


# Initialize the app
if __name__ == "__main__":
    app = ctk.CTk()
    ImageObjectDetectionApp(app)
    app.mainloop()
