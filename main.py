from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label

import datetime
import math
import operator
import time
import pyttsx3
engine = pyttsx3.init("sapi5")
import cv2
import speech_recognition as sr
import torch
import wikipedia
from PIL import Image
import easyocr
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast
from ultralytics import YOLO

class MyApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        label = Label(text='Rika AI', font_size=60, size=(200, 50),
                      pos_hint={'center_x': 0.5, 'center_y': 2.0}, bold=True, size_hint_y=2)
        label.padding = (0, 0, 0, 20)

        # Create an empty widget to push the label and button to the center
        layout.add_widget(Label())

        # Add the label to the layout
        layout.add_widget(label)

        # Create a button
        btn = Button(text='Activate', size_hint=(None, None), size=(190, 190),
                     pos_hint={'center_x': 0.5, 'center_y': 0.5},color='white',
                     background_color='gray', bold=True, font_size=25)
        btn.bind(on_press=self.on_button_click)

        label2 = Label(text='', font_size=60, size=(200, 50),
                      pos_hint={'center_x': 0.5, 'center_y': 2.0}, bold=True, size_hint_y=2)
        label2.padding = (0, 0, 0, 20)

        # Add the button to the layout
        layout.add_widget(btn)

        layout.add_widget(label2)

        return layout

    def on_button_click(self, instance):
        model2 = YOLO("yolo-Weights/yolov8n.pt")

        classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                      "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                      "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                      "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                      "baseball bat",
                      "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                      "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                      "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                      "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                      "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                      "teddy bear", "hair drier", "toothbrush"
                      ]

        listener = sr.Recognizer()
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)
        engine.setProperty('voice', 110)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)

        tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        hello_rika = 0

        def image_caption():
            talk('Fine')
            camera = cv2.VideoCapture(0)

            _, image = camera.read()
            cv2.imshow('Image', image)
            cv2.imwrite('test2.jpg', image)

            camera.release()
            cv2.destroyAllWindows()
            image_path = Image.open("test2.jpg")
            img = image_processor(image_path, return_tensors="pt").to(device)

            # Generating captions
            output = model.generate(**img)

            # decode the output
            caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            print(caption)
            talk(caption)

        def text_reader():
            talk(' Sure ')
            print('Sure')

            camera = cv2.VideoCapture(0)
            _, image = camera.read()
            cv2.imshow('Image', image)
            cv2.imwrite('test2.jpg', image)
            camera.release()
            cv2.destroyAllWindows()

            reader = easyocr.Reader(['en', 'ru'])
            results = reader.readtext('test2.jpg', detail=0, paragraph=True)

            if not results or results == '':
                talk('Nothing has been detected')
                print('Nothing has been detected')
            else:
                text = results
                print(text)
                talk(text)

        def object_detector():
            talk(" Okay ")

            cap = cv2.VideoCapture(0)
            cap.set(3, 640)
            cap.set(4, 480)

            # object classes
            start_time = time.time()

            while True:
                success, img = cap.read()
                results = model2(img, stream=True)

                # coordinates

                for r in results:
                    boxes = r.boxes

                    for box in boxes:
                        # bounding box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                        # box in cam
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                        # confidence
                        confidence = math.ceil((box.conf[0] * 100)) / 100
                        print("Confidence:", confidence)

                        # class name
                        cls = int(box.cls[0])
                        print("Class name:", classNames[cls])
                        # object details
                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2

                        cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                        if confidence > 0.8:
                            talk("detected" + classNames[cls])

                cv2.imshow('Webcam', img)
                if time.time() - start_time > 30:
                    talk("I am sorry to inform you but time limit reached. Exiting. ")
                    break

            cap.release()
            cv2.destroyAllWindows()

        def talk(text):
            engine.say(text)
            engine.runAndWait()

        def take_command():
            try:
                with sr.Microphone() as source:
                    print('listening...')
                    voice = listener.listen(source)
                    command = listener.recognize_google(voice)
                    command = command.lower()
                    if 'rika' in command:
                        command = command.replace('rika', '')
                        print(command)
                    elif 'erica' in command:
                        command = command.replace('erica', '')
                        print(command)
                    elif 'rica' in command:
                        command = command.replace('rica', '')
                        print(command)
            except:
                pass
            return command

        def get_operator_fn(op):
            return {
                '+': operator.add,
                '-': operator.sub,
                'x': operator.mul,
                'divided': operator.__truediv__,
                'Mod': operator.mod,
                'mod': operator.mod,
                '^': operator.xor,
            }[op]

        def eval_binary_expr(op1, oper, op2):
            op1, op2 = int(op1), int(op2)
            return get_operator_fn(oper)(op1, op2)

        def math_solver():
            print('I am ready')
            talk(' I am ready ')
            while True:
                command2 = take_command()
                if 'stop' in command2:
                    print(' Well enough ')
                    talk(' Well enough ')
                    break
                asd = eval_binary_expr(*(command2.split()))
                print(asd)
                talk(asd)

        def run_rika():
            command = take_command()
            if 'time' in command:
                time = datetime.datetime.now().strftime('%I:%M %p')
                talk('Current time is ' + time)
            elif 'who is' in command:
                wiki = command.replace('who is', '')
                info = wikipedia.summary(wiki, 1)
                print(info)
                talk(info)
            elif 'what is' in command:
                wiki = command.replace('what is', '')
                info = wikipedia.summary(wiki, 1)
                print(info)
                talk(info)
            elif 'text' in command:
                text_reader()
            elif 'object' in command:
                object_detector()
            elif 'image' in command:
                image_caption()
            elif 'math' in command or 'mass' in command:
                math_solver()
            elif "goodbye" in command or 'stop' in command or 'finish' in command:
                print('Goodbye')
                talk(' Goodbye ')
                exit()
            else:
                return run_rika()

        while True:
            if hello_rika == 0:
                print('Hello. I am Rika AI. How I can help you?')
                talk(' Hello. I am Rika AI. How I can help you? ')
                hello_rika = 1
            elif hello_rika == 1:
                talk('Anything else')
            run_rika()


if __name__ == '__main__':
    app = MyApp()
    app.run()
