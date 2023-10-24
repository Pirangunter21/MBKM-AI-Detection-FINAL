import PySimpleGUI as sg
import io
import base64
from PIL import Image
from datetime import datetime
import cv2
import socket
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import time
import joblib
from sklearn.decomposition import PCA
from skimage.feature import graycomatrix, graycoprops
import random


BORDER_COLOR = '#2c78c9'
DARK_HEADER_COLOR = '#010101'
img_HEIGHT = 650
img_WIDTH = 850
image_SIZE = (img_WIDTH, img_HEIGHT)

################  Icon  ########################
with open("logo.png", "rb") as img_file:
    iconb64 = base64.b64encode(img_file.read())
icon = iconb64
################################################

########### Function Popup Message #############

def get_popup(Message):
  sg.popup(Message, background_color='#282828',no_titlebar=True,icon = iconb64)

def get_popup_auto(Message):
  sg.popup_auto_close(Message, background_color='#282828',no_titlebar=True,icon = iconb64, auto_close_duration=1.5)

################################################

########### convert image to base64 image #############
def get_image64(filename):
    with open(filename, "rb") as img_file:
        image_data = base64.b64encode(img_file.read())
    buffer = io.BytesIO()
    imgdata = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(imgdata))
    new_img = img.resize(image_SIZE)  # x, y
    new_img.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue())
    return img_b64

img_b64 = get_image64("Offline.jpg")
image_display = [
    [sg.Image(data=img_b64, pad=(0, 0), key='image')]
    ]

################################################


############# Save image options ###############
set_file_saving = [sg.Column(
    [
        [sg.T('D I S P L A Y  I M A G E',
              font=('Helvetica', 15, "bold"),
              text_color='#000000',background_color=BORDER_COLOR)],
        [sg.CB('Enable Camera',
               font=('Helvetica', 12),
               enable_events=True,
               k='-isRealtime-',
               background_color=BORDER_COLOR)],
        [sg.T('Save Directory:',
              font=('Helvetica', 12),
              background_color=BORDER_COLOR)],
        [sg.Input(sg.user_settings_get_entry('-locImage-', ''),
                  key='-locImage-',
                  enable_events=True,
                  disabled=True,
                  use_readonly_for_disable=False,), sg.FolderBrowse()],
        [ sg.Button('Save Image', button_color=('#000000', '#d8ff34'),size=(13, 1))]
    ], pad=((0, 0), (0, 0)), background_color=BORDER_COLOR
)]

################################################

############# TCP/IP Settings ##################


set_tcp_ip = [sg.Column(
    [
        [sg.T('T C P / I P   C O N F I G', font=('Helvetica',
              15, "bold"),text_color='#000000', background_color=BORDER_COLOR)],
        [sg.CB('enable TCP', font=('Helvetica', 12), enable_events=True, k='-isTCPActive-',
               background_color=BORDER_COLOR, default=sg.user_settings_get_entry('-isTCPActive-', ''))],
        [sg.T('TCP Server IP : Port', font=('Helvetica', 12),
              background_color=BORDER_COLOR)],
        [sg.Input(sg.user_settings_get_entry('-IPSetting-', ''), key='-IPSetting-', size=(15, 1)),
         sg.T(':', font=('Helvetica', 12), background_color=BORDER_COLOR),
         sg.Input(sg.user_settings_get_entry('-PortSetting-', ''), disabled = True,
                  key='-PortSetting-', size=(10, 1)),
         sg.B('update', key='updateIpTcpServer')
         ]
    ], pad=((0, 0), (0, 0)), background_color=BORDER_COLOR
)]

set_device_id = [sg.Column(
    [
        [sg.T('Device ID', font=('Helvetica', 16), background_color=BORDER_COLOR)],
        [sg.Input(sg.user_settings_get_entry('-deviceName-', ''), key='-deviceName-', size=(30, 1)),
         sg.B('update', key='updateDevice')]
    ], pad=((0, 0), (0, 0)), background_color=BORDER_COLOR
)]

################################################
##################    Decision   ###############
set_Decision = [sg.Column(
    [
        [sg.T('Decision',
              size=(13, 1),
              font=('Helvetica', 15, "bold"),
              background_color=BORDER_COLOR,
              key='-decisionlabel-',
                text_color='#000000',
              justification='left')],
        [sg.Button('ANALYZE', button_color=('#000000', '#d8ff34'),size=(13, 1))],
       [sg.Multiline(size=(50,10), font='Tahoma 13', k='-Output-',disabled=True, autoscroll=True)]
    ], background_color=BORDER_COLOR
)]
################################################
############# IP Host Name ##################

hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)

get_host_tcp_ip = [sg.Column(
    [
        [sg.T('H O S T   I N F O', font=('Helvetica',
              15, "bold"),text_color='#000000', background_color=BORDER_COLOR)],
        [sg.T("Hostname: ", font=('Helvetica', 12),
              background_color=BORDER_COLOR)],
        [sg.T(hostname,text_color='#000000', font=('Helvetica', 12, "bold"),
              background_color=BORDER_COLOR)],       
        [sg.T("IP Address: ", font=('Helvetica', 12),
              background_color=BORDER_COLOR)],
        [sg.T(ip_address,text_color='#000000', font=('Helvetica', 12, "bold"),
              background_color=BORDER_COLOR)],  
    ], pad=((0, 0), (0, 0)), background_color=BORDER_COLOR
)]

################################################
cnt_setting = [set_file_saving,
               get_host_tcp_ip,
               set_tcp_ip,
               set_device_id,
               set_Decision
               ]

content_layout = [
       sg.Column(cnt_setting,
              size=(500, 800),
              pad=((0, 0), (0, 0)),
              background_color=BORDER_COLOR),
    sg.Column(image_display,
              size=image_SIZE,
              pad=((0, 0), (0, 0))
              )]

layout = [content_layout]



window = sg.Window('DVI - Decal Visual Inspection',
                   layout, finalize=True,
                   resizable=False,
                   no_titlebar=False,
                   margins=(0, 0),
                   grab_anywhere=True,
                    background_color='#e8ebf3',
                   icon=icon, location=(0, 0))

element_justification='c'

################################################

########### MACHINE LEARNING CORE ##############
def process_image(image_path):
    image = Image.open(image_path)

    # Langkah 1: Potong gambar
    crop_coordinates = (1200, 1200, 2020, 1750)
    left, upper, right, lower = crop_coordinates
    cropped_image = image.crop((left, upper, right, lower))

    # Langkah 2: Konversi ke grayscale
    gray_image = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2GRAY)

    # Langkah 3: Thresholding
    thresholded_image = gray_image.copy()
    threshold_value = 120
    thresholded_image[gray_image > threshold_value] = 255

    #plt.imshow(thresholded_image, cmap='gray') 
    #plt.axis('off')
    #plt.show()

    return thresholded_image

def model_1(image):
    pca_model_1 = r'Code Philip MF/model/pca_model_1.pkl'
    classifier_model_1 = r'Code Philip MF/model/model_1_RF.h5'
    
    pca = joblib.load(pca_model_1)
    rf_classifier = joblib.load(classifier_model_1)
    
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    X = hist[20:100]
    X = X.reshape(1, -1)
    
    X_transformed = pca.transform(X)
    
    X_transformed_subset = X_transformed[:, 0:3]
    predictions = rf_classifier.predict(X_transformed_subset)

    if predictions == [0]:
        kesimpulan = 'No'
    else :
        kesimpulan = "Yes"

    return kesimpulan

def calculate_glcm_properties(image):
    distances = [1]
    
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm_properties = []
    glcm_energies = []
    glcm_homogeneities = []

    for angle in angles:
        glcm = graycomatrix(image, distances=distances, angles=[angle], symmetric=True, normed=True)
        
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        glcm_energies.append(energy)
        glcm_homogeneities.append(homogeneity)

    return glcm_energies, glcm_homogeneities


def model_2(image):
    pca_model_2 = r'Code Philip MF/model/pca_model_2.pkl'
    classifier_model_2 = r'Code Philip MF/model/model_2_RF.h5'
    
    pca = joblib.load(pca_model_2)
    rf_classifier = joblib.load(classifier_model_2)
    
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    X = hist[0:50]
    
    energy_values, homogeneity_values = calculate_glcm_properties(image)
    
    #print(X)
    #print(energy_values)
    #print(homogeneity_values)

    combined_data = np.hstack((energy_values, homogeneity_values, X))

    X = combined_data.reshape(1, -1)
    X_gabungan = pca.transform(X)        
    X_gabungan = X_gabungan[:, 0:4]
    
    predictions = rf_classifier.predict(X_gabungan)

    if predictions == [0]:
        kesimpulan = 'No'
    else :
        kesimpulan = "Yes"

    return kesimpulan

def model_3(image_asli):
    image = np.copy(image_asli)
    image[image == 255] = 0
    
    num_objects = 0
    
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corner_coordinates = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x2, y2 = x + w, y + h
        
        contour_area = cv2.contourArea(contour)
        
        min_contour_area = 1
        if contour_area >= min_contour_area:
            corner_coordinates.append(((x, y), (x2, y2)))
            
    if corner_coordinates:
        image_with_boxes = cv2.cvtColor(image_asli, cv2.COLOR_GRAY2BGR)
        for i, ((x, y), (x2, y2)) in enumerate(corner_coordinates, start=1):
            cv2.rectangle(image_with_boxes, (x, y), (x2, y2), (0, 0, 255), 2)
        
        num_objects = len(corner_coordinates)

        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
        plt.title('Vision System Detection')
        plt.axis('off')
        plt.show()
    else:
        window['-Output-'].print("\n No objects were found in the image. \n")

    return num_objects

def cek_botol(image_path):
    start_time = time.time()
    result_image = process_image(image_path)
    result_1 = model_1(result_image)
    result_2 = model_2(result_image)
    model3 = model_3(result_image)

    if model3 == 12 :
        result_3 = 'No'
    else:
        result_3 = 'Yes'

    if result_1 == 'Yes' or result_2 == 'Yes' or result_3 == 'Yes':
        kesimpulan = "Botol Rejected"
    else:
        kesimpulan = "Botol Good"

    waktu = time.time() - start_time

    window['-Output-'].print("    Philip Bottle Vision System     ")
    window['-Output-'].print("------------------------------------")
    window['-Output-'].print("")
    window['-Output-'].print("Objek Terdeteksi : ", model3)
    window['-Output-'].print("")
    window['-Output-'].print("Cat Pudar     : ", result_1)
    window['-Output-'].print("Print Kurang  : ", result_2)
    window['-Output-'].print("Over Printing : ", result_3)
    window['-Output-'].print("")
    window['-Output-'].print("------------------------------------")
    window['-Output-'].print("Kesimpulan : ", kesimpulan)
    window['-Output-'].print("Deteksi Selesai Dalam", "{:.2f}".format(waktu), "detik")

image_path = r"Code Philip MF/Sebagian Dataset yang Digunakan/reject/REJECT_AVENT20230728113951796247.jpg"
image_path = r"Code Philip MF/Sebagian Dataset yang Digunakan/good/GOOD_AVENT20230728112550235490.jpg"

################################################


############## Camera Encoding #################
cap = cv2.VideoCapture(0)
cap.set(3, 800)
cap.set(4, 800)
cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

################################################


TCPEnable = 0
host = sg.user_settings_get_entry('-IPSetting-', '')
if sg.user_settings_get_entry('-PortSetting-', '') == "":
    port = 5000
else:
    port = int(sg.user_settings_get_entry('-PortSetting-', ''))
    
client_socket = socket.socket()  # instantiate

def save_image(client_socket, directory, imageSaving):
    if values['-isRealtime-'] == True:
            ret, frame = cap.read()
            frameShow = cv2.resize(frame, image_SIZE)
            if imageSaving:
                now = datetime.now()
                filename = now.strftime("ObjectChecked_%Y%m%d%H%M%S%f") + ".png"
                new_file_name = os.path.join(directory, filename)
                cv2.imwrite(new_file_name, frameShow)
                get_popup_auto("Image Saved")
            imgbytesSend = cv2.imencode('.png', cv2.resize(frameShow, (800,600)))[1].tobytes()
            dataImage = base64.b64encode(imgbytesSend).decode('ascii')
            dataResponse = {
            "response": "complete",
            "data": {
            "deviceID": id,
            "deviceName": deviceName,
            "result": 0,
            "resultDescription": "good",
            "imageRaw": dataImage
        }
    }
    else:
        get_popup_auto("Camera is inactive! \nmake sure camera is enabled!")
    TCPdataResponse = json.dumps(dataResponse)
    client_socket.sendall(TCPdataResponse.encode())


def capture_image():
    ret, frame = cap.read()
    frame = cv2.resize(frame, image_SIZE)
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()
    dataImage = base64.b64encode(imgbytes).decode('ascii')    
    content = base64.b64decode(dataImage)
    window['image'].update(data=content)
    
def DeactivateCamera():
    imageSample = get_image64("Offline.jpg")
    window['image'].update(data=imageSample)

camera_realtime = 0
directory = sg.user_settings_get_entry('-locImage-', '')
id = 1



while True:
    event, values = window.read(timeout=20)
    if event == 'EXIT' or event == sg.WIN_CLOSED:
        break  # exit button clicked
    if camera_realtime:
       capture_image()
    if event == '-locImage-':
        sg.user_settings_set_entry('-locImage-', values['-locImage-'])
        directory = sg.user_settings_get_entry('-locImage-', '')
        window['-isTCPActive-'].update(False)
        TCPEnable = False

    if event == 'ANALYZE':
        #cek_botol(r"Code Philip MF/Sebagian Dataset yang Digunakan/good/GOOD_AVENT20230728112848956624.jpg")
        cek_botol(r"Code Philip MF/file_image/test.jpg")
    elif event == 'updateIpTcpServer':
        sg.user_settings_set_entry('-IPSetting-', values['-IPSetting-'])
        sg.user_settings_set_entry('-PortSetting-', values['-PortSetting-'])
        host = sg.user_settings_get_entry('-IPSetting-', '')
        port = int(sg.user_settings_get_entry('-PortSetting-', '')) 

    elif event == '-isTCPActive-':
        sg.user_settings_set_entry('-isTCPActive-', values['-isTCPActive-'])
        new_TCPEnable = sg.user_settings_get_entry('-isTCPActive-', '')
        if new_TCPEnable != TCPEnable:  # Check if TCPEnable changed
            TCPEnable = new_TCPEnable
            if TCPEnable:
                client_socket = socket.socket()  # instantiate
                try:
                    # connect to the server
                    get_popup_auto("Connecting device...")
                    client_socket.connect((host,port))
                except:
                    # Terjadi kesalahan, keluar dari loop
                     get_popup("Error 53 : Device can not connect to server!")
                     window['-isTCPActive-'].update(False)
            else:
                client_socket.close()
                window['-isTCPActive-'].update(False)
    elif event == '-isAIDetected-':
        Ai_detection = values['-isAIDetected-']
        if values['-isAIDetected-'] == True:
            get_popup_auto("WARNING: Detection is ON")

    elif event == 'updateDevice':
        sg.user_settings_set_entry('-deviceName-', values['-deviceName-'])
        deviceName = values['-deviceName-']

    elif event == '-isRealtime-':
        camera_realtime = values['-isRealtime-']
        if values['-isRealtime-'] == True:
            get_popup_auto("WARNING: Camera is ON")
        else:
            DeactivateCamera()
    elif event == 'Save Image':
       save_image(client_socket, directory, True)

if TCPEnable:
    client_socket.close()
window.close()
