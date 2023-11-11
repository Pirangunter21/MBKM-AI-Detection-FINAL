import PySimpleGUI as sg
import PySimpleGUI as psg
import pygame
from PIL import ImageTk
import tkinter as tk
import time
import base64
from PIL import Image
import io
import socket
import threading
import json
import numpy as np
import ctypes
import cv2

ctypes.windll.user32.SetProcessDPIAware()
pygame.init()
infos = pygame.display.Info()

width_display = (infos.current_w,infos.current_h)
print(width_display)
BORDER_COLOR = '#101010'
DARK_HEADER_COLOR = '#1f1f1f'

with open("Logo.png", "rb") as img_file:
    iconb64 = base64.b64encode(img_file.read())
icon = iconb64

itemWidth = (width_display[0] - 32) / 3
itemHeight = (width_display[1] - 32) / 2
sizeTitle = (int(itemWidth * 162 / width_display[0]) , 1)
image_SIZE = (800, 600)


fontTitle = ('Impact', 16)
fontText = ('Arial', 12)

sizeResult = (int(itemWidth * 500 / width_display[0]) , 1)
fontResult = ('Helvetica', 20)

def get_popup(Message):
  sg.popup_auto_close(Message, background_color='#282828',no_titlebar=True,icon = iconb64, auto_close_duration=1.5)

def get_ERROR_popup(Message):
  sg.popup(Message, background_color='#282828',no_titlebar=True,icon = iconb64)

def get_image64(filename):
    with open(filename, "rb") as img_file:
        image_data = base64.b64encode(img_file.read())
    buffer = io.BytesIO()
    imgdata = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(imgdata))
    new_img = img.resize((800,600))  # x, y
    new_img.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue())
    return img_b64

imageSample = get_image64("Offline.jpg")
#imageChange = get_image64("images.png") 

img_b64 = imageSample 
  
def createSetting():
   compItem = sg.Column([
          [ sg.Text('In Robot (P35)', font=('Any 22'), background_color=DARK_HEADER_COLOR, size=(16,1),),
            sg.Text(':', font=('Any 22'), background_color=DARK_HEADER_COLOR),
            sg.Text('0' , font='Any 22', key='inRobot', background_color=DARK_HEADER_COLOR)
          ],
          [ sg.Text('Out Confirm (P36)', font=('Any 22'), background_color=DARK_HEADER_COLOR, size=(16,1),),
            sg.Text(':', font=('Any 22'), background_color=DARK_HEADER_COLOR),
            sg.Text('0' , font='Any 22', key='outConfirm', background_color=DARK_HEADER_COLOR)
          ],
          [ sg.Text('Out Pass(P37)' , font=('Any 22'), background_color=DARK_HEADER_COLOR, size=(16,1),),
            sg.Text(':', font=('Any 22'), background_color=DARK_HEADER_COLOR),
            sg.Text('0' , font='Any 22', key='outPass', background_color=DARK_HEADER_COLOR)
          ],
          [ sg.Text('Out Fail (P40)', font=('Any 22'), background_color=DARK_HEADER_COLOR, size=(16,1),),
            sg.Text(':', font=('Any 22'), background_color=DARK_HEADER_COLOR),
            sg.Text('0' , font='Any 22', key='outFail', background_color=DARK_HEADER_COLOR)
          ],
#          [ sg.Text('Column2', background_color='green', size=(10,2)),
#            sg.Button('capture', key='capture', button_color=('white', 'firebrick3'))                               
#          ]
          ],
          size=(itemWidth, itemHeight), background_color=DARK_HEADER_COLOR, pad = ((10, 0), (10, 0)))
   return compItem


def createComponet(deviceID,devicePosition,EnableCam,Image): 

    compItem = sg.Column([
       [sg.T(devicePosition, font= fontTitle, background_color=BORDER_COLOR, size=sizeTitle, justification='c', pad = (0,0))], 
        [sg.T('Device Name: '+ deviceID, font= fontText, background_color=DARK_HEADER_COLOR),
#        sg.Input(sg.user_settings_get_entry('-dev1-', ''), key='-dev1-', size=(15, 1), font= fontText),
#         sg.B('Update', key=updDev1, font= fontText,button_color='#2e2e2e'), 
#          sg.CB("Enable Static Camera", font=fontText, enable_events=True, key=EnableCam, background_color=DARK_HEADER_COLOR, default=False)
        ], [sg.T('Result', font= fontResult, background_color=DARK_HEADER_COLOR, size=sizeResult, justification='c', pad = (0,0))],
        [sg.Image(data=imageSample, pad=(0, 0), key='image' + deviceID, size=(itemWidth, itemHeight))]
      ],size=(itemWidth, itemHeight), background_color=DARK_HEADER_COLOR, pad = ((2, 0), (2, 0)))
    
    if deviceID == "config":
        compItem = createSetting()
  
    return compItem

#contTop = [frontItem, leftItem, rightItem]
deviceList = ('1', '2', '3')
deviceLocation = ('T O P  C H E C K I N G', 'L E F T C H E C K I N G', 'B A CK  C H E C K I N G')
ImageCamera = ('Image01','Image02','Image03')
CamEnable = ('Cam01', 'Cam02', 'Cam03')
contTop = [createComponet(deviceName,deviceLocation,CamEnable,ImageCamera) for deviceName,deviceLocation,CamEnable,ImageCamera in zip(deviceList,deviceLocation,CamEnable,ImageCamera)]

deviceList1 = ('4', '5', 'config')
deviceLocation1 = ('B O T T O M  C H E C K I NG', 'R I G H T  C H E C K I N G', '')
CamEnable1 = ('Cam04', 'Cam05', '')
ImageCamera1 = ('Image04','Image05', '')
contButtom = [createComponet(deviceName,deviceLocation,CamEnable,ImageCamera) for deviceName,deviceLocation,CamEnable,ImageCamera in zip(deviceList1,deviceLocation1,CamEnable1,ImageCamera1)]

layout = [contTop, contButtom]

window = sg.Window('Quality Checker Monitor',
                layout, finalize=True,
                resizable=True,
                no_titlebar=False, #SHOW TITLE BAR
                margins=(0, 0),
                grab_anywhere=True,
                icon=icon,
                background_color='#282828',
                element_justification='c',
                location=(0, 0), right_click_menu=sg.MENU_RIGHT_CLICK_EDITME_VER_EXIT)
window.maximize()

def GetCamera(ScreenName,CameraPlacement):
  cam = cv2.VideoCapture(CameraPlacement)
  if cam is None or not cam.isOpened():
     get_ERROR_popup("(ERROR CODE 45) Camera is not detected \nmake sure to install correctly.")
  else:
    img_counter = 0
    ret, frame = cam.read()
    k = cv2.waitKey(1)
    frame = cv2.resize(frame, image_SIZE)
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
    window[ScreenName].update(data=imgbytes)
  cam.release()

def DeactivateCamera(ScreenName):
  window[ScreenName].update(data=img_b64)













# Create a TCP/IP socket

ServerSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = '10.29.202.217'
port = 5000
server_address = ('10.29.202.217', 5000)
ServerSocket.bind(server_address)

# Listen for incoming connections
ServerSocket.listen()

captureRequest = 0

def on_new_client(client_socket, addr):
  
  status = "ok"
  try:
    thread = threading.Thread(target=sendData, args=(client_socket, addr))  # create the thread
    thread.start()  #
    dataAll = ""
    status = "ok"
    number = 0
    BUFF_SIZE = 1024
    while True:
      data = client_socket.recv(BUFF_SIZE).decode('utf-8')
      if not data:
        break        
      
      dataAll += data
      number += 1
      try:
        dataJson = json.loads(dataAll)
        deviceID = str(dataJson["data"]["deviceID"])
        my_bytes = dataJson["data"]["imageRaw"]
        imgbytes = base64.b64decode(my_bytes)
        
        #change image size
        base64Data = base64.b64encode(imgbytes)
        decoded_data = base64.b64decode(base64Data)
        np_data = np.fromstring(decoded_data,np.uint8)
        img = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
        imgbytesSend = cv2.imencode('.png', cv2.resize(img, (650,400)))[1].tobytes()  # ditto
        dataImage = base64.b64encode(imgbytesSend).decode('ascii')

        window['image' + deviceID].update(data=dataImage)
        #print(f"{addr} >> {dataAll}")
        print(f"{addr} >> {number}")
        dataAll = ""
        number = 0
      except:         
        status = "reading"
        
      time.sleep(0.0001)  

    client_socket.close()
    thread.join()
  except:      
    status = "fail read data"
    print(status)

def sendData(client_socket, addr):  
  checkData = 0
  print(addr[0])
  while True:
    time.sleep(0.05)
    if checkData != captureRequest:
      try:
        checkData = captureRequest
        datajson = {"request" : "checking"}
        sendJson = json.dumps(datajson)
        client_socket.sendall(sendJson.encode())
        print("changeData")
      except:
         print(f"disconnect {addr} ")
         break

def s_changes():
  while True:
    Client, address = ServerSocket.accept()
    print('Connected to: ' + address[0] + ':' + str(address[1]))
    thread = threading.Thread(target=on_new_client, args=(Client, address))  # create the thread
    thread.start()  # start the thread` 1 `


thread = threading.Thread(target=s_changes)
thread.daemon = True
thread.start()

"""def on_new_client(client_socket, addr):
  thread = threading.Thread(target=sendData, args=(client_socket, addr))  # create the thread
  thread.start()  #

  dataAll = recvall(client_socket)
  dataJson = json.loads(dataAll)
  deviceID = str(dataJson["data"]["deviceID"])
  my_bytes = dataJson["data"]["imageRaw"]
  imgbytes = base64.b64decode(my_bytes)
  
  #change image size
  base64Data = base64.b64encode(imgbytes)
  decoded_data = base64.b64decode(base64Data)
  np_data = np.fromstring(decoded_data,np.uint8)
  img = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
  imgbytesSend = cv2.imencode('.png', cv2.resize(img, (650,400)))[1].tobytes()  # ditto
  dataImage = base64.b64encode(imgbytesSend).decode('ascii')

  #window[ScreenName + deviceID].update(data=dataImage)
  time.sleep(0.05)   

    
    
  client_socket.close()
  thread.join()"""

running01,running02,running03,running04,running05 = False,False,False,False,False

while True:
#  event, values = window.read(timeout=50)

#  window['capture'].update(captureRequest)

  event, values = window.read()
  if event in (sg.WINDOW_CLOSED, 'Exit'):
    break
#  elif event == 'capture':
#    captureRequest = captureRequest + 1
#    if captureRequest > 10:
#      captureRequest = 0
#    print(captureRequest)

  if values['Cam01'] == True:
    running01 = True
    get_popup("Running Camera 1")
    GetCamera('Image01',0)
  elif values['Cam01'] == False and running01:
    running01 = False
    get_popup("WARNING : Deactivating CAMERA 1")
    DeactivateCamera('Image01')

  if values['Cam02'] == True and not running02:
    running02 = True
    get_popup("Running Camera 2")
    GetCamera('Image02',1)
  elif values['Cam02'] == False and running02:
    running02 = False
    get_popup("WARNING : Deactivating CAMERA 2")
    DeactivateCamera('Image02')

  if values['Cam03'] == True and not running03:
    running03 = True
    print('test')
    get_popup("Running Camera 3")
    GetCamera('Image03',2)
  elif values['Cam03'] == False and running03:
    running03 = False
    get_popup("WARNING : Deactivating CAMERA 3")
    DeactivateCamera('Image03')

  if values['Cam04'] == True and not running04:
    running04 = True
    get_popup("Running Camera 4")
    GetCamera('Image04',3)
  elif values['Cam04'] == False and running04:
    running04 = False
    get_popup("WARNING : Deactivating CAMERA 4")
    DeactivateCamera('Image04')

  if values['Cam05'] == True and not running05:
    running05 = True
    print('test')
    get_popup("Running Camera 5")
    GetCamera('Image05',4)
  elif values['Cam05'] == False and running05:
    running05 = False
    get_popup("WARNING : Deactivating CAMERA 5")
    DeactivateCamera('Image05')

  elif event == 'capture':
      captureRequest = captureRequest + 1
      print(captureRequest)

window.close()