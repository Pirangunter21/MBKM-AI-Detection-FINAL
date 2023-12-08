import PySimpleGUI as sg
import PySimpleGUI as psg
import pygame
from PIL import Image
import tkinter as tk
import time
import base64
import io
import socket
import threading
import json
import numpy as np
import ctypes
import cv2
import Jetson.GPIO as GPIO


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


fontTitle = ('Helvetica', 16)
fontText = ('Helvetica', 12)

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
#          ],
          [
            sg.Text('Time', font=('Any 22'), background_color=DARK_HEADER_COLOR, size=(16,1),),
            sg.Text(':', font=('Any 22'), background_color=DARK_HEADER_COLOR),
            sg.Text('', font='Any 22', key='timetext', background_color=DARK_HEADER_COLOR, size=(61, 1))
          ],
          [
            sg.Text('Elapsed Robot Time:', font=('Any 22'), background_color=DARK_HEADER_COLOR, size=(16,1),),
            sg.Text(':', font=('Any 22'), background_color=DARK_HEADER_COLOR),
            sg.Text('', font='Any 22', key='robottime', background_color=DARK_HEADER_COLOR, size=(61, 1)),
            sg.Text('Seconds', font=('Any 22'), background_color=DARK_HEADER_COLOR)
          ],
          ],
          size=(itemWidth, itemHeight), background_color=DARK_HEADER_COLOR, pad = ((10, 0), (10, 0)))
   return compItem


def createComponet(deviceID,devicePosition,Image): 

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
deviceLocation = ('C A M 01', 'C A M 02', 'C A M 03')
ImageCamera = ('Image01','Image02','Image03')
header = [sg.Text('', font='Any 22', key='timetext', background_color=DARK_HEADER_COLOR, size=(61, 1))]
contTop = [createComponet(deviceName,deviceLocation,ImageCamera) for deviceName,deviceLocation,ImageCamera in zip(deviceList,deviceLocation,ImageCamera)]

deviceList1 = ('4', '5', 'config')
deviceLocation1 = ('C A M 04', 'C A M 05', '')
CamEnable1 = ('Cam04', 'Cam05', '')
ImageCamera1 = ('Image04','Image05', '')
contButtom = [createComponet(deviceName,deviceLocation,ImageCamera) for deviceName,deviceLocation,ImageCamera in zip(deviceList1,deviceLocation1,ImageCamera1)]

layout = [contTop, contButtom]

window = sg.Window('Quality Checker Monitor',
                layout, finalize=True,
                resizable=True,
                no_titlebar=False,
                margins=(0, 0),
                grab_anywhere=True,
                icon=icon,
                background_color='#282828',
                element_justification='c',
                location=(0, 0), right_click_menu=sg.MENU_RIGHT_CLICK_EXIT)
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

# Create a TCP/IP socket

ServerSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
hostname = socket.gethostname()
host = '192.168.1.104'
port = 5000
server_address = (host, 5000)
ServerSocket.bind(server_address)

# Listen for incoming connections
ServerSocket.listen()

captureRequest = 0
capRequest = 0



def on_new_client(client_socket, addr):


  status = "ok"
  try:
    thread = threading.Thread(target=sendData, args=(client_socket, addr))  # create the thread
    thread.start()
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
        results = str(dataJson["data"]["resultDescription"])
        my_bytes = dataJson["data"]["imageRaw"]
        imgbytes = base64.b64decode(my_bytes)
        
        #Check final results
        if results == "Botol Good":
           analyzeIO(results)
        if results == "Botol Rejected":
           analyzeIO(results)
           
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

pinInpRobot = 35
pinConfirm = 36
pinPass = 37
pinFail = 40
pinTime = 33
GPIO.setmode(GPIO.BOARD)
GPIO.setup([pinInpRobot,pinTime],  GPIO.IN)
GPIO.setup([pinConfirm, pinPass, pinFail], GPIO.OUT, initial=GPIO.HIGH)
GPIO.setwarnings(False)


dataRobotInput = 0
TimeRobotInput = 0

IOPass = False
IOFail = False


Var_outConfirm = 0
Var_outFail = 0
Var_outPass = 0 

hasil = "test"


def analyzeIO(results):
  global IOPass
  global IOFail

  global Var_outFail
  global Var_outPass

  global hasil

  if results == "Botol Rejected":
    hasil = results

    #Var_outFail +=1
    #GPIO.output(pinFail, GPIO.LOW)
    
    #IOFail = True

  if results == "Botol Good":
    hasil = results
    print(results)
    #Var_outPass +=1
    #GPIO.output(pinPass, GPIO.LOW)
    #IOPass = True
    

def time_as_int():
    return int(round(time.time() * 100))

current_time = 0
start_time = time_as_int()


def Robot_Time(Start):
  global start_time, current_time
  current_time = time_as_int() - start_time

  if Start:
    start_time = start_time + time_as_int() - 0
  else:

    start_time = 0 
    current_time = 0


def TimerIO(prevDataInp, prevCaptureRequest):

  getTimeRobot = GPIO.input(pinTime)
  if getTimeRobot != prevDataInp:
    Robot_Time(True)
  else:
    GPIO.output(pinFail, GPIO.HIGH)
    GPIO.output(pinPass, GPIO.HIGH)
    Robot_Time(False)

    prevDataInp = getTimeRobot

  return [prevDataInp, prevCaptureRequest]

def checkDataIO(prevDataInp, prevCaptureRequest):
  global Var_outConfirm
  global Var_outFail
  global Var_outPass

  global IOPass
  global IOFail

  global hasil

  actINRobot = GPIO.input(pinInpRobot)

  window['inRobot'].update(str(actINRobot))
  if actINRobot != prevDataInp:
    if  dataRobotInput == 0:
      prevCaptureRequest += 1 
      if prevCaptureRequest > 10:
        prevCaptureRequest = 0
      GPIO.output(pinConfirm, GPIO.LOW)
      if pinConfirm:
        Var_outConfirm +=1
      if "Botol Rejected" in hasil:
        print("hasil jelek")
        GPIO.output(pinFail, GPIO.LOW)
        Var_outFail +=1

      if "Botol Good" in hasil:
        print("hasil bagus")
        GPIO.output(pinPass, GPIO.LOW)
        Var_outPass +=1

#      if IOFail is True:
#        Var_outFail +=1
#        GPIO.output(pinFail, GPIO.LOW)
 #     if IOPass is True:
#        Var_outPass +=1
#        GPIO.output(pinPass, GPIO.LOW)
      window['outConfirm'].update(Var_outConfirm)
      window['outPass'].update(Var_outPass)
      window['outFail'].update(Var_outFail)
      time.sleep(0.1)
    else:
      GPIO.output(pinConfirm, GPIO.HIGH)
      window['outConfirm'].update(Var_outConfirm)
      window['outPass'].update(Var_outPass)
      window['outFail'].update(Var_outFail)

    prevDataInp = actINRobot

  return [prevDataInp, prevCaptureRequest]

while True:
  window['timetext'].update(time.strftime('%H:%M:%S'))
  event, values = window.read(timeout=50)
  window['robottime'].update('{:02d}'.format((-current_time // 10000000000) // 280))
  dataRobotInput , captureRequest = checkDataIO(dataRobotInput, captureRequest)
  TimeRobotInput, capRequest = TimerIO(TimeRobotInput, capRequest)
#  window['capture'].update(captureRequest)
  if event in (sg.WINDOW_CLOSED, 'Exit'):
    break
  elif event == 'capture':
    captureRequest = captureRequest + 1
    if captureRequest > 10:
      captureRequest = 0
    print(captureRequest)


thread.join()   
ServerSocket.close()
window.close()
