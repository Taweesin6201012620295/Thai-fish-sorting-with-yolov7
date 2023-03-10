import time
from nanpy import *

selected_fishes = ['pod', 'ku_lare', 'see_kun', 'too', 'hang_lueang', 'khang_pan', 'sai_dang', 'sai_dum']
#pred_fish = "pod"

#state = []

connection = SerialManager(device='COM3')
a = ArduinoApi(connection=connection)

servo = Servo(9, connection=connection)
servo.write(0)
a.pinMode(12, a.INPUT) # IR sensor

a.pinMode(3, a.OUTPUT) # ENA
a.pinMode(4, a.OUTPUT) # IN1
a.pinMode(5, a.OUTPUT) # IN2

def call_arduino(pred_fish):
    
    if pred_fish in selected_fishes:
        index_fish = selected_fishes.index(pred_fish)
        out_val = "servo" + str(index_fish)
        print(out_val)
        #a.pinMode(index_fish+6, a.INPUT) # IR sensor
        while True:
            if a.digitalRead(12) == False: #edit ir pin/servo for each type of fish(idex+6)
                servo.write(60)
                time.sleep(7)
                servo.write(0)
                break
    else:
        print("Error Fish")
        servo.write(0)

def convenyor():
    while True:
        a.analogWrite(3, 80)
        a.digitalWrite(4, a.LOW)
        a.digitalWrite(5, a.HIGH)
        '''time.sleep(2)
        a.analogWrite(3, 0)
        a.digitalWrite(4, a.LOW)
        a.digitalWrite(5, a.HIGH)
        time.sleep(2)'''

def convenyor_run():
    a.analogWrite(3, 100)
    a.digitalWrite(4, a.LOW)
    a.digitalWrite(5, a.HIGH)

def convenyor_stop():
    a.analogWrite(3, 0)
    a.digitalWrite(4, a.LOW)
    a.digitalWrite(5, a.HIGH)
