import time
from nanpy import SerialManager, ArduinoApi, Servo

# pin number
# servo
servo_pin0 = 9
servo_pin1 = 10
servo_pin2 = 10
servo_pin3 = 10
servo_pin4 = 10
servo_pin5 = 10
servo_pin6 = 10
servo_pin7 = 10
# IR sensor
ir_pin0 = 12
ir_pin1 = 13
ir_pin2 = 13
ir_pin3 = 13
ir_pin4 = 13
ir_pin5 = 13
ir_pin6 = 13
ir_pin7 = 13
# motor driver
ENA_pin = 3
IN1_pin = 4
IN2_pin = 5


connection = SerialManager(device='COM3')
a = ArduinoApi(connection=connection)

a.pinMode(ENA_pin, a.OUTPUT)
a.pinMode(IN1_pin, a.OUTPUT)
a.pinMode(IN2_pin, a.OUTPUT)

def call_arduino(pred_fish, select_fish):
    
    if pred_fish in select_fish:
        index_fish = select_fish.index(pred_fish)
        print("servo" + str(index_fish))
        # check servo pin
        if index_fish == 0:
            servo_pin = servo_pin0
            ir_pin = ir_pin0
        elif index_fish == 1:
            servo_pin = servo_pin1
            ir_pin = ir_pin1
        elif index_fish == 2:
            servo_pin = servo_pin2
            ir_pin = ir_pin2
        elif index_fish == 3:
            servo_pin = servo_pin3
            ir_pin = ir_pin3
        elif index_fish == 4:
            servo_pin = servo_pin4
            ir_pin = ir_pin4
        elif index_fish == 5:
            servo_pin = servo_pin5
            ir_pin = ir_pin5
        elif index_fish == 6:
            servo_pin = servo_pin6
            ir_pin = ir_pin6
        elif index_fish == 7:
            servo_pin = servo_pin7
            ir_pin = ir_pin7

        servo = Servo(servo_pin, connection=connection)
        servo.write(180)
        a.pinMode(ir_pin, a.INPUT)

        #a.pinMode(index_fish+6, a.INPUT) # IR sensor
        while True:
            if a.digitalRead(ir_pin) == False: #edit ir pin/servo for each type of fish(idex+6)
                servo.write(135)
                time.sleep(4)
                servo.write(110)
                time.sleep(0.5)
                servo.write(180)
                break
    else:
        print("Error Fish")

def convenyor_run():
    a.analogWrite(ENA_pin, 100)
    a.digitalWrite(IN1_pin, a.LOW)
    a.digitalWrite(IN2_pin, a.HIGH)

def convenyor_stop():
    a.analogWrite(ENA_pin, 0)
    a.digitalWrite(IN1_pin, a.LOW)
    a.digitalWrite(IN2_pin, a.HIGH)