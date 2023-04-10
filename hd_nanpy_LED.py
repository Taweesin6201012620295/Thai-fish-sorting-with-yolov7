import time
from nanpy import SerialManager, ArduinoApi, Servo

# pin number
# servo
servo_pin = 9
# LED
led_pin1 = 2
led_pin2 = 6
led_pin3 = 7
led_pin4 = 8
led_pin5 = 10
led_pin6 = 11
#led_pin7 = 12

# IR sensor
ir_pin_right = 12
ir_pin_left = 13

# motor driver
ENA_pin = 3
IN1_pin = 4
IN2_pin = 5


connection = SerialManager(device='COM3')
a = ArduinoApi(connection=connection)
servo = Servo(servo_pin, connection=connection)
servo.write(180)

a.pinMode(ENA_pin, a.OUTPUT)
a.pinMode(IN1_pin, a.OUTPUT)
a.pinMode(IN2_pin, a.OUTPUT)

def call_arduino(pred_fish, select_fish):
    
    if pred_fish in select_fish:
        index_fish = select_fish.index(pred_fish)
        print("servo" + str(index_fish))
        # check servo pin
        a.pinMode(ir_pin_right, a.INPUT)
        a.pinMode(ir_pin_left, a.INPUT)
        while True:
            if  (a.digitalRead(ir_pin_right) == False) or (a.digitalRead(ir_pin_left) == False):
                if index_fish == 0:
                    servo = Servo(servo_pin, connection=connection)
                    servo.write(180)
                    print("open gate 0")
                    servo.write(135)
                    time.sleep(3)
                    servo.write(110)
                    time.sleep(0.5)
                    servo.write(180)
                    break
                elif index_fish == 1:
                    print("open gate 1")
                    time.sleep(5)
                    a.pinMode(led_pin1, a.OUTPUT)
                    a.digitalWrite(led_pin1, a.HIGH)
                    time.sleep(3)
                    a.digitalWrite(led_pin1,a.LOW)
                    break
                elif index_fish == 2:
                    print("open gate 2")
                    time.sleep(5)
                    a.pinMode(led_pin2, a.OUTPUT)
                    a.digitalWrite(led_pin2, a.HIGH)
                    time.sleep(3)
                    a.digitalWrite(led_pin1,a.LOW)
                    break
                elif index_fish == 3:
                    print("open gate 3")
                    time.sleep(5)
                    a.pinMode(led_pin3, a.OUTPUT)
                    a.digitalWrite(led_pin3, a.HIGH)
                    time.sleep(4)
                    a.digitalWrite(led_pin3,a.LOW)
                    break
                elif index_fish == 4:
                    print("open gate 4")
                    time.sleep(5)
                    a.pinMode(led_pin4, a.OUTPUT)
                    a.digitalWrite(led_pin4, a.HIGH)
                    time.sleep(3)
                    a.digitalWrite(led_pin4,a.LOW)
                    break
                elif index_fish == 5:
                    print("open gate 5")
                    time.sleep(5)
                    a.pinMode(led_pin5, a.OUTPUT)
                    a.digitalWrite(led_pin5, a.HIGH)
                    time.sleep(3)
                    a.digitalWrite(led_pin5,a.LOW)
                    break
                elif index_fish == 6:
                    print("open gate 6")
                    time.sleep(5)
                    a.pinMode(led_pin6, a.OUTPUT)
                    a.digitalWrite(led_pin6, a.HIGH)
                    time.sleep(3)
                    a.digitalWrite(led_pin6,a.LOW)
                    break
                elif index_fish == 7:
                    print("open gate 7")
                    """time.sleep(5)
                    a.pinMode(led_pin7, a.OUTPUT)
                    a.digitalWrite(led_pin7, a.HIGH)
                    time.sleep(3)
                    a.digitalWrite(led_pin7,a.LOW)
                    break"""
                  
    else:
        print("This fish doesn't sellect")
        

def convenyor_run():
    a.analogWrite(ENA_pin, 100)
    a.digitalWrite(IN1_pin, a.LOW)
    a.digitalWrite(IN2_pin, a.HIGH)

def convenyor_stop():
    a.analogWrite(ENA_pin, 0)
    a.digitalWrite(IN1_pin, a.LOW)
    a.digitalWrite(IN2_pin, a.HIGH)
