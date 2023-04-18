import time
from nanpy import SerialManager, ArduinoApi, Servo

# pin number
# servo
servo_pin = 9

# IR sensor
ir_pin_right = 12
ir_pin_left = 13

# motor driver
ENA_pin = 3
IN1_pin = 4
IN2_pin = 5


connection = SerialManager(device='COM6')
a = ArduinoApi(connection=connection)
servo = Servo(servo_pin, connection=connection)
servo.write(180)

a.pinMode(ENA_pin, a.OUTPUT)
a.pinMode(IN1_pin, a.OUTPUT)
a.pinMode(IN2_pin, a.OUTPUT)

a.pinMode(ir_pin_right, a.INPUT)
a.pinMode(ir_pin_left, a.INPUT)

def call_arduino(pred_fish, select_fish):
    
    if pred_fish in select_fish:
        index_fish = select_fish.index(pred_fish)
        print("servo" + str(index_fish))
        # check servo pin
        while True:
            if  a.digitalRead(ir_pin_right) == False or a.digitalRead(ir_pin_left) == False:
                if index_fish == 0:
                    print("open door")
                    servo.write(135)
                    time.sleep(2)
                    servo.write(110)
                    time.sleep(0.5)
                    servo.write(180)
                    break
                elif index_fish == 1:
                    print("open gate 1")
                elif index_fish == 2:
                    print("open gate 2")
                elif index_fish == 3:
                    print("open gate 3")
                elif index_fish == 4:
                    print("open gate 4")
                elif index_fish == 5:
                    print("open gate 5")
                elif index_fish == 6:
                    print("open gate 6")
                elif index_fish == 7:
                    print("open gate 7")
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

#convenyor_run