from nanpy import *
import time

connection = SerialManager(device='COM3') #ที่ตั้งของบอร์ด Arduino ที่เราอัพโหลด Nanpy firmware
a = ArduinoApi(connection=connection) # เรียกใช้ Arduino API
a.pinMode(12, a.INPUT) # ir sensor
servo = Servo(9, connection=connection)
servo.write(180)
a.pinMode(3, a.OUTPUT) # ENA
a.pinMode(4, a.OUTPUT) # IN1
a.pinMode(5, a.OUTPUT) # IN2
a.analogWrite(3, 100)
a.digitalWrite(4, a.LOW)
a.digitalWrite(5, a.HIGH)

while True:

    if a.digitalRead(12) == False:
        a.pinMode(13, a.OUTPUT)
        a.digitalWrite(13, a.HIGH)
        servo.write(135)
        time.sleep(4)
        servo.write(110)
        time.sleep(0.5)
        
    else:
        a.digitalWrite(13, a.LOW)
        servo.write(180)
        time.sleep(0.5)