import serial
import time

class Gripper:
    def __init__(self, device="/dev/ttyUSB0", boudrate=115200, timeout=1):
        self._serial = serial.Serial(device, boudrate, timeout=timeout)

    def send(self, state):
        '''<0 - close, >0 - open'''
        if state < 0:
            self._serial.write(b'Close\n')
        else: 
            self._serial.write(b'Open\n')



        
