import serial
import time

baud_rate = 9600
com = serial.Serial()
for i in range(0, 15):
    try:
        port = "COM" + str(i)
        com.port = port
        com.baudrate = baud_rate
        # Doesn't effect write issues
        com.timeout = 0.1
        # 0 = no exception but no echo, anything else
        # = instant SerialTimeoutException
        com.writeTimeout = 0
        com.setDTR(True)
        com.open()
        break
    except:
        pass

time.sleep(1)
while not com.read():
    pass

print("Serial Connected")


def send_signal(s):
    com.write(s)


def get_distance():
    d = com.readline().decode().replace("\n", "").replace("\t", "").strip()

    if len(d) == 2:
        dist = int(d[0] + d[1])
        return dist
    elif len(d) == 1:
        dist = int(d[0])
        return dist
    else:
        return 100


def is_someone_near():
    return get_distance() <= 50

# def read():
#     return com.readline()
#
#
# while True:
#
#     #send_signal('0'.encode())
#     d = read()
#
#     print(">",d,"<")
#     time.sleep(1)
