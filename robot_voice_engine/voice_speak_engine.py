import pyttsx3
from playsound import playsound
#from arduino_communication import send_signal
import time
# from mutagen.mp3 import MP3


class VoiceSpeachEngine:
    def __init__(self):
        # Initialize the converter
        self.speak_engine = pyttsx3.init()
        self.speak_engine.setProperty('rate', 150)
        # Set volume 0-1
        self.speak_engine.setProperty('volume', 1)

    def say_to_wear_mask(self, face_id):
        #send_signal('1'.encode())
        # Queue the entered text
        self.speak_engine.say("Hi " + face_id)
        self.speak_engine.say("Please wear a face mask to get access!")

        # Empties the say() queue
        self.speak_engine.runAndWait()

        #send_signal('0'.encode())

    def say_welcome(self, face_id):
        #send_signal('2'.encode())
        #time.sleep(3)
        #send_signal('1'.encode())
        # Queue the entered text
        self.speak_engine.say("Welcome " + face_id + "Now you can enter, Thanks!")

        # Empties the say() queue
        self.speak_engine.runAndWait()

        #send_signal('0'.encode())
        #send_signal('3'.encode())

    def say_not_allowed(self):
        #send_signal('4'.encode())
        #send_signal('1'.encode())
        self.speak_engine.say("Unauthorized Not Allowed, Thanks!")
        self.speak_engine.runAndWait()
        #send_signal('0'.encode())

    def say_alert_for_spoofing_attack(self):
        self.speak_engine.say('Alert Alert, intruder seems to be spoofing attacker.')
        self.speak_engine.runAndWait()

    def say_hello_and_ask_for_facemask_female_voice(self, speech):
        #send_signal('1'.encode())

        playsound(speech)

        #send_signal('0'.encode())

    def say_welcome_female_voice(self, speech):
        #send_signal('2'.encode())
        #time.sleep(3)
        #send_signal('1'.encode())

        playsound(speech)

        #send_signal('0'.encode())
        #send_signal('3'.encode())

    def say_not_allowed_female_voice(self, speech):
        #send_signal('4'.encode())
        #send_signal('1'.encode())

        playsound(speech)

        #send_signal('0'.encode())

    def say_alert_for_spoofing_attack_female_voice(self, speech):
        #send_signal('4'.encode())
        #send_signal('1'.encode())

        playsound(speech)

        #send_signal('0'.encode())
