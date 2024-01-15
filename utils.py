import sys
import time


def simulate_typing(text, speed=0.05):
    print("ChatBot: ", end="", flush=True)
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(speed)
