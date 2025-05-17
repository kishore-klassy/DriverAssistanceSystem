import pygame

pygame.mixer.init()
pygame.mixer.music.load("assets/warning-alarm.wav")

def play_alarm():
    pygame.mixer.music.play()

def play_alarm_loop():
    pygame.mixer.music.play(-1)

def stop_alarm():
    pygame.mixer.music.stop()
