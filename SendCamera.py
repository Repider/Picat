# send_camera.py
import cv2
import socket
import struct
import pickle

camera = cv2.VideoCapture(0)  # Use libcamera apps or PiCamera2 if needed
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('172.20.10.3', 8000))

while True:
    ret, frame = camera.read()
    if not ret:
        break
    # Optionally resize or compress
    data = pickle.dumps(frame)
    message = struct.pack("Q", len(data)) + data
    client_socket.sendall(message)