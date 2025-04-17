# receive_stream.py
import socket
import struct
import pickle
import cv2

HOST = '172.20.10.4'
PORT = 8000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print(f"[+] Waiting for connection on {HOST}:{PORT}...")
conn, addr = server_socket.accept()
print(f"[+] Connection from {addr}")

data = b""
payload_size = struct.calcsize("Q")

while True:
    while len(data) < payload_size:
        packet = conn.recv(4*1024)
        if not packet:
            break
        data += packet

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]

    while len(data) < msg_size:
        data += conn.recv(4*1024)

    frame_data = data[:msg_size]
    data = data[msg_size:]
    frame = pickle.loads(frame_data)

    # 🎯 Process frame here (e.g., object detection)
    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

conn.close()