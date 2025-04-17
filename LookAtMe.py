# laptop_camera_client.py
import socket
import cv2
import pickle
import struct

# IP of your PiCar-X
pi_ip = '192.168.1.100'  # Replace with actual IP
port = 8000

# Set up client socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((pi_ip, port))

data = b""
payload_size = struct.calcsize("Q")

try:
    while True:
        # Receive message length
        while len(data) < payload_size:
            data += client_socket.recv(4096)
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        # Receive frame data
        while len(data) < msg_size:
            data += client_socket.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]

        # Deserialize and show frame
        frame = pickle.loads(frame_data)
        cv2.imshow("PiCar-X Stream", frame)

        if cv2.waitKey(1) == 27:
            break  # Press ESC to exit

except Exception as e:
    print(f"Error: {e}")

finally:
    client_socket.close()
    cv2.destroyAllWindows()
