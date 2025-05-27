# Picat
This is a project for the Leiden University Robotics Class Spring 2025. Our team plans to turn a a PiCar-X into a cat-like entity that drives around a table and pushes items off.

PiCat: A Feline-Inspired Object-Toppling Robot

# Overview

PiCat is a playful robotics project developed as part of the Robotics 2025 course at LIACS, Universiteit Leiden. Inspired by cat behavior, PiCat uses computer vision and a servo-mounted paw to detect a bottle or cup, approach it, and knock it off the table. It demonstrates how simple perception-action loops can result in expressive, anthropomorphic robotic behavior.

# Demo Video

Demo Video can be found in the pdf also uploaded on Github

# Hardware Requirements

PiCar-X robot (by Sunfounder)

Raspberry Pi (with Raspberry Pi OS)

Pan-tilt camera module (RGB)

FS5106R continuous rotation servo (cat paw actuator)

Ultrasonic distance sensor (built-in)

Grayscale cliff sensors (built-in)

Robot HAT with audio output (for sound)

# Software & Dependencies

Install the following Python packages:

sudo apt update && sudo apt install python3-pip -y
pip3 install opencv-python numpy onnxruntime

# Additional libraries:

pip3 install robot-hat picar-x

PiCamera2 (optional, falls back to OpenCV if unavailable):

sudo apt install python3-picamera2

# How to Run

Place the following files in the same folder:

FinaleChallenge9002.py (main script)

yolo11n.onnx (YOLOv5 ONNX model)

meow_song_long.mp3 (background audio)

Run the script with elevated privileges for audio:

sudo python3 FinaleChallenge9002.py

# Robot Behavior (FSM)

PiCat runs on a finite state machine:

SWEEP: Pan and tilt camera to find a bottle or cup

TRACK: Center the object and align steering

APPROACH: Move toward object using ultrasonic sensing

WAIT: Look up and detect a human face

KNOCK: Swipe the object with a servo-actuated paw

BACKUP: Reverse slightly and return to sweep

# YOLOv5 Object Detection

Model: yolo11n.onnx (custom YOLOv5 export)

Classes used: bottle, cup

Input size: 480x480 (resized with letterboxing)

# Folder Structure

PiCat/
├── FinaleChallenge9002.py
├── yolo11n.onnx
├── meow_song_long.mp3 #or whatever song you want to have played, just remember to replace the respective file name in the code
├── README.md

# Additional Resources

# Cat paw model: [Thingiverse link](https://www.thingiverse.com/thing:3741160)

# Academic inspiration: Dutta et al., 2023 (IROS) - non-prehensile pushing for visuo-tactile inference

# Known Issues

Servo strength must be calibrated to avoid missing the object

Detection may fail under poor lighting

Limited to COCO-dataset classes (bottle, cup)

No obstacle avoidance implemented

# Authors

Nataliia Kaminskaia

Michael Olthof

Abdolrahim Tooranian

Amber van der Tuin

Robert C. Weber
