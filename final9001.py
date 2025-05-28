#!/usr/bin/env python3
"""
PiCar-X bottle-toppler — hunts bottles forever, plays background music,
and, during the 5-second WAIT phase, tilts the camera up to look for
standing people; as soon as a face is centred it knocks the bottle.
After the knock it backs up and resets the camera before resuming the hunt.
Now includes cliff detection to prevent falls.
"""

from __future__ import annotations
from pathlib import Path
from os import geteuid
import platform, sys, time, statistics, random
import cv2, numpy as np, onnxruntime as ort

from picarx import Picarx
from robot_hat import Servo, Music

# ─── MUSIC (unchanged) ────────────────────────────────────────────
MUSIC_DIR = (Path(__file__).parent).expanduser()
SONG_NAME = "meow_song_long.mp3"
SONG_VOL  = 100
def pick_song() -> Path:
    if SONG_NAME:
        p = (MUSIC_DIR / SONG_NAME).expanduser()
        if not p.exists(): sys.exit(f"❌ Song '{p}' not found")
        return p
    for ext in ("*.mp3", "*.wav", "*.flac"):
        files = list(MUSIC_DIR.glob(ext))
        if files: return files[0]
    sys.exit(f"❌ No audio files in {MUSIC_DIR}")

if geteuid() != 0:
    print("\033[0;33m⚠ Run with sudo or the speaker may be silent.\033[0m")

music = Music()
def start_music(song: Path):
    music.music_set_volume(SONG_VOL)
    music.music_play(str(song))

# ─── ARM-SERVO (unchanged) ───────────────────────────────────────
ARM_CH, NEUTRAL_US, GATE_US = 11, 1450, 60
OFFSET_US, T_90             = 200, 0.57
BRAKE_US, BRAKE_MS, SETTLE_MS = 70, 35, 120
def _pulse(off):
    if abs(off) <= GATE_US: off = 0
    arm.angle((NEUTRAL_US + off - 1500) / 11.11)
def knock():
    _pulse(-OFFSET_US); time.sleep(T_90)
    _pulse(BRAKE_US);   time.sleep(BRAKE_MS/1000)
    _pulse(0);          time.sleep(SETTLE_MS/1000)
    _pulse(+OFFSET_US); time.sleep(T_90)
    _pulse(-BRAKE_US);  time.sleep(BRAKE_MS/1000)
    _pulse(0);          time.sleep(SETTLE_MS/1000)

# ─── YOLO / NAVIGATION CONSTANTS (unchanged) ─────────────────────
MODEL_PATH = Path(__file__).with_name("yolo11n.onnx")
TARGET, CONF_THR = {"cup", "bottle"}, 0.35
COCO = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup"
]
FRAME_W = FRAME_H = IMG_SZ = 480
SCAN_SPEED, SCAN_DELAY      = 30, 0.8
PAN_GAIN,  TILT_GAIN        = 10/FRAME_W, 10/FRAME_H
PAN_LIM,   TILT_LIM         = 80, 80
STEER_SMOOTH, FORWARD_TIME, APPROACH_SPEED = 1, 0.2, 1
CENTER_FRAMES               = 3
TARGET_DIST                 = 100          # mm stop distance for ranger
BURST_TIME                  = 0.1
BACKUP_SPEED, BACKUP_TIME   = -1, 0.5
SWEEP_PANS  = [5, -5, 15, -15, 30, -30, 45, 60, 75, 90]
SWEEP_TILTS = [0, 10, 25]
SWEEP_DELAY  = SCAN_DELAY
CLOSE_IMG_RATIO = 0.55
FACE_PAN_GAIN  = 10 / FRAME_W
FACE_TILT_GAIN = 10 / FRAME_H
INITIAL_TILT_UP = 25
INITIAL_PAN_RIGHT = 50
FACE_PAN_OFFSET     = 10     # NEW – look slightly to the right
FACE_CENTER_THRESH = 15
CLIFF_BACKUP_SPEED = -1
CLIFF_BACKUP_TIME  = 1.5
CLIFF_REFERENCE    = [200, 200, 200]
RANGER_PRESENT_MAX = 300            # mm – consider “seen” if d < this
BACKUP_LOST_TIME   = 0.7           # s  – reverse when target lost

clamp = lambda v, lo, hi: max(lo, min(hi, v))

# ─── helper funcs (letterbox, camera, detect, ranger) ────────────
def letterbox(img):
    h,w = img.shape[:2]; scale = IMG_SZ/max(h,w)
    nh,nw=int(h*scale), int(w*scale)
    resized=cv2.resize(img,(nw,nh))
    pad_x,pad_y=IMG_SZ-nw, IMG_SZ-nh
    t,l=pad_y//2, pad_x//2
    boxed=np.full((IMG_SZ,IMG_SZ,3),114,np.uint8)
    boxed[t:t+nh, l:l+nw] = resized
    return boxed, scale, l, t

def get_camera():
    try:
        from picamera2 import Picamera2
        cam=Picamera2()
        cam.configure(cam.create_preview_configuration(
            main={"size":(FRAME_W,FRAME_H),"format":"RGB888"}))
        cam.start(); return cam, lambda: cam.capture_array()
    except Exception:
        backend=cv2.CAP_V4L2 if platform.system()=="Linux" else 0
        cap=cv2.VideoCapture(0,backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,FRAME_H)
        if not cap.isOpened(): sys.exit("No camera")
        return cap, lambda: cap.read()[1]

def detect_target(sess, inp, frame, wanted:set[str]):
    boxed,scale,px,py = letterbox(frame)
    blob=boxed.transpose(2,0,1)[None].astype(np.float32)/255
    det=sess.run(None,{inp:blob})[0][0]
    best=None; best_conf=0
    for d in det:
        if len(d)==7: _,x1,y1,x2,y2,conf,cls=d
        else: x1,y1,x2,y2,conf,cls=d
        if conf<CONF_THR or int(cls)>=len(COCO): continue
        name=COCO[int(cls)]
        if name not in wanted: continue
        if conf>best_conf:
            best_conf=conf
            best=(name,(x1-px)/scale,(y1-py)/scale,(x2-px)/scale,(y2-py)/scale,conf)
    return best

def median_distance(ranger,n=3):
    vals=[ranger.read() for _ in range(n)]
    vals=[v for v in vals if v>0]
    return statistics.median(vals)*10 if vals else None

def check_cliff(px):
    if px.get_cliff_status(px.get_grayscale_data()):
        print("⚠ Cliff detected! Backing up...")
        px.backward(abs(CLIFF_BACKUP_SPEED)); time.sleep(CLIFF_BACKUP_TIME); px.stop()
        px.set_dir_servo_angle(random.choice([-40, 40]))
        px.forward(APPROACH_SPEED); time.sleep(FORWARD_TIME*2); px.stop()
        px.set_dir_servo_angle(0)
        return True
    return False

# ╔══════════════════════════════════════════════════════════════╗
# ║                           MAIN                                ║
# ╚══════════════════════════════════════════════════════════════╝
def main():
    if not MODEL_PATH.exists(): sys.exit("❌ YOLO model not found")
    song = pick_song(); print(f"♫ Playing: {song.name}")

    sess  = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    inp   = sess.get_inputs()[0].name
    cam, grab = get_camera()
    global px; px = Picarx()
    ranger = px.ultrasonic
    global arm; arm = Servo("P11")
    px.set_cliff_reference(CLIFF_REFERENCE)

    start_music(song)

    while True:                             # endless hunt loop
        pan=tilt=dir_ang=0.0
        pan_idx=tilt_idx=0; centred_ctr=0
        state="SWEEP"

        while True:
            frame = grab()

            if check_cliff(px):
                state="SWEEP"; pan=tilt=dir_ang=0
                pan_idx=tilt_idx=centred_ctr=0
                continue

            # ── SWEEP ──────────────────────────────────────────
            if state=="SWEEP":
                pan_angle  = SWEEP_PANS[pan_idx]
                tilt_angle = SWEEP_TILTS[tilt_idx]
                px.set_cam_pan_angle(pan_angle); px.set_cam_tilt_angle(tilt_angle)
                time.sleep(SWEEP_DELAY)
                if detect_target(sess, inp, frame, {"bottle"}):
                    state="TRACK"; pan,tilt = pan_angle, tilt_angle
                else:
                    tilt_idx += 1
                    if tilt_idx >= len(SWEEP_TILTS):
                        tilt_idx = 0
                        pan_idx  = (pan_idx + 1) % len(SWEEP_PANS)

            # ── TRACK ───────────────────────────────────────────────────
            elif state == "TRACK":
                det = detect_target(sess, inp, frame, {"bottle"})
                if not det:                         # lost → sweep again
                    px.forward(0)
                    state   = "SWEEP"
                    pan_idx = tilt_idx = 0
                    continue

                # camera pan/tilt
                _, x1, y1, x2, y2, _ = det
                cx, cy  = (x1 + x2) / 2, (y1 + y2) / 2
                pan_err = cx - FRAME_W / 2
                tilt_err= cy - FRAME_H / 2
                pan  = clamp(pan  + pan_err * PAN_GAIN,  -PAN_LIM,  PAN_LIM)
                tilt = clamp(tilt - tilt_err * TILT_GAIN, -TILT_LIM, TILT_LIM)
                px.set_cam_pan_angle(pan)
                px.set_cam_tilt_angle(tilt)

                # wheels track camera
                dir_ang = clamp(pan, -PAN_LIM, PAN_LIM)
                px.set_dir_servo_angle(dir_ang)
                lost_cnt = 0                       # NEW counter for approach
                state    = "APPROACH"
                continue

            # ── APPROACH ────────────────────────────────────────────────
            elif state == "APPROACH":
                d   = median_distance(ranger)      # mm or None
                det = detect_target(sess, inp, frame, {"bottle"})

                # steer (if vision available) ---------------------------
                # ── camera & wheel steering while vision is available ──────────
                if det:
                    _, x1, y1, x2, y2, _ = det
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    pan_err  = cx - FRAME_W / 2
                    tilt_err = cy - FRAME_H / 2

                    # update camera pan/tilt to keep bottle centred
                    pan  = clamp(pan  + pan_err  * PAN_GAIN,  -PAN_LIM,  PAN_LIM)
                    tilt = clamp(tilt - tilt_err * TILT_GAIN, -TILT_LIM, TILT_LIM)
                    px.set_cam_pan_angle(pan)
                    px.set_cam_tilt_angle(tilt)

                    # steer wheels in the **same direction** as camera pan
                    dir_ang = clamp(pan, -PAN_LIM, PAN_LIM)
                    px.set_dir_servo_angle(dir_ang)
                else:
                    # no vision → keep wheels straight
                    px.set_dir_servo_angle(0)

                # too close → small back-off ---------------------------
                if d and d < TARGET_DIST - 30:
                    px.backward(0.5); time.sleep(0.3); px.forward(0)
                    lost_cnt = 0
                    continue# ── camera & wheel steering while vision is available ──────────

                if det:
                    _, x1, y1, x2, y2, _ = det
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    pan_err  = cx - FRAME_W / 2
                    tilt_err = cy - FRAME_H / 2

                    # update camera pan/tilt to keep bottle centred
                    pan  = clamp(pan  + pan_err  * PAN_GAIN,  -PAN_LIM,  PAN_LIM)
                    tilt = clamp(tilt - tilt_err * TILT_GAIN, -TILT_LIM, TILT_LIM)
                    px.set_cam_pan_angle(pan)
                    px.set_cam_tilt_angle(tilt)

                    # steer wheels in the **same direction** as camera pan
                    dir_ang = clamp(pan, -PAN_LIM, PAN_LIM)
                    px.set_dir_servo_angle(dir_ang)
                else:
                    # no vision → keep wheels straight
                    px.set_dir_servo_angle(0)
                # reached stopping distance ---------------------------
                if d and d <= TARGET_DIST:
                    px.backward(0.5); time.sleep(0.1);
                    px.forward(0)
                    state      = "WAIT"
                    wait_start = time.time()
                    px.set_cam_tilt_angle(INITIAL_TILT_UP)
                    px.set_cam_pan_angle(INITIAL_PAN_RIGHT)
                    continue

                # gentle forward burst with live ranger check ----------
                px.forward(0.5)
                t0 = time.time()
                while time.time() - t0 < BURST_TIME:
                    d_live = median_distance(ranger)
                    check_cliff(px)
                    if d_live and d_live <= TARGET_DIST:
                        d = d_live
                        break
                    time.sleep(0.05)
                px.forward(0)

                # update lost counter ----------------------------------
                if det or (d and d < RANGER_PRESENT_MAX):
                    lost_cnt = 0            # bottle still somewhere ahead
                else:
                    lost_cnt += 1

                # give up only after 3 consecutive empty frames --------
                if lost_cnt >= 2:
                    print("Target lost – backing up before new sweep")
                    px.backward(0.7); time.sleep(BACKUP_LOST_TIME); px.forward(0)
                    state   = "SWEEP"
                    pan_idx = tilt_idx = 0
                    px.set_cam_pan_angle(0); px.set_cam_tilt_angle(0)

            # ── WAIT / FACE / KNOCK / BACKUP (unchanged) ─────────
            elif state=="WAIT":
                elapsed=time.time()-wait_start
                face=detect_target(sess, inp, frame, {"person"})
                if face:
                    _,x1,y1,x2,y2,_=face
                    cx,cy=(x1+x2)/2,(y1+y2)/2
                    pan_err=cx-FRAME_W/2; tilt_err=cy-FRAME_H/2
                    pan  = clamp(pan + pan_err*FACE_PAN_GAIN,-PAN_LIM,PAN_LIM)
                    tilt = clamp(tilt-tilt_err*FACE_TILT_GAIN,-TILT_LIM,TILT_LIM)
                    px.set_cam_pan_angle(pan); px.set_cam_tilt_angle(tilt)
                    if abs(pan_err)<FACE_CENTER_THRESH and abs(tilt_err)<FACE_CENTER_THRESH:
                        knock(); state="BACKUP"; backup_start=time.time()
                        px.backward(abs(BACKUP_SPEED))
                        px.set_cam_pan_angle(0); px.set_cam_tilt_angle(0); continue
                if elapsed>=5:
                    knock(); state="BACKUP"; backup_start=time.time()
                    px.backward(abs(BACKUP_SPEED))
                    px.set_cam_pan_angle(0); px.set_cam_tilt_angle(0)

            elif state=="BACKUP":
                if time.time()-backup_start >= BACKUP_TIME:
                    px.forward(0)
                    print("Bottle knocked!  Searching for next bottle …")
                    break

            if cv2.waitKey(1)&0xFF==ord('q'):
                px.forward(0); music.music_stop(); cv2.destroyAllWindows(); return

            cv2.imshow("View", frame)

if __name__ == "__main__":
    main()