import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import math

print("Script başlatılıyor...")
print(f"MediaPipe version: {mp.__version__}")
print(f"OpenCV version: {cv2.__version__}")

# KURULUM:
#   pip install opencv-python mediapipe pyautogui numpy
# AYARLAR

CAM_INDEX = 0
FRAME_W, FRAME_H = 960, 540
MARGIN = 0.12

# İmleç stabilitesi
MAX_STEP_PX = 25
MOVE_DEADZONE_PX = 2

# One Euro Filter
ONEEURO_MIN_CUTOFF = 1.9
ONEEURO_BETA = 0.02
ONEEURO_D_CUTOFF = 1.0


THUMB_FOLD_ON = 115    
THUMB_FOLD_OFF = 135   

INDEX_EXTENDED_MIN = 155
# Tap / DoubleClick / Drag
DRAG_HOLD_T = 0.35
DRAG_MOVE_PX = 10
DOUBLECLICK_GAP = 0.35

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0
screen_w, screen_h = pyautogui.size()

print("MediaPipe başlatılıyor...")
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    mp_draw = mp.solutions.drawing_utils
    print("MediaPipe başarıyla yüklendi!")
except Exception as e:
    print(f"MediaPipe hatası: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
def clamp(x, a, b):
    return max(a, min(b, x))
def angle_deg(a, b, c):
   
    ax, ay = a
    bx, by = b
    cx, cy = c
    v1 = np.array([ax - bx, ay - by], dtype=np.float32)
    v2 = np.array([cx - bx, cy - by], dtype=np.float32)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0

    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))
# One Euro Filter
def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)
class LowPass:
    def __init__(self):
        self.inited = False
        self.y = 0.0
    def filt(self, x, a):
        if not self.inited:
            self.inited = True
            self.y = x
            return x
        self.y = a * x + (1.0 - a) * self.y
        return self.y

class OneEuro:
    def __init__(self, min_cutoff=1.9, beta=0.02, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_f = LowPass()
        self.dx_f = LowPass()
        self.last_t = None
        self.last_x = None
    def filt(self, x, t):
        if self.last_t is None:
            self.last_t = t
            self.last_x = x
            return self.x_f.filt(x, 1.0)
        dt = max(1e-6, t - self.last_t)
        dx = (x - self.last_x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = self.dx_f.filt(dx, a_d)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a_x = _alpha(cutoff, dt)
        x_hat = self.x_f.filt(x, a_x)
        self.last_t = t
        self.last_x = x
        return x_hat

fx = OneEuro(ONEEURO_MIN_CUTOFF, ONEEURO_BETA, ONEEURO_D_CUTOFF)
fy = OneEuro(ONEEURO_MIN_CUTOFF, ONEEURO_BETA, ONEEURO_D_CUTOFF)

# Durumlar

cur_x, cur_y = None, None

thumb_pending = False
thumb_start_t = 0.0
thumb_anchor_x, thumb_anchor_y = 0.0, 0.0
thumb_dragging = False
last_tap_time = 0.0

paused = False

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"HATA: Kamera {CAM_INDEX} açılamadı!")
    print("Başka bir kamera indeksini deneyin (örn: CAM_INDEX = 1)")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

print("Kamera başarıyla açıldı!")
print("Çıkış: 'q' | Pause/Resume: 'p'")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Kamera okunamadı.")
        break

    now = time.time()
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    # Aktif alan kutusu
    x1 = int(MARGIN * w); y1 = int(MARGIN * h)
    x2 = int((1 - MARGIN) * w); y2 = int((1 - MARGIN) * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
    status = [f"paused={paused} (p: toggle)"]
    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
        # Landmarklar
        # Thumb: 2(MCP),3(IP),4(TIP)
        # Index: 5(MCP),6(PIP),8(TIP)
        th_mcp = (lm.landmark[2].x, lm.landmark[2].y)
        th_ip  = (lm.landmark[3].x, lm.landmark[3].y)
        th_tip = (lm.landmark[4].x, lm.landmark[4].y)
        ix_mcp = (lm.landmark[5].x, lm.landmark[5].y)
        ix_pip = (lm.landmark[6].x, lm.landmark[6].y)
        ix_tip = (lm.landmark[8].x, lm.landmark[8].y)

        cursor_pt = (
            0.65 * ix_tip[0] + 0.35 * ix_pip[0],
            0.65 * ix_tip[1] + 0.35 * ix_pip[1]
        )
      
        nx = clamp(cursor_pt[0], MARGIN, 1 - MARGIN)
        ny = clamp(cursor_pt[1], MARGIN, 1 - MARGIN)
        nx = (nx - MARGIN) / (1 - 2 * MARGIN)
        ny = (ny - MARGIN) / (1 - 2 * MARGIN)

        target_x = nx * screen_w
        target_y = ny * screen_h

        # Filtre
        filt_x = fx.filt(target_x, now)
        filt_y = fy.filt(target_y, now)

        if cur_x is None:
            cur_x, cur_y = filt_x, filt_y

        dx = filt_x - cur_x
        dy = filt_y - cur_y
        step = math.hypot(dx, dy)

        if not paused:
            if step >= MOVE_DEADZONE_PX:
                if step > MAX_STEP_PX:
                    s = MAX_STEP_PX / step
                    dx *= s
                    dy *= s
                cur_x += dx
                cur_y += dy

            pyautogui.moveTo(int(cur_x), int(cur_y))

        
        thumb_angle = angle_deg(th_mcp, th_ip, th_tip)
        index_angle = angle_deg(ix_mcp, ix_pip, ix_tip)

        index_extended = index_angle >= INDEX_EXTENDED_MIN

      
        active_state = (thumb_pending or thumb_dragging)
        thumb_fold = thumb_angle < (THUMB_FOLD_OFF if active_state else THUMB_FOLD_ON)

        status.append(f"thumb_angle={thumb_angle:.0f}  index_angle={index_angle:.0f}")
        status.append(f"index_extended={index_extended}  thumb_fold={thumb_fold}")

        
        if not paused and index_extended:
            if thumb_fold and (not thumb_pending) and (not thumb_dragging):
                thumb_pending = True
                thumb_start_t = now
                thumb_anchor_x, thumb_anchor_y = cur_x, cur_y
                status.append("LEFT: thumb fold start")

            elif thumb_fold and thumb_pending and (not thumb_dragging):
                held = now - thumb_start_t
                moved = math.hypot(cur_x - thumb_anchor_x, cur_y - thumb_anchor_y)
                if held >= DRAG_HOLD_T or moved >= DRAG_MOVE_PX:
                    pyautogui.mouseDown(button="left")
                    thumb_dragging = True
                    status.append("LEFT: drag start")

            elif (not thumb_fold) and thumb_pending:
                if thumb_dragging:
                    pyautogui.mouseUp(button="left")
                    status.append("LEFT: drag end")
                else:
                    # tap / double tap
                    if (now - last_tap_time) <= DOUBLECLICK_GAP:
                        pyautogui.doubleClick()
                        status.append("LEFT: DOUBLE CLICK")
                        last_tap_time = 0.0
                    else:
                        pyautogui.click(button="left")
                        status.append("LEFT: CLICK")
                        last_tap_time = now

                thumb_pending = False
                thumb_dragging = False

            elif (not thumb_fold) and thumb_dragging:
                pyautogui.mouseUp(button="left")
                thumb_dragging = False
                thumb_pending = False
                status.append("LEFT: forced release")

       
        if not index_extended and (thumb_pending or thumb_dragging):
            if thumb_dragging:
                pyautogui.mouseUp(button="left")
            thumb_pending = False
            thumb_dragging = False
            status.append("LEFT: canceled (index not extended)")

    # Overlay
    y0 = 30
    for s in status[:10]:
        cv2.putText(frame, s, (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y0 += 26

    cv2.imshow("Hand Mouse (Win11) - Thumb Fold Click", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("p"):
        paused = not paused

cap.release()
cv2.destroyAllWindows()
hands.close()
