import cv2
import os
import base64
import threading
import numpy as np
from time import time
from datetime import datetime
from shapely.geometry import LineString
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors

# Google API Key
GOOGLE_API_KEY = "AIzaSyCOvtWthmSJlH1Tlq9_IPCwhB24rHsYBQI"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

class SpeedEstimator(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_region()
        self.spd = {}
        self.trkd_ids = []
        self.trk_pt = {}
        self.trk_pp = {}
        self.saved_ids = set()
        self.gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        os.makedirs("crop", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

    def analyze_and_log_response(self, image_path, track_id, speed, timestamp):
        try:
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Extract ONLY the vehicle number plate text from the image. Respond only with the plate text."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        "description": "Image of a vehicle"
                    }
                ]
            )

            response = self.gemini_model.invoke([message])
            number_plate = response.content.strip()

            # Save to text file
            log_file = f"logs/vehicle_{track_id}.txt"
            with open(log_file, "w") as f:
                f.write(f"Track ID: {track_id}\n")
                f.write(f"Speed: {speed} km/h\n")
                f.write(f"Time: {timestamp}\n")
                f.write(f"Number Plate: {number_plate}\n")

            print(f"📝 Number plate saved to {log_file}")

        except Exception as e:
            print(f"❌ Gemini AI Error: {e}")

    def estimate_speed(self, im0):
        self.annotator = Annotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)

        # Draw region line manually
        if len(self.region) >= 2:
            cv2.line(im0, self.region[0], self.region[1], (104, 0, 123), self.line_width * 2)

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.store_tracking_history(track_id, box)

            if track_id not in self.trk_pt:
                self.trk_pt[track_id] = time()
            if track_id not in self.trk_pp:
                self.trk_pp[track_id] = box

            prev_pos = self.trk_pp[track_id]
            curr_pos = box

            if LineString([prev_pos[:2], curr_pos[:2]]).intersects(LineString(self.region)):
                direction = "known"
            else:
                direction = "unknown"

            if direction == "known" and track_id not in self.trkd_ids:
                self.trkd_ids.append(track_id)
                time_diff = time() - self.trk_pt[track_id]
                if time_diff > 0:
                    speed = np.linalg.norm(np.array(curr_pos[:2]) - np.array(prev_pos[:2])) / time_diff
                    self.spd[track_id] = round(speed)

            self.trk_pt[track_id] = time()
            self.trk_pp[track_id] = curr_pos

            speed_value = self.spd.get(track_id, 0)
            label = f"ID: {track_id} {speed_value} km/h"
            self.annotator.box_label(box, label=label, color=colors(track_id, True))

            if track_id in self.spd and track_id not in self.saved_ids:
                x1, y1, x2, y2 = map(int, box)

                # Add padding to crop for better number plate visibility
                pad = 20
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(im0.shape[1], x2 + pad), min(im0.shape[0], y2 + pad)

                cropped_image = im0[y1:y2, x1:x2]
                if cropped_image.size != 0:
                    filename = f"crop/{track_id}_{speed_value}kmh.jpg"
                    cv2.imwrite(filename, cropped_image)
                    print(f"📷 Image saved: {filename}")

                    threading.Thread(
                        target=self.analyze_and_log_response,
                        args=(filename, track_id, speed_value, current_time),
                        daemon=True
                    ).start()

                    self.saved_ids.add(track_id)

        self.display_output(im0)
        return im0

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse: x={x}, y={y}")

# Start video capture
cap = cv2.VideoCapture('vid.mp4')  # Replace with 0 for webcam
region_points = [(0, 276), (1020, 276)]  # Set line for speed detection

speed_obj = SpeedEstimator(region=region_points, model="best.pt", line_width=2)

cv2.namedWindow("Speed Estimation")
cv2.setMouseCallback("Speed Estimation", mouse_callback)
frame_count=0
while True:
    ret, frame = cap.read()
    frame_count += 1
    if frame_count % 2 != 0:
        continue
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    result = speed_obj.estimate_speed(frame)
    cv2.imshow("Speed Estimation", result)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
