from flask import Flask, request, render_template
from ultralytics import YOLO
from geopy.geocoders import Nominatim
import os, uuid, cv2, math

app = Flask(__name__)

model = YOLO("best.pt")

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

geolocator = Nominatim(user_agent="pothole_guard")

@app.route("/")
def home():
    return render_template("index.html")

# ---------------- IMAGE (UPLOAD OR CAMERA) ----------------
@app.route("/detect_image", methods=["POST"])
def detect_image():
    image = request.files.get("image")
    lat = request.form.get("lat")
    lon = request.form.get("lon")

    if not image:
        return "No image received"

    name = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(UPLOAD_FOLDER, name)
    image.save(path)

    results = model(path, conf=0.25)
    boxes = results[0].boxes
    count = len(boxes) if boxes else 0

    annotated = results[0].plot()
    out_name = f"result_{name}"
    out_path = os.path.join(RESULT_FOLDER, out_name)
    cv2.imwrite(out_path, annotated)

    address = "Location not available"
    if lat and lon:
        try:
            address = geolocator.reverse(f"{lat},{lon}").address
        except:
            pass

    return render_template(
        "result.html",
        potholes=count,
        address=address,
        lat=lat,
        lon=lon,
        images=[out_name]
    )

# ---------------- VIDEO ----------------
@app.route("/detect_video", methods=["POST"])
def detect_video():
    video = request.files.get("video")
    lat = request.form.get("lat")
    lon = request.form.get("lon")

    if not video:
        return "No video received"

    name = f"{uuid.uuid4().hex}.mp4"
    path = os.path.join(UPLOAD_FOLDER, name)
    video.save(path)

    results = model(path, conf=0.25, stream=True)

    counts = []
    images = []

    for i, r in enumerate(results):
        if i % 15 != 0:
            continue

        if r.boxes:
            counts.append(len(r.boxes))

        frame = r.plot()
        img_name = f"{uuid.uuid4().hex}.jpg"
        cv2.imwrite(os.path.join(RESULT_FOLDER, img_name), frame)
        images.append(img_name)

        if len(images) == 3:
            break

    avg = sum(counts)/len(counts) if counts else 0
    potholes = math.ceil(avg * 5)

    address = "Location not available"
    if lat and lon:
        try:
            address = geolocator.reverse(f"{lat},{lon}").address
        except:
            pass

    return render_template(
        "result.html",
        potholes=potholes,
        address=address,
        lat=lat,
        lon=lon,
        images=images
    )

if __name__ == "__main__":
    app.run(debug=True, port=8000)