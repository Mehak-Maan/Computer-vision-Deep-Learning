from ultralytics import YOLO
import cv2
import pytesseract

# PREPROCESSING FUNCTION
def preprocess_plate(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )
    return thresh

# CLEANING OCR TEXT
def clean_text(text):
    t = text.upper().replace(" ", "")
    t = t.replace("O", "0")
    t = t.replace("I", "1")
    t = t.replace("|", "1")
    return t

# LOAD YOLO MODEL
model = YOLO("best.pt")   # Your trained YOLO11n model

# LOAD IMAGE
img = cv2.imread("Cars6.png")   # Replace this with your image path
results = model(img)[0]

# DETECT + OCR LOOP
for box in results.boxes.xyxy:
    x1, y1, x2, y2 = map(int, box)

    crop = img[y1:y2, x1:x2]
    processed = preprocess_plate(crop)

    raw = pytesseract.image_to_string(processed, config="--psm 7")
    cleaned = clean_text(raw)

    print("Raw OCR:", raw)
    print("Cleaned OCR:", cleaned)

    # Show for debug (comment out on headless Pi)
    cv2.imshow("Plate Crop", crop)
    cv2.imshow("Processed", processed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
