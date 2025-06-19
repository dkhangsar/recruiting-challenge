from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import cv2, numpy as np, os
from typing import Optional, Dict

AGE_PROTO    = "models/age_deploy.prototxt"
AGE_MODEL    = "models/age_net.caffemodel"
GENDER_PROTO = "models/gender_deploy.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"

AGE_BUCKETS  = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST  = ['Male', 'Female']

# these two lines do the actual load once
age_net    = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)


app = FastAPI()

class Profile(BaseModel):
    description: str
    age: Optional[str] = None
    gender: Optional[str] = None
    skin_tone: Optional[str] = None
    hair_color: Optional[str] = None

class VerificationResult(BaseModel):
    match: bool
    mismatches: Dict[str, tuple]

# Creates the profile by grabbing the file (requesting it), and loads it to be able to processed by the analyze_face function.
# Profile_text ends up being a string response that analyzes specific aspects of the face that is analyzed (if a face is found).

@app.post("/create-profile", response_model=Profile)
async def create_profile(file: UploadFile = File(...)):
    img = Image.open(BytesIO(await file.read()))
    desc, age, gender, skin_tone, hair_color = analyze_face(img)
    return Profile(description=desc, age=age, gender=gender, skin_tone=skin_tone, hair_color=hair_color)

@app.post("/verify-profile", response_model=VerificationResult)
async def verify_profile(file: UploadFile = File(...), profile: Profile = ...):
    img = Image.open(BytesIO(await file.read()))
    desc, age, gender, skin_tone, hair_color = analyze_face(img)
    mismatches = {}
    if profile.age != age:
        mismatches['age'] = (profile.age, age)
    if profile.gender != gender:
        mismatches['gender'] = (profile.gender, gender)
    if profile.skin_tone != skin_tone:
        mismatches['skin_tone'] = (profile.skin_tone, skin_tone)
    if profile.hair_color != hair_color:
        mismatches['hair_color'] = (profile.hair_color, hair_color)
    match = len(mismatches) == 0
    return {"match": match, "mismatches": mismatches}

try:
    import cv2.data
    haarcascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
except (ImportError, AttributeError):
    # Fallback for older OpenCV versions
    haarcascade_path = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml')

_face_cascade = cv2.CascadeClassifier(haarcascade_path)

def rgb_to_simple_color(rgb):
    # Map average RGB to a simple color name (very basic)
    r, g, b = rgb
    if r < 60 and g < 60 and b < 60:
        return "black"
    if r > 200 and g > 200 and b > 200:
        return "white"
    if r > 150 and g > 100 and b < 80:
        return "blonde"
    if r > 100 and g < 80 and b < 80:
        return "red"
    if r > 80 and g > 60 and b > 40:
        return "brown"
    if r < 120 and g < 120 and b < 120:
        return "dark brown"
    if b > r and b > g:
        return "gray"
    return f"rgb({r},{g},{b})"

# This function expects a PIL Image
def analyze_face(image: Image.Image):
    MAX_DIM = 600
    if max(image.size) > MAX_DIM:
        scale = MAX_DIM / max(image.size)
        image = image.resize((int(image.size[0]*scale), int(image.size[1]*scale)))
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(gray, 1.3, 5)
    if not len(faces):
        return "No face detected.", None, None, None, None
    x, y, w, h = faces[0]
    face_img = img_np[y:y+h, x:x+w]
    # Convert face_img from RGB to BGR for DNN
    face_img_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
    # Skin tone: sample center 40x40 region of face
    cx, cy = w//2, h//2
    skin_patch = face_img[max(0,cy-20):min(h,cy+20), max(0,cx-20):min(w,cx+20)]
    skin_rgb = np.mean(skin_patch.reshape(-1, 3), axis=0).astype(int)
    skin_tone = rgb_to_simple_color(skin_rgb)
    # Hair color: sample 40x20 region just above the face
    hair_y1 = max(0, y-30)
    hair_y2 = y
    hair_x1 = x + w//4
    hair_x2 = x + 3*w//4
    hair_patch = img_np[hair_y1:hair_y2, hair_x1:hair_x2]
    if hair_patch.size > 0:
        hair_rgb = np.mean(hair_patch.reshape(-1, 3), axis=0).astype(int)
        hair_color = rgb_to_simple_color(hair_rgb)
    else:
        hair_color = None
    # Age/Gender
    blob = cv2.dnn.blobFromImage(face_img_bgr, 1.0, (227, 227), (78.42633776, 87.76891437, 114.89584775), swapRB=False)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_BUCKETS[age_preds[0].argmax()]
    desc = f"Face @({x},{y}), size={w}×{h}; Gender: {gender}; Age: {age}; Skin tone: {skin_tone}; Hair color: {hair_color}."
    return desc, age, gender, skin_tone, hair_color