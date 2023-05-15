import cv2
import mediapipe as mp
import keras.models
import numpy as np
import threading

# Read images from webcam
cap = cv2.VideoCapture(0)

# initialization Mediapipe library

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Load Model
model = keras.models.load_model("HumanModel1.h5")


lm_list = []


def make_landmark_timestep(results):
    print(results.pose_landmarks.landmark)
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    # draw pose connections
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    # draw connection points
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id,lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (0,0,255), cv2.FILLED)
    return img

def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img,label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis= 0)
    #print(lm_list.shape)
    results = model.predict(lm_list)
    print(results)
    if results[0][0] > 0.5:
        label = "One Hand"
    else:
        label = "Two hand"
    return label

i = 0
warmup_frames = 20
label = "..."

while True:
    success, img = cap.read()
    # Recognize Pose
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    i = i + 1
    if i > warmup_frames:
        print("Start detect ...")

        if results.pose_landmarks:
            # record skeleton parameters
            c_lm = make_landmark_timestep(results)
            lm_list.append(c_lm)
            # list > 10 frames -> detect
            if len(lm_list) == 10:
                # Detect
                thread_01 = threading.Thread(target=detect,args=(model, lm_list,))
                thread_01.start()
                lm_list = []

                # Show result on image

            # draw skeleton in Image
            img = draw_landmark_on_image(mpDraw, results, img)

        img = draw_class_on_image(label, img)
        cv2.imshow("image", img)
        if cv2.waitKey(1) == ord('q'):
            break
# # write data to csv file
# df = pd.DataFrame(lm_list)
# df.to_csv(label + ".txt")
cap.release()
cv2.destroyAllWindows()