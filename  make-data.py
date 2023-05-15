import cv2
import mediapipe as mp
import pandas as pd

# Read images from webcam
cap = cv2.VideoCapture(0)

# initialization Mediapipe library

mpPose = mp.solutions.pose
pose = mpPose.Pose()

mpDraw = mp.solutions.drawing_utils

lm_list = []
label = "TwoHands"
nb_of_frame = 600

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


while len(lm_list) <= nb_of_frame:
    ret, frame = cap.read()
    if ret:
        # Recognize Pose
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)

        if results.pose_landmarks:
            # record skeleton parameters
            lm = make_landmark_timestep(results)
            lm_list.append(lm)

            # draw skeleton in Image
            frame = draw_landmark_on_image(mpDraw, results, frame)

        cv2.imshow("image", frame)
        if cv2.waitKey(1) == ord('q'):
            break
# write data to csv file
df = pd.DataFrame(lm_list)
df.to_csv(label + ".txt")
cap.release()
cv2.destroyAllWindows()