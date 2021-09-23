import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(
    static_image_mode = True,
    max_num_hands = 2,
    min_detection_confidence = 0.5) as hands:
    # thu load 1 and test
    image = cv2.imread('hand.jpg')
    results =hands.process(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        print('Continue')
    
    #hien thi anh co ve diem landmark
    print('Hand landmark',results.multi_handedness)
    image_height,image_wwidth ,_ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
        print(
            f' Index finger :',
            f'{ hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*image_wwidth}'
            f'{ hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*image_height}'
        )
        mp_drawing.draw_landmarks(
            annotated_image,hand_landmarks,mp_hands.HAND_CONNECTIONS
        )
    cv2.imwrite(r'hands.png',annotated_image)

cap = cv2.VideoCapture(0)
preTime = 0 
with mp_hands.Hands(
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
) as hands:
    while cap.isOpened():
        success , image = cap.read()
        if not success:
            print('loi hien thi')
            continue

        image = cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS)

        currTime = time.time()
        fps = 1 /(currTime - preTime)
        preTime = currTime
        cv2.putText(image,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,196,255),2)
        cv2.imshow('Camera hien thi', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()