from detector import recognize_faces
import streamlit as st
import cv2
import numpy as np

def main():
    st.title("Webcam Face Detection and Recognition")
    run = st.checkbox('Open Webcam')
    take_picture =st.button('Take Picture' , key="action")
    FRAME_WINDOW = st.image([])
    RESULT_WINDOW = st.image([])
    reset = st.button('Reset', key='reset')

    if run:
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            # print(ret)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, )
            # image_show = frame
            FRAME_WINDOW.image(rgb_frame)

            if take_picture:
                cap.release()
                cv2.destroyAllWindows()
                run = False
                # save_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite('gbrimage.jpg', frame)
                cv2.imwrite('rgbimage.jpg', rgb_frame)

        
        face_detected_image =  recognize_faces("gbrimage.jpg")
        
        # face_detected_image is pillow image so we need to convert it and show it as streamlit image
        face_detected_image = np.array(face_detected_image)
        RESULT_WINDOW.image(face_detected_image, caption='Detected Faces')
    # Add reset button
    if reset:
        run = True
        # remove detected faces image
        RESULT_WINDOW.image([])


if __name__ == '__main__':
    main()