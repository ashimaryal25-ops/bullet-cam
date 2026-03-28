import cv2
import mediapipe as mp
import numpy as np

class BulletMirror:
    def __init__(self, filename):
        # setting up the face tracker
        mp_face = mp.solutions.face_mesh
        self.face_mesh = mp_face.FaceMesh(max_num_faces=2)
        self.sz = 180

        # load the mascot png
        img = cv2.imread(filename)
        if img is None:
            print("couldn't find image")
            exit()

        self.mascot = self.remove_white_bg(img)

    def remove_white_bg(self, img):
        #  4th alpha channel for transparency 
        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        
        # split into individual color grids
        b = bgra[:, :, 0]
        g = bgra[:, :, 1]
        r = bgra[:, :, 2]
        
        # white pixels have high b, g, and r values
        white_pixels = (b > 180) & (g > 180) & (r > 180)
        
        # make white pixels transparent, keep everything else
        bgra[:, :, 3] = np.where(white_pixels, 0, 255)
        
        return bgra

    def overlay_on_head(self, frame, face):
        h = frame.shape[0]
        w = frame.shape[1]

        # landmark 10 is the forehead point
        hx = int(face.landmark[10].x * w)
        hy = int(face.landmark[10].y * h)

        # figure out where to put the mascot
        x_start = hx - (self.sz // 2)
        y_start = hy - 160

        # only draw if the mascot is fully inside the screen
        # otherwise it crashes trying to draw outside the frame
        x_end = x_start + self.sz
        y_end = y_start + self.sz
        if x_start > 0 and y_start > 0 and x_end < w and y_end < h:
            
            resized = cv2.resize(self.mascot, (self.sz, self.sz))
            
            # alpha crhannel tells us how visible each pixel is (0 to 1)
            alpha = resized[:, :, 3] / 255.0
            
            # blend mascot onto frame for each color channel
            for c in range(3):
                mascot_pixels = resized[:, :, c]
                frame_pixels = frame[y_start:y_end, x_start:x_end, c]
                
                # if alpha is 1 show mascot, if 0 show webcam.
                frame[y_start:y_end, x_start:x_end, c] = (alpha * mascot_pixels + (1 - alpha) * frame_pixels)

        return frame, hx, hy

    def run(self):
        # open the webcam
        cap = cv2.VideoCapture(0)
        print("press q to quit")

        try:
            while cap.isOpened():
                # grab a frame from webcam
                ret, frame = cap.read()
                if not ret:
                    break

                # flipping so it acts like a mirror
                frame = cv2.flip(frame, 1)

                # mediapipe takes rgb but opencv gives bgr so we convert
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    for face in results.multi_face_landmarks:
                        
                        # put mascot on head
                        frame, hx, hy = self.overlay_on_head(frame, face)

                        # check if mouth is open by comparing lip landmarks
                        top_lip = face.landmark[13].y
                        bottom_lip = face.landmark[14].y
                        dist = abs(top_lip - bottom_lip)
                        
                        if dist > 0.05:
                            cv2.putText(frame, "GO BULLETS!", (hx + 80, hy - 60),
                                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 3)

                cv2.imshow('Bullet Mirror', frame)

                # quit if q pressed or window x button clicked
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or cv2.getWindowProperty('Bullet Mirror', cv2.WND_PROP_VISIBLE) < 1:
                    break

        finally:
            # release camera when done
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = BulletMirror('bullet.png')
    app.run()