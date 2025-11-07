import numpy as np
import cv2

# Importing other modules
import Triangulate as TrigL
import PoseGraph as PGL
import Display as DispL
import Match as MatchL
import Descriptor as DescL

F: int = 500
W, H = 1920 // 2, 1080 // 2
K: np.ndarray = np.array([[F, 0, W // 2],
                          [0, F, H // 2],
                          [0, 0, 1]])

desc_dict, inp = None, None

if __name__ == '__main__':
    inp = input("Enter video source: ")
    desc_dict = DescL.Descriptor(W, H, K)
    desc_dict.create_viewer()
    disp = DispL.Display(W, H)


def img_resize(image: np.ndarray):
    image: np.ndarray = cv2.resize(image, (W, H))
    return image


# This method unifies all other modules (.py files) into 1 algorithm
def generate_SLAM(image: np.ndarray):
    image = img_resize(image)

    frame = PGL.Camera(desc_dict, image, K)
    if frame.id == 0:
        return
    frame1 = desc_dict.frames[-1]
    frame2 = desc_dict.frames[-2]

    # Find matches between 2 frames
    x1, x2, id = MatchL.generate_match(frame1, frame2)
    frame1.pose = np.dot(id, frame2.pose)
    for i, ind in enumerate(x2):
        if frame2.pts[ind] is not None:
            frame2.pts[ind].add_observation(frame1, x1[i])

    # Homogeneous 3D coords
    pts_4d = TrigL.triangulate(frame1.pose, frame2.pose, frame1.key_pts[x1], frame2.key_pts[x2])
    safe_div = np.where(pts_4d == 0, 1e-12, pts_4d) # Handle div-by-zero
    pts_4d /= safe_div[:, 3:]

    unmatched_points = np.array([frame1.pts[i] is None for i in x1])
    print(f"Adding: {np.sum(unmatched_points)} points")
    good_pts_4d = (np.abs(pts_4d[:, 3]) > 0.005) & (pts_4d[:, 2] > 0) & unmatched_points

    # Add keypoint
    for i, p in enumerate(pts_4d):
        if not good_pts_4d[i]:
            continue

        pt = DescL.Point(desc_dict, p)
        pt.add_observation(frame1, x1[i])
        pt.add_observation(frame2, x2[i])

    if len(desc_dict.frames) % 5 == 0:
        print("G2O optimisation")
        desc_dict.optimise()

    # Mark keypoint for visualisation
    for pt1, pt2 in zip(frame1.key_pts[x1], frame2.key_pts[x2]):
        u1, v1 = PGL.denormalise(K, pt1)
        u2, v2 = PGL.denormalise(K, pt2)
        cv2.circle(image, (u1, v1), color=(0,255,0), radius=1)
        cv2.line(image, (u1, v1), (u2, v2), color=(255, 255,0))

    # 2D display
    if disp is not None:
        disp.display2D(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # 3D display
    if desc_dict is not None:
        desc_dict.display()


# Main
if __name__ == "__main__":
    # Get video input
    cap = cv2.VideoCapture(inp) # Can try real-time but camera equipment and high performance compute required
    print("Start capture")

    print("Start SLAM")
    while cap.isOpened():
        # Get next frame
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Video ended.")
            break
        print("Next frame")
        
        # Input display
        frame1 = cv2.resize(frame, (720, 400)) # Resizing the original window
        cv2.imshow("Frame",frame1)    
        if cv2.waitKey(1) & 0xFF == ord('q'): # Quit Condition
            break

        # Run SLAM program on current frame
        generate_SLAM(frame)

    cv2.waitKey()
    # Release resources and close all windows
    cap.release() 
    cv2.destroyAllWindows()
    disp.clear()