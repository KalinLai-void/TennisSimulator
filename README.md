Last Demo Video: [https://drive.google.com/file/d/1IS7XXpkDAJFSkqCL16LljOe32Ng_hmkz/view?usp=sharing](https://drive.google.com/file/d/1IS7XXpkDAJFSkqCL16LljOe32Ng_hmkz/view?usp=sharing)
# TennisSimulator
- ## Intro
  This project simulates tennis ball trajectories in Unity using the coordinates predicted by [TrackNetV3](https://github.com/alenzenx/TrackNetV3/tree/main) and video frames from both overhead and side views. The system converts pixel coordinates to real‑world meters, then streams 3‑D positions to Unity via UDP for trajectory playback.  
### 相機視角轉換(Camera perspective conversion) 
![123](/Camera-perspective-conversion.png)
---

## 前置準備 (Pre‑Requirements)
- ### TrackNetV3
  Follow the steps in this link to download [TracketV3](https://github.com/alenzenx/TrackNetV3/tree/main)
- ### 開發環境 (Environment and requirements)
  | demand | directions |
  | --- | --- |
  | Anaconda 3 | [Download](https://www.anaconda.com/) |
  | Python | version 3.8.18 |
  | Unity | version 2021.3.18f1 |
  | Python Package(lib) | [requirements.txt](/requirements.txt) |
---
## 下載 (Installation)
1. git clone/download this repository. 
2. download the models and lib files from [requirements.txt](/requirements.txt), and unzip to this repository's root. 
3. create your python env (Anaconda and Python 3.8.18). 
4. activate your env, and pip install -r requirement.txt.
5. Download [Unity](https://unity.com/download)
6. choose the select New Project in Uinty's project and select the correct version

---
## Program Function Overview
  > The following summarizes each **Python** script’s functions and workflows, with a minimal `python` code snippet that showcases its **key use‑case**.  
- **Main.py:**  
  - **set path**  
    cap is the video path  
    csv is the coordinates of the tennis ball in the film  
    ```python
    cap1=cv2.VideoCapture(r"D:/tennis_MovieData/Datasets/Datasets/Outdoor Field/Cross-court Shot/Side-View/TrackNet/OCS17_pred.mp4")
    cap2=cv2.VideoCapture(r"D:/tennis_MovieData/Datasets/Datasets/Outdoor Field/Cross-court Shot/Top-View/TrackNet/OCT17_pred.mp4")
    csv_cap1=(r"D:/tennis_MovieData/Datasets/Datasets/Outdoor Field/Cross-court Shot/Side-View/TrackNet/OCS17_ball.csv")
    csv_cap2=(r"D:/tennis_MovieData/Datasets/Datasets/Outdoor Field/Cross-court Shot/Top-View/TrackNet/OCT17_ball.csv")
    
    ```
    ---
  - **Set the corner of the court to view the camera**  
    ```python
    pX1, pY1 = 507, 296 # left-top corner
    pX2, pY2 = 1315, 283 # right-top corner
    pX3, pY3 = 112, 909 # left-bottom corner
    pX4, pY4 = 1717, 872 # right-bottom corner
    ```
  - **Start the tennis simulation**  
    1. Open Unity and press the strat_button  
    ![GitHub图像](/unity_screen.png)  
    2. Open Main.py and execute
    3. Watch the result on Unity
  
  
  ---
- **LoadWebcam.py — Webcam‑capture thread**    
  - **Purpose:** Wraps `cv2.VideoCapture` in a `threading.Thread`, reads frames asynchronously, and pushes them to a fixed‑length `queue`.  
  ```python
  import threading
  import queue
  import cv2
  class LoadWebcam(threading.Thread):
      def __init__(self, source, name=None, skip=0, buffer_size=1):
          print(f"LoadWebcam: {source} ({name})")
          threading.Thread.__init__(self)
          self._skip = skip
          self.name = name
          self._buffer_size = buffer_size
          self._img0s = queue.Queue(maxsize=self._buffer_size)
          self._source = source
          self._cap = cv2.VideoCapture(self._source)
          fourcc = cv2.VideoWriter.fourcc('M', 'J', 'P', 'G')
          self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)
          self._cap.set(cv2.CAP_PROP_FPS, 60)
          self.isLooping = True
          self.start()
  
  ```
  ---
- **DetectTennis‑kmeans‑unity.py — Red‑clay demo video detection**    
  - **Purpose:** Uses KNN background subtraction & HSV segmentation to locate the ball center, computes its trajectory, sends data to Unity, and plots landing points with Matplotlib.  
  ```python
  ret, frame1 = cap1.read()
  ret, frame2 = cap2.read()
  rows,cols = frame2.shape[:2]
  pts_o = np.float32([[549,414],[1266,398],[306,907],[1515,873]])
  pts_d = np.float32([[0,0],[766,0],[0,826],[766,826]])
  M = cv2.getPerspectiveTransform(pts_o, pts_d)
  dst = cv2.warpPerspective(frame2, M, (766, 826))
  q = isinstance(left_find_ball(frame1), list)
  if q == False:
      side_center = left_find_ball(frame1)
      air_center  = right_find_ball(dst)
      if len(side_center) >= 1:
          side_tennis_x = side_center[0][1]
          side_tennis_y = side_center[0][0]
      if len(air_center) >= 1:
          air_tennis_x = air_center[0][1]
          air_tennis_y = air_center[0][0]
      if len(side_center) >=1 and len(air_center) >=1 and side_tennis_x > side1[0]:
          pts_0.append([side_tennis_x,side_tennis_y])
          pts_1.append([air_tennis_x,air_tennis_y])
          Ypoint.append(side_tennis_y)
  
  ```
  ---
- **DetectTennis‑unity‑indoor.py — Indoor hard‑court live detection**    
  - **Purpose:** Indoor‑specific HSV & scales, computes trajectory and writes to output.csv.  
  ```python
  for cnt1 in contours1:
      area1 = cv2.contourArea(cnt1)
      if area1 > 20:
          x1, y1, w1, h1 = cv2.boundingRect(cnt1)
          tennis_x1, tennis_y1 = x1 + w1/2 ,y1 + h1/2
          if tennis_x1 < 1600:  
              start_processing = True
          if start_processing:
              frame1 = cv2.circle(frame1,(int(tennis_x1),int(tennis_y1)),10,(0,0,225),-1)
              pts_0.append([tennis_x1,tennis_y1,frame_id])
              if tennis_x1 > 777:
                  Ypoint.append(tennis_y1)
              if len(pts_0) > 2 and (pts_0[0][2]==pts_0[1][2] or pts_0[1][2]==pts_0[2][2]):
                  min_x = min(pts_0, key=lambda x: x[0])
                  pts_0.remove(min_x)

  ```
   ---
- **DetectTennis‑unity‑redclay.py — Outdoor red‑clay live detection**    
  - **Purpose:** Optimized HSV for red‑clay; computes initial speed as “top‑view displacement ÷ FPS.”    
  ```python
  if len(pts_0) > 1 and len(pts_1) > 1 and pts_0[0][0] > pts_0[1][0] and pts_1[0][1] > pts_1[1][1]:
    side_start_point = pts_0[1]
    side_end_point   = pts_0[0]
    air_start_point  = pts_1[0]
    air_end_point    = pts_1[1]
    side_y = side_end_point[1] - side_start_point[1]
    side_x = side_end_point[0] - side_start_point[0]
    air_y  = air_end_point[1] - air_start_point[1]
    air_x  = air_end_point[0] - air_start_point[0]
    ymin   = max(Ypoint)
    startY = ((ymin - side_end_point[1])) + 0.05
    startY = startY * side_scales
    startX = ((pts_1[0][1]) * (-Xscales)) - 6.3
    if pts_1[1][0] >= 385:
        startZ = pts_1[1][0] * (-Zscales) + 5.47
    elif pts_1[1][0] < 385:
        startZ = pts_1[1][0] * Zscales - 5.47
    air_distance = math.dist(pts_1[1], pts_1[0])
    side_angle = 90 - degrees(atan2(side_x, side_y))
    air_angel  = degrees(atan2(air_x, air_y))

  ```
  ---
- **VirtualTennis_Real‑time.py — Half‑court scaled‑down field**    
  - **Purpose:** Two‑camera setup for a 3.32 × 3 m half‑court, computes angle/speed, and streams to Unity port 65522.  
  ```python
  while(True):
    try:
        Run(cap1, VIEW_NAMES[0], knn1)
        Run(cap2, VIEW_NAMES[1], knn2)
        if not LABELING_POINT_MODE:
            params = calculatePhysics()
            if len(params) > 0:
                print(params)
                unitySocket.sendto(str.encode(str(params)), unityAddressPort)
                sidePointsAndTimestamps.clear()
                overheadPointsAndTimestamps.clear()
                sideYpoints.clear()
        keycode = cv2.waitKey(1)
        if keycode == ord("q"): break
        if keycode == ord("a"): # testing
            unitySocket.sendto(str.encode(str([9,-13.59,12,1.36,0.3,2.51])), unityAddressPort)
    except Exception as error:
        print("An exception occurred:", error)
        break
  
  ```
  ---
- **Tennis_Real‑time.py — Full‑size court live detection**    
  - **Purpose:** Targets a 10.97 × 9.91 m half‑court with full perspective correction, streaming to Unity port 65520.  
  ```python
  while(True):
    try:
        Run(cap1, VIEW_NAMES[0], knn1)
        Run(cap2, VIEW_NAMES[1], knn2)
        if not LABELING_POINT_MODE:
            params = calculatePhysics()
            if len(params) > 0:
                unitySocket.sendto(str.encode(str(params)), unityAddressPort)
                sidePointsAndTimestamps.clear()
                overheadPointsAndTimestamps.clear()
                sideYpoints.clear()
        keycode = cv2.waitKey(1)
        if keycode == ord("q"): break
        if keycode == ord("a"): # testing
            unitySocket.sendto(str.encode(str([7.0,-5.0,20.0,2.02,1,6.68])), unityAddressPort)
    except Exception as error:
        print("An exception occurred:", error)
        break

  ```
  
  
  
  
