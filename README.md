Last Demo Video: [https://drive.google.com/file/d/1IS7XXpkDAJFSkqCL16LljOe32Ng_hmkz/view?usp=sharing](https://drive.google.com/file/d/1IS7XXpkDAJFSkqCL16LljOe32Ng_hmkz/view?usp=sharing)
# TennisSimulator
## Intro

This project simulates tennis ball trajectories in Unity using the coordinates predicted by [TrackNetV3](https://github.com/alenzenx/TrackNetV3/tree/main) and video frames from both overhead and side views. The system converts pixel coordinates to real‑world meters, then streams 3‑D positions to Unity via UDP for trajectory playback.  
### 相機視角轉換(Camera perspective conversion) 
![GitHub图像](/Camera-perspective-conversion.png)
---

## 前置準備 (Pre‑Requirements)
### TrackNetV3
Follow the steps in this link to download [TracketV3](https://github.com/alenzenx/TrackNetV3/tree/main)
### 開發環境 (Environment and requirements)
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
## How to use?
### set path
cap is the video path  
csv is the coordinates of the tennis ball in the film 
```python
cap1=cv2.VideoCapture(r"D:/tennis_MovieData/Datasets/Datasets/Outdoor Field/Cross-court Shot/Side-View/TrackNet/OCS17_pred.mp4")
cap2=cv2.VideoCapture(r"D:/tennis_MovieData/Datasets/Datasets/Outdoor Field/Cross-court Shot/Top-View/TrackNet/OCT17_pred.mp4")
csv_cap1=(r"D:/tennis_MovieData/Datasets/Datasets/Outdoor Field/Cross-court Shot/Side-View/TrackNet/OCS17_ball.csv")
csv_cap2=(r"D:/tennis_MovieData/Datasets/Datasets/Outdoor Field/Cross-court Shot/Top-View/TrackNet/OCT17_ball.csv")

```
### Set the corner of the court to view the camera
```python
pX1, pY1 = 507, 296 # left-top corner
pX2, pY2 = 1315, 283 # right-top corner
pX3, pY3 = 112, 909 # left-bottom corner
pX4, pY4 = 1717, 872 # right-bottom corner
```

