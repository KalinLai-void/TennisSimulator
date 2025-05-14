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
|  |
---
## 下載 (Installation)
1. git clone/download this repository. 
2. download the models and lib files from here, and unzip to this repository's root. 
3. create your python env (Anaconda and Python 3.10.9). 
4. activate your env, and pip install -r requirement.txt. 
5. run Main.py! 

---
## How to use?

```bash
git clone https://github.com/<your‑id>/TennisVisionTracker.git
cd TennisVisionTracker
python -m venv .venv         # 或 conda create -n tvt python=3.10
source .venv/Scripts/activate
pip install -r requirements.txt
