Last Demo Video: [https://drive.google.com/file/d/1IS7XXpkDAJFSkqCL16LljOe32Ng_hmkz/view?usp=sharing](https://drive.google.com/file/d/1IS7XXpkDAJFSkqCL16LljOe32Ng_hmkz/view?usp=sharing)
# TennisSimulator
## Intro

A vision‑based pipeline that detects tennis ball trajectories from *dual‑view* videos using classic computer‑vision (background subtraction + HSV color segmentation + K‑Means). The system converts pixel coordinates to real‑world meters, then streams 3‑D positions to Unity via UDP for trajectory playback.  

---

## 目錄
1. [前置準備](#前置準備-pre‑requirements)  
2. [開發環境](#開發環境)  
3. [安裝](#安裝)  
4. [使用方式](#使用方式)  
5. [檔案結構](#檔案結構)  
7. [License](#license)  
8. [Citation](#citation)

---

## 前置準備 (Pre‑Requirements)

| demand | directions |
| --- | --- |
| Anaconda 3 | [Download](https://www.anaconda.com/) |
| Python | version 3.8.18 |
| Unity | version 2021.3.18f1 |

---

## 開發環境

| 套件 | 版本（建議） |
| --- | --- |
| opencv‑python | 4.8+ |
| numpy | 1.24+ |
| imutils | 0.5+ |
| scikit‑learn | 1.4+ |
| **其他** | 详见 `requirements.txt` |

> 以上依程式 `import` 列表整理。:contentReference[oaicite:3]{index=3}

---
## How to execution

## 安裝

```bash
git clone https://github.com/<your‑id>/TennisVisionTracker.git
cd TennisVisionTracker
python -m venv .venv         # 或 conda create -n tvt python=3.10
source .venv/Scripts/activate
pip install -r requirements.txt
