# Drone
Autonomous UAV Navigation using ROS2 Jazzy, Gazebo Harmonic, and OpenCV featuring dual-camera perception, dynamic obstacle avoidance, autonomous goal alignment, and mission completion.
# 🚁 AI Drone Obstacle Avoidance

An autonomous UAV navigation system developed using **ROS2 Jazzy**, **Gazebo Harmonic**, and **OpenCV** featuring dual-camera perception, dynamic obstacle avoidance, autonomous goal alignment, and mission completion.

---

## 📌 Project Overview

This project demonstrates a fully autonomous drone operating inside a simulated environment.

The UAV is capable of:

- Autonomous Takeoff
- Stable Hover
- Forward Navigation
- Dynamic Obstacle Avoidance
- Goal Detection
- Goal Alignment
- Autonomous Mission Completion

The project uses classical OpenCV techniques instead of deep learning, making it computationally lightweight and suitable for robotics education and research.

---

## ✨ Features

- ROS2 Jazzy
- Gazebo Harmonic
- X3 UAV Simulation
- Dual Camera Architecture
- OpenCV Image Processing
- Dynamic Obstacle Avoidance
- Goal Recognition
- Goal Alignment
- Finite State Machine Navigation
- cmd_vel Velocity Control
- ROS-Gazebo Bridge Integration

---

## 🏗 System Architecture

```
Gazebo Harmonic
        │
        ▼
     X3 UAV
        │
        ▼
 ROS-Gazebo Bridge
        │
        ▼
 ROS2 Navigation Node
        │
 ┌──────┴────────┐
 │               │
 ▼               ▼
Front Camera   Down Camera
 │               │
Obstacle      Goal Detection
Detection         │
 │               ▼
Obstacle     Goal Alignment
Avoidance         │
 └──────┬─────────┘
        ▼
     cmd_vel
        │
        ▼
 Autonomous Flight
```

---

## 🎯 Navigation Pipeline

```
Takeoff

↓

Forward Navigation

↓

Obstacle Detection

↓

Obstacle Avoidance

↓

Goal Detection

↓

Goal Alignment

↓

Mission Complete

↓

Hover
```

---

## 📷 Dual Camera Architecture

### Front Camera

Responsibilities:

- Obstacle Detection
- Free Space Analysis
- Dynamic Obstacle Avoidance

---

### Down Camera

Responsibilities:

- Goal Detection
- Goal Alignment
- Mission Completion

---

## 🛠 Technologies Used

- ROS2 Jazzy
- Gazebo Harmonic
- Python
- OpenCV
- cv_bridge
- URDF
- ros_gz_bridge
- Robot State Publisher

---

## 📁 Project Structure

```
AI-Drone-Obstacle-Avoidance/

├── ros2_ws/
│   └── src/
│       └── opencv_drone_vision/
│
├── docs/
│
├── screenshots/
│
├── videos/
│
└── README.md
```

---

## 📸 Screenshots

Add:

- Gazebo Environment
- Drone Spawn
- Takeoff
- Front Camera
- Down Camera
- Obstacle Avoidance
- Goal Alignment
- Mission Completion

---

## 🚀 Future Work

- YOLOv8 Object Detection
- Path Planning
- SLAM
- PID Altitude Control
- Multi-Goal Navigation
- Real Drone Deployment

---

## 📜 License

MIT License

---

## 👨‍💻 Author

Boiri Sai Kiran

KIIT University

School of Computer Engineering

Artificial Intelligence & Machine Learning
