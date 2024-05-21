import gym
from gym import spaces
import numpy as np
import torch
import random
from torchvision.transforms import transforms
import pyrealsense2 as rs
import cv2
from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class ObjectDetectionEnv(gym.Env):
    def __init__(self, model):
        super(ObjectDetectionEnv, self).__init__()
        self.silo_states = np.zeros(5)  # Stores the number of balls in each silo
        self.ball_color = None  # Color of the ball (1 for red, -1 for blue)
        self.team_color = None  # Current team color (1 for red, -1 for blue)
        self.game_over = False
        self.model = model  # Your PyTorch object detection model
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 480)),  # Resize to a size divisible by 32
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to [0, 1]
        ])
        self.action_space = spaces.Discrete(5)  # Number of silos
        self.observation_space = spaces.Dict({
            "silo_states": spaces.Box(low=0, high=3, shape=(5,), dtype=np.float32),
            "ball_color": spaces.Discrete(2),  # Red or blue
            "team_color": spaces.Discrete(2)   # Red or blue
        })

    def reset(self):
        # Reset environment to initial state
        self.silo_states = np.zeros(5)
        self.ball_color = None
        self.team_color = np.random.choice([0, 1])  # Randomly assign team color (0 for red, 1 for blue)
        self.game_over = False
        return self._get_observation()

    def perform_object_detection(self):
        detections = []  # List to store detections
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        pipeline.start(config)

        try:
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # Convert color frame to numpy array
                frame = np.asanyarray(color_frame.get_data())

                # Perform object detection using the model
                with torch.no_grad():
                    detections = self.model(frame)

                # Append detections to the list
                detections.append(detections)

                # Process detections
                for detection in detections:
                    label, confidence, bbox = 0, 0, [0]*29
                    if len(detection) >= 31:
                        label = int(detection[30])  # Assuming detection format: [x_min, y_min, x_max, y_max, confidence, class_label]
                        confidence = detection[29]
                        bbox = detection[:29]

                    # Render bounding box and label on the frame
                    if confidence > 0.5:
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                        cv2.putText(frame, str(label), (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Display the frame
                cv2.imshow('Object Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            # Stop streaming
            pipeline.stop()
            cv2.destroyAllWindows()
        return detections

    def step(self, action):
        # Execute action and return new state, reward, and done flag
        if self.game_over:
            raise ValueError("Episode is already done. Please reset the environment.")

        # Perform action (place ball in silo)
        self.silo_states[action] += 1

        # Perform live object detection
        detections = self.perform_object_detection()

        # Update state with information about detected objects
        self.update_state_with_detections(detections)

        # Check if any silo is full
        if np.any(self.silo_states >= 3):
            # Check for winning condition
            if np.sum(self.silo_states[self.silo_states >= 3]) >= 3:
                # Team wins if at least 3 silos are filled
                reward = 100  # High reward for winning
                self.game_over = True
            else:
                # Count balls in filled silos as points
                reward = np.sum(self.silo_states[self.silo_states >= 3]) * 30
                self.game_over = True
        else:
            reward = 0  # No reward if the game is still ongoing

        # Return observation, reward, done flag, and additional information
        return self._get_observation(), reward, self.game_over, {}
    
    def update_state_with_detections(self, detections):
        # Process detections
        for detection in detections:
            label, confidence, bbox = 0, 0, [0]*29
            if len(detection) >= 31:
                label = int(detection[30])  # Assuming detection format: [x_min, y_min, x_max, y_max, confidence, class_label]
                confidence = detection[29]
                bbox = detection[:29]

            if label == "red ball" and confidence > 0.5:
                self.ball_color = 0
            elif label == "blue ball" and confidence > 0.5:
                self.ball_color = 1
            elif label == "NULL" and confidence > 0.5:
                # Check if ball is in a silo
                for i, silo_bbox in enumerate(self.get_silo_bboxes()):
                    if self.check_overlap(bbox, silo_bbox):
                        self.silo_states[i] -= 1
                        break

    def _get_observation(self):
        # Return current observation (state of the game)
        return {
            "silo_states": self.silo_states.copy(),
            "ball_color": self.ball_color if self.ball_color is not None else -1,  # Use -1 as default value if ball_color is None
            "team_color": self.team_color if self.team_color is not None else -1  # Use -1 as default value if team_color is None
        }

    def render(self, mode='human'):
        # Rendering code using OpenCV
        frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Create a blank frame
        # Add visualization of silo states, ball color, team color, etc.
        cv2.putText(frame, f"Silo states: {self.silo_states}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Ball color: {'Red' if self.ball_color == 0 else 'Blue' if self.ball_color == 1 else 'Unknown'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Team color: {'Red' if self.team_color == 0 else 'Blue' if self.team_color == 1 else 'Unknown'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow('Environment', frame)
        cv2.waitKey(1)  # Add a slight delay to display the frame (1ms)

        # You can also save the frame as an image or write to a video file if needed
        # cv2.imwrite('environment_frame.png', frame)
        # out.write(frame)  # Assuming 'out' is a video writer object

        # Optionally return the rendered frame or other visualization data
        return frame


def main():
    # Define your PyTorch object detection model
    model = YOLO("/home/cadt-02/Downloads/model_- 6 may 2024 19_25.pt")

    env = ObjectDetectionEnv(model)
    # Wrap the environment in a DummyVecEnv to make it compatible with stable-baselines3
    env = DummyVecEnv([lambda: env])

    # Create the RL agent using PPO algorithm
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_logs/")

    # Train the agent
    model.learn(total_timesteps=int(1e5))

if __name__ == "__main__":
    main()
