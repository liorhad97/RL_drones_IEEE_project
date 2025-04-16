#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Person Detection Utilities

This module provides computer vision utilities for person detection,
feature extraction, and matching for the person finder drone system.
"""

import os
import numpy as np
import cv2
import logging
import torch
from typing import Dict, List, Optional, Union, Tuple, Any

# Check if optional dependencies are available
try:
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.transforms import functional as F
    
    TORCHVISION_AVAILABLE = True
except ImportError:
    print("Warning: torchvision not available. Using simulated person detection.")
    TORCHVISION_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class PersonDetector:
    """
    Person detection and recognition module.
    
    This class handles:
    - Person detection in images
    - Feature extraction for person matching
    - Computing match scores between detected and target persons
    
    Attributes:
        detection_threshold (float): Confidence threshold for person detection
        detector (torch.nn.Module): Person detection model
        feature_extractor (torch.nn.Module): Feature extraction model
        device (torch.device): Device for running models
        logger (logging.Logger): Logger for the detector
        simulate_detection (bool): Whether to simulate detection
    """
    
    def __init__(
        self,
        detection_threshold: float = 0.5,
        detector_model_path: Optional[str] = None,
        feature_extractor_path: Optional[str] = None,
        device: str = 'cpu',
        simulate_detection: bool = False
    ):
        """
        Initialize the person detector.
        
        Args:
            detection_threshold (float): Confidence threshold for person detection
            detector_model_path (str, optional): Path to detector model weights
            feature_extractor_path (str, optional): Path to feature extractor weights
            device (str): Device for model inference ('cpu' or 'cuda')
            simulate_detection (bool): Whether to simulate detection
        """
        self.detection_threshold = detection_threshold
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.simulate_detection = simulate_detection or not TORCHVISION_AVAILABLE
        
        # Initialize logger
        self.logger = logging.getLogger('PersonDetector')
        
        # Load detector model
        if not self.simulate_detection:
            self.detector = self._initialize_detector(detector_model_path)
            self.feature_extractor = self._initialize_feature_extractor(feature_extractor_path)
        else:
            self.detector = None
            self.feature_extractor = None
            self.logger.info("Using simulated person detection")
    
    def _initialize_detector(self, model_path: Optional[str]) -> Optional[torch.nn.Module]:
        """
        Initialize person detection model.
        
        Args:
            model_path (str, optional): Path to detector model weights
            
        Returns:
            torch.nn.Module: Detector model
        """
        try:
            # Initialize Faster R-CNN model with pretrained weights
            model = fasterrcnn_resnet50_fpn(pretrained=True)
            model.eval().to(self.device)
            
            # Load custom weights if provided
            if model_path and os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.logger.info(f"Loaded custom detector weights from {model_path}")
            
            return model
        except Exception as e:
            self.logger.error(f"Failed to initialize person detector: {e}")
            self.simulate_detection = True
            return None
    
    def _initialize_feature_extractor(self, model_path: Optional[str]) -> Optional[torch.nn.Module]:
        """
        Initialize feature extraction model for person matching.
        
        Args:
            model_path (str, optional): Path to feature extractor weights
            
        Returns:
            torch.nn.Module: Feature extractor model
        """
        try:
            # Use ResNet50 as feature extractor
            model = torchvision.models.resnet50(pretrained=True)
            # Remove the classification layer
            model = torch.nn.Sequential(*list(model.children())[:-1])
            model.eval().to(self.device)
            
            # Load custom weights if provided
            if model_path and os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.logger.info(f"Loaded custom feature extractor from {model_path}")
            
            return model
        except Exception as e:
            self.logger.error(f"Failed to initialize feature extractor: {e}")
            return None
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from a person image.
        
        Args:
            image (np.ndarray): BGR image of person
            
        Returns:
            np.ndarray: Feature vector
        """
        if self.feature_extractor is None or image is None:
            # Return simulated features
            return self._simulate_features()
        
        try:
            # Convert to RGB and preprocess
            if image.shape[2] == 3 and image.dtype == np.uint8:
                # Assume BGR format from OpenCV
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Check if image is valid
            if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
                self.logger.warning("Invalid image for feature extraction")
                return self._simulate_features()
            
            # Prepare tensor
            input_tensor = F.to_tensor(image)
            # Apply normalization for ImageNet
            input_tensor = F.normalize(input_tensor, 
                                     mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            input_tensor = input_tensor.unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.feature_extractor(input_tensor)
                features = features.squeeze().cpu().numpy()
            
            return features
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return self._simulate_features()
    
    def _simulate_features(self) -> np.ndarray:
        """
        Generate simulated feature vector for testing.
        
        Returns:
            np.ndarray: Simulated feature vector
        """
        # Generate random feature vector
        features = np.random.randn(2048)
        # Normalize to unit length
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def detect_persons(self, image: np.ndarray) -> List[Dict]:
        """
        Detect persons in an image.
        
        Args:
            image (np.ndarray): BGR input image
            
        Returns:
            List[Dict]: List of detected persons with bounding boxes and features
        """
        if self.simulate_detection or image is None:
            return self._simulate_detections(image)
        
        try:
            # Convert to RGB and preprocess
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_tensor = F.to_tensor(rgb_image).to(self.device)
            
            # Perform detection
            with torch.no_grad():
                predictions = self.detector([input_tensor])
            
            # Process detections (keep only persons with confidence > threshold)
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()
            
            # Person class id is 1 in COCO dataset
            persons = []
            for box, score, label in zip(boxes, scores, labels):
                if label == 1 and score >= self.detection_threshold:  # Person class
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Ensure box is within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(image.shape[1], x2)
                    y2 = min(image.shape[0], y2)
                    
                    # Extract person image
                    person_img = image[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else None
                    
                    # Extract features if image is valid
                    features = None
                    if person_img is not None and person_img.size > 0:
                        features = self.extract_features(person_img)
                    
                    persons.append({
                        'box': np.array([x1, y1, x2, y2]),
                        'score': float(score),
                        'features': features,
                        'image': person_img
                    })
            
            self.logger.info(f"Detected {len(persons)} persons in image")
            return persons
        
        except Exception as e:
            self.logger.error(f"Error detecting persons: {e}")
            return self._simulate_detections(image)
    
    def _simulate_detections(self, image: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Simulate person detections for testing.
        
        Args:
            image (np.ndarray, optional): Input image for context
            
        Returns:
            List[Dict]: List of simulated person detections
        """
        # Number of simulated detections (1-3)
        num_detections = np.random.randint(1, 4)
        
        # Get image dimensions for realistic boxes
        if image is not None:
            height, width = image.shape[:2]
        else:
            height, width = 480, 640
        
        persons = []
        for i in range(num_detections):
            # Generate random box
            box_width = np.random.randint(50, width // 3)
            box_height = np.random.randint(100, height // 2)
            x1 = np.random.randint(0, width - box_width)
            y1 = np.random.randint(0, height - box_height)
            x2 = x1 + box_width
            y2 = y1 + box_height
            
            # Create a simulated person image (colored rectangle)
            if image is not None:
                person_img = image[y1:y2, x1:x2].copy()
            else:
                person_img = np.zeros((box_height, box_width, 3), dtype=np.uint8)
                # Add some color variation
                color = np.random.randint(0, 256, size=3)
                person_img[:, :] = color
            
            # Generate random features
            features = self._simulate_features()
            
            persons.append({
                'box': np.array([x1, y1, x2, y2]),
                'score': float(np.random.uniform(0.5, 0.95)),
                'features': features,
                'image': person_img
            })
        
        return persons
    
    def compute_match_score(
        self,
        person_features: np.ndarray,
        target_features: np.ndarray
    ) -> float:
        """
        Compute similarity score between person and target features.
        
        Args:
            person_features (np.ndarray): Features of detected person
            target_features (np.ndarray): Features of target person
            
        Returns:
            float: Similarity score (0-1)
        """
        if person_features is None or target_features is None:
            return 0.0
        
        try:
            if SKLEARN_AVAILABLE:
                # Reshape features for sklearn
                person_features_2d = person_features.reshape(1, -1)
                target_features_2d = target_features.reshape(1, -1)
                similarity = cosine_similarity(person_features_2d, target_features_2d)[0, 0]
            else:
                # Manual cosine similarity calculation
                person_norm = np.linalg.norm(person_features)
                target_norm = np.linalg.norm(target_features)
                
                if person_norm > 0 and target_norm > 0:
                    similarity = np.dot(person_features, target_features) / (person_norm * target_norm)
                else:
                    similarity = 0.0
            
            # Ensure the similarity is in [0, 1]
            similarity = float(max(0, min(1, similarity)))
            return similarity
        
        except Exception as e:
            self.logger.error(f"Error computing match score: {e}")
            return 0.0
    
    def find_best_match(
        self,
        detected_persons: List[Dict],
        target_features: np.ndarray
    ) -> Tuple[int, float]:
        """
        Find the best matching person from detection results.
        
        Args:
            detected_persons (List[Dict]): List of detected persons
            target_features (np.ndarray): Target person features
            
        Returns:
            Tuple[int, float]: Index of best match and match score
        """
        if not detected_persons or target_features is None:
            return -1, 0.0
        
        best_match_idx = -1
        best_score = 0.0
        
        # Compare each detection with target
        for i, person in enumerate(detected_persons):
            if person['features'] is not None:
                score = self.compute_match_score(person['features'], target_features)
                person['match_score'] = score
                
                if score > best_score:
                    best_score = score
                    best_match_idx = i
        
        return best_match_idx, best_score
    
    def visualize_detections(
        self,
        image: np.ndarray,
        detected_persons: List[Dict],
        target_image: Optional[np.ndarray] = None,
        best_match_idx: int = -1
    ) -> np.ndarray:
        """
        Visualize person detections and matching results.
        
        Args:
            image (np.ndarray): Original image
            detected_persons (List[Dict]): Detected persons
            target_image (np.ndarray, optional): Target person image
            best_match_idx (int): Index of best matching person
            
        Returns:
            np.ndarray: Visualization image
        """
        # Create a copy of the image
        vis_img = image.copy() if image is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw detection boxes
        for i, person in enumerate(detected_persons):
            box = person.get('box')
            match_score = person.get('match_score', 0)
            
            if box is not None:
                x1, y1, x2, y2 = map(int, box)
                
                # Color based on match (green for best match, blue for others)
                if i == best_match_idx:
                    color = (0, 255, 0)  # Green for best match
                else:
                    # Color based on match score (blue->red scale)
                    blue = int(255 * (1 - match_score))
                    red = int(255 * match_score)
                    color = (blue, 0, red)
                
                # Draw bounding box
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                
                # Draw match score
                score_text = f"{match_score:.2f}"
                cv2.putText(vis_img, score_text, (x1, y1-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add target image if available
        if target_image is not None:
            h, w = vis_img.shape[:2]
            # Target image size
            target_size = min(150, h // 4)
            # Resize target image
            target_resized = cv2.resize(target_image, (target_size, target_size))
            # Place in top-left corner
            try:
                vis_img[10:10+target_size, 10:10+target_size] = target_resized
                cv2.putText(vis_img, "Target Person", (10, 10+target_size+20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            except:
                # Handle case where target image doesn't fit
                pass
        
        return vis_img


class TextPersonMatcher:
    """
    Text-based person matching module.
    
    This class handles matching between text descriptions and visual detections.
    In a real implementation, this would use text-image models like CLIP.
    For this example, we provide a simulated implementation.
    
    Attributes:
        match_threshold (float): Threshold for considering a match
        logger (logging.Logger): Logger for the matcher
    """
    
    def __init__(self, match_threshold: float = 0.7):
        """
        Initialize the text-based person matcher.
        
        Args:
            match_threshold (float): Threshold for considering a match
        """
        self.match_threshold = match_threshold
        self.logger = logging.getLogger('TextPersonMatcher')
        
        # Pre-defined attributes for simulated matching
        self.attributes = {
            'color': ['red', 'blue', 'green', 'yellow', 'black', 'white'],
            'clothing': ['shirt', 'jacket', 'hat', 'jeans', 'pants', 'dress'],
            'accessory': ['backpack', 'bag', 'glasses', 'cap']
        }
    
    def parse_description(self, description: str) -> Dict[str, List[str]]:
        """
        Parse text description into attributes.
        
        Args:
            description (str): Text description of person
            
        Returns:
            Dict[str, List[str]]: Parsed attributes
        """
        # Simple parsing by checking for known attributes
        parsed = {category: [] for category in self.attributes}
        
        # Convert to lowercase
        description = description.lower()
        
        # Check for each attribute
        for category, terms in self.attributes.items():
            for term in terms:
                if term in description:
                    parsed[category].append(term)
        
        self.logger.info(f"Parsed description: {parsed}")
        return parsed
    
    def match_person(
        self,
        person_image: np.ndarray,
        description: str
    ) -> float:
        """
        Match a person image against a text description.
        
        In a real implementation, this would use a text-image matching model.
        Here we provide a simulated implementation.
        
        Args:
            person_image (np.ndarray): Person image
            description (str): Text description
            
        Returns:
            float: Match score (0-1)
        """
        # Parse description
        parsed = self.parse_description(description)
        
        # Count number of attributes
        total_attributes = sum(len(attrs) for attrs in parsed.values())
        
        if total_attributes == 0:
            return 0.5  # Default score for empty descriptions
        
        # Simulate detection of attributes in the image
        # In a real implementation, use computer vision to detect these attributes
        
        # Generate a random score biased by the number of attributes
        # More specific descriptions will have more variance in match scores
        base_score = np.random.uniform(0.3, 0.7)
        variance = min(0.3, 0.1 * total_attributes)
        match_score = base_score + np.random.uniform(-variance, variance)
        
        # Ensure score is in [0, 1]
        match_score = max(0, min(1, match_score))
        
        return float(match_score)
    
    def find_best_match(
        self,
        detected_persons: List[Dict],
        description: str
    ) -> Tuple[int, float]:
        """
        Find person that best matches the description.
        
        Args:
            detected_persons (List[Dict]): Detected persons
            description (str): Target person description
            
        Returns:
            Tuple[int, float]: Index of best match and match score
        """
        if not detected_persons or not description:
            return -1, 0.0
        
        best_match_idx = -1
        best_score = 0.0
        
        # Match each detection with description
        for i, person in enumerate(detected_persons):
            person_img = person.get('image')
            if person_img is not None:
                score = self.match_person(person_img, description)
                person['match_score'] = score
                
                if score > best_score:
                    best_score = score
                    best_match_idx = i
        
        return best_match_idx, best_score


def get_camera_image(env, width=640, height=480):
    """
    Get camera image from environment.
    
    Args:
        env: Simulation environment
        width (int): Image width
        height (int): Image height
        
    Returns:
        np.ndarray: Camera image or None
    """
    # Try to get image from environment
    # Different simulators have different camera interfaces
    try:
        # Try PyBullet camera
        if hasattr(env, 'env') and hasattr(env.env, 'getDroneImages'):
            # Get drone state for camera position
            drone_state = env.env.getDroneStates()[0]
            drone_pos = drone_state[0:3]
            drone_rpy = drone_state[7:10]
            
            # Configure camera
            camera_width = width
            camera_height = height
            camera_up = [0, 0, 1]  # Z-up
            camera_forward = [1, 0, 0]  # Forward along X
            
            # Get image from drone perspective
            rgb, depth, segmentation = env.env.getDroneImages(
                drone=0,
                segmentation=False
            )
            
            if rgb is not None:
                # Convert to BGR for OpenCV
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except:
        pass
    
    # Fallback to simulation
    return simulate_camera_image(env, width, height)

def simulate_camera_image(env, width=640, height=480):
    """
    Simulate a camera image based on environment state.
    
    Args:
        env: Simulation environment
        width (int): Image width
        height (int): Image height
        
    Returns:
        np.ndarray: Simulated camera image
    """
    # Create blank image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Get drone position if available
    position = None
    try:
        if hasattr(env, 'last_obs'):
            position = env.last_obs[:3]
        elif hasattr(env, 'env') and hasattr(env.env, 'getDroneStates'):
            drone_state = env.env.getDroneStates()[0]
            position = drone_state[0:3]
    except:
        position = np.array([0, 0, 0])
    
    # Draw a simple simulated view
    if position is not None:
        # Draw horizon line
        horizon_y = int(height/2 - position[2] * 30)
        horizon_y = max(0, min(height-1, horizon_y))
        cv2.line(image, (0, horizon_y), (width, horizon_y), (100, 100, 100), 2)
        
        # Draw ground grid
        grid_spacing = 50
        for i in range(0, width, grid_spacing):
            cv2.line(image, (i, horizon_y), (width//2, height), (50, 50, 50), 1)
        for i in range(0, width, grid_spacing):
            cv2.line(image, (i, horizon_y), (0, height), (50, 50, 50), 1)
        
        # Add a sky
        image[:horizon_y, :] = (135, 206, 235)  # Light blue
    
    return image

def create_target_person_image(color=(255, 0, 0), size=(200, 400)):
    """
    Create a simple target person image for testing.
    
    Args:
        color (tuple): BGR color
        size (tuple): Image size (width, height)
        
    Returns:
        np.ndarray: Target person image
    """
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw a simple person shape
    # Head
    head_radius = width // 4
    head_center = (width // 2, height // 6)
    cv2.circle(image, head_center, head_radius, color, -1)
    
    # Body
    body_top = (width // 2, height // 4)
    body_bottom = (width // 2, 3 * height // 4)
    body_width = width // 3
    cv2.rectangle(image, 
                (body_top[0] - body_width//2, body_top[1]),
                (body_bottom[0] + body_width//2, body_bottom[1]),
                color, -1)
    
    # Legs
    leg_width = width // 6
    leg_height = height // 4
    # Left leg
    cv2.rectangle(image,
                (width//2 - body_width//4 - leg_width//2, body_bottom[1]),
                (width//2 - body_width//4 + leg_width//2, body_bottom[1] + leg_height),
                color, -1)
    # Right leg
    cv2.rectangle(image,
                (width//2 + body_width//4 - leg_width//2, body_bottom[1]),
                (width//2 + body_width//4 + leg_width//2, body_bottom[1] + leg_height),
                color, -1)
    
    # Arms
    arm_width = width // 6
    arm_height = height // 3
    # Left arm
    cv2.rectangle(image,
                (width//2 - body_width//2 - arm_width, body_top[1] + height//10),
                (width//2 - body_width//2, body_top[1] + height//10 + arm_height),
                color, -1)
    # Right arm
    cv2.rectangle(image,
                (width//2 + body_width//2, body_top[1] + height//10),
                (width//2 + body_width//2 + arm_width, body_top[1] + height//10 + arm_height),
                color, -1)
    
    return image
