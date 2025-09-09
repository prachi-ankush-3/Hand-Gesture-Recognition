import streamlit as st
import yaml
import hashlib
import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import tensorflow as tf
import time

# Set page config
st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="👋",
    layout="wide"
)

# --- 🔹 User Authentication Functions ---
USER_FILE = "users.yaml"

def load_users():
    """Load user credentials from a YAML file."""
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as file:
            return yaml.safe_load(file)
    return {}

def save_users(users):
    """Save user credentials to a YAML file."""
    with open(USER_FILE, "w") as file:
        yaml.dump(users, file)

def hash_password(password):
    """Hash the password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def login():
    """Login page"""
    st.title("🔐 Login to Access Hand Gesture App")

    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")
    login_btn = st.button("Login")

    users = load_users()
    
    if login_btn:
        if username in users and users[username] == hash_password(password):
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success(f"✅ Logged in as {username}")
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

def register():
    """Registration page"""
    st.title("📝 Register a New Account")

    new_username = st.text_input("👤 Choose a Username")
    new_password = st.text_input("🔑 Choose a Password", type="password")
    register_btn = st.button("Register")

    users = load_users()

    if register_btn:
        if new_username in users:
            st.error("❌ Username already exists! Choose another.")
        elif len(new_password) < 6:
            st.error("❌ Password must be at least 6 characters long.")
        else:
            users[new_username] = hash_password(new_password)
            save_users(users)
            st.success("✅ Registration successful! Please log in.")
            st.balloons()

# --- 🔹 Handle Authentication ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    option = st.sidebar.radio("🔑 Choose an Option", ["Login", "Register"])
    
    if option == "Login":
        login()
    else:
        register()

    st.stop()  # Prevent the main app from loading until logged in

# --- 🔹 Logout Function ---
if st.sidebar.button("🚪 Logout"):
    st.session_state["logged_in"] = False
    st.session_state["username"] = None
    st.rerun()

st.sidebar.success(f"✅ Logged in as {st.session_state['username']}")

# --- 🎥 Gesture Recognition Code ---
st.title("Hand Gesture to Text : Voice for Voiceless")

st.markdown("""
    This application detects hand gestures to recognize American Sign Language (ASL) alphabets.
    Position your hand in front of the camera to see it in action!
""")

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    detection_confidence = st.slider("Detection Confidence", 0.5, 1.0, 0.8, 0.05)
    buffer_size = st.slider("Prediction Stability (frames)", 1, 10, 3, 1)

    # Display the ASL alphabet reference
    st.header("ASL Alphabet Reference")
    st.image(os.path.join(os.getcwd(), "reference.jpg"), 
             caption="ASL Alphabet Reference")

# Load model
@st.cache_resource
def load_gesture_model():
    try:
        model = tf.keras.models.load_model("gesture_model_finetuned2.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Initialize components
detector = None
model = None
gesture_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Initialize session state variables
if 'previous_prediction' not in st.session_state:
    st.session_state.previous_prediction = None
if 'prediction_buffer' not in st.session_state:
    st.session_state.prediction_buffer = []
if 'predicted_label' not in st.session_state:
    st.session_state.predicted_label = ""
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0.0
if 'running' not in st.session_state:
    st.session_state.running = False

# Main interface layout with columns
col1, col2 = st.columns([2, 1])

# Create placeholders for video and skeleton
with col1:
    video_placeholder = st.empty()
with col2:
    skeleton_placeholder = st.empty()
    
        # Create a styled box for the alphabet display
    st.markdown("""
        <style>
        .alphabet-box {
            border: 5px solid #4CAF50;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
            height: 200px;
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #f0f2f6;
        }
        .alphabet-box h1 {
            color: black;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3>Detected Alphabet</h3>", unsafe_allow_html=True)
    alphabet_box = st.empty()
    
    # Confidence display
    confidence_text = st.empty()

# Button columns
button_col1, button_col2 = st.columns(2)

# Start and stop buttons
with button_col1:
    start_button = st.button("Start Camera", key="start_button", disabled=st.session_state.running)

with button_col2:
    stop_button = st.button("Stop Camera", key="stop_button", disabled=not st.session_state.running)

# Main app logic
if start_button:
    st.session_state.running = True
    
    # Load the model
    model = load_gesture_model()
    if model is None:
        st.error("Failed to load the gesture recognition model. Please check if the model file exists.")
        st.session_state.running = False
        st.experimental_rerun()
    
    # Initialize detector
    detector = HandDetector(maxHands=1, detectionCon=detection_confidence)

# Function to run the camera processing
def process_camera():
    cap = cv2.VideoCapture(0) #Starts webcam.
    
    while st.session_state.running and cap.isOpened(): #    Loop runs while the webcam is on and app is running.
        success, frame = cap.read() #Reads a frame (image) from webcam.
        if not success:
            st.error("Failed to capture image from camera") #  If webcam fails, show error and stop.
            break
            
        # Process frame
        frame = cv2.flip(frame, 1) # Flips the frame for mirror view.
        display_frame = frame.copy() #  Makes a copy of the frame for showing on screen.
        
        # Finds hand in the frame using MediaPipe (no drawing).
        hands, img = detector.findHands(frame, draw=False, flipType=True)
        
        #Creates a white 224x224 image to draw skeleton.
        skeleton_canvas = np.ones((224, 224, 3), dtype=np.uint8) * 255
        
        #  If a hand is found, pick the first hand.
        if hands:
            hand = hands[0]
            
            # Get list of landmark points of the hand. 
            if 'lmList' in hand:
                landmarks = hand['lmList']
                
                # Define connections for hand skeleton
                # Defines which landmarks to connect (fingers + palm).
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
                    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
                    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
                    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                    (5, 9), (9, 13), (13, 17)  # Palm connections
                ]
                
                # Scale landmarks
                x, y, w, h = hand['bbox'] #Get bounding box around the hand.
                scaled_landmarks = [] #  Empty list to store scaled points.
                
                #Scales landmarks to fit in 224x224 canvas.
                for lm in landmarks:
                    x_point, y_point, _ = lm
                    scaled_x = int((x_point - x) * 224 / max(w, 1))
                    scaled_y = int((y_point - y) * 224 / max(h, 1))
                    scaled_landmarks.append((scaled_x, scaled_y))
                
                #  Draws green lines (skeleton) between points.
                # Scales landmarks to fit in 224x224 canvas.
                for connection in connections:
                    start_idx, end_idx = connection
                    if start_idx < len(scaled_landmarks) and end_idx < len(scaled_landmarks):
                        pt1 = scaled_landmarks[start_idx]
                        pt2 = scaled_landmarks[end_idx]
                        cv2.line(skeleton_canvas, pt1, pt2, (0, 255, 0), 2)
                
                #   Draws red dots on each landmark. 
                for point in scaled_landmarks:
                    cv2.circle(skeleton_canvas, point, 3, (0, 0, 255), -1)
                
                # Prepares image for model (normalize and expand dimensions).
                skeleton_input = np.expand_dims(skeleton_canvas, axis=0)
                skeleton_input = skeleton_input / 255.0
                
                #Model predicts which alphabet it sees.
                prediction = model.predict(skeleton_input, verbose=0)
                predicted_idx = np.argmax(prediction) 
                # Gets highest score (index) and stores confidence.
                st.session_state.confidence = float(np.max(prediction))
                
                # Adds prediction to buffer
                st.session_state.prediction_buffer.append(predicted_idx)
                
                # Stabilize prediction
                if len(st.session_state.prediction_buffer) >= buffer_size:
                    most_common = max(set(st.session_state.prediction_buffer), key=st.session_state.prediction_buffer.count)
                    current_label = gesture_labels[most_common]
                    
                    # Only update if the prediction changed
                    if current_label != st.session_state.predicted_label:
                        st.session_state.predicted_label = current_label
                        
                        # Update the alphabet display box
                        alphabet_box.markdown(
                            f"""
                            <div class="alphabet-box">
                                <h1 style='font-size: 100px;'>{st.session_state.predicted_label}</h1>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    
                    # Keep buffer at fixed size
                    st.session_state.prediction_buffer = st.session_state.prediction_buffer[-buffer_size:]
                
                # Draw hand landmarks on display frame
                for connection in connections:
                    start_idx, end_idx = connection
                    if start_idx < len(landmarks) and end_idx < len(landmarks):
                        start_point = (landmarks[start_idx][0], landmarks[start_idx][1])
                        end_point = (landmarks[end_idx][0], landmarks[end_idx][1])
                        cv2.line(display_frame, start_point, end_point, (0, 255, 0), 2)
                
                # Draws red dots on hand in the display view. 
                for lm in landmarks:
                    cv2.circle(display_frame, (lm[0], lm[1]), 3, (0, 0, 255), -1)
        else:
            # Clear prediction buffer when no hand is detected
            st.session_state.prediction_buffer = []
        
        # Display the predicted alphabet on frame
        cv2.putText(display_frame, f"Detected: {st.session_state.predicted_label}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert frames to RGB for Streamlit
        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        skeleton_rgb = cv2.cvtColor(skeleton_canvas, cv2.COLOR_BGR2RGB)
        
        # Display video and skeleton
        video_placeholder.image(display_frame_rgb, channels="RGB", use_container_width=True)
        skeleton_placeholder.image(skeleton_rgb, channels="RGB", use_container_width=True, caption="Hand Skeleton")
        
        # Update confidence display
        confidence_text.progress(st.session_state.confidence) #Shows how confident the model is.
        
        #  Stop the loop if user clicks stop.
        if not st.session_state.running:
            break
            
        # Wait to reduce CPU usage
        time.sleep(0.03) # Wait a little to reduce CPU usage.
    
    # Stops the camera, updates UI, and refreshes.
    cap.release()
    st.session_state.running = False
    st.experimental_rerun()  # Refresh the UI

# Handle stop button
if stop_button:
    st.session_state.running = False
    st.success("Camera stopped")
    st.rerun()

# Run camera processing if in running state
if st.session_state.running:
    with st.spinner('Starting camera...'):
        process_camera()
else:
    # Display placeholder image when not started
 st.image("reference.jpg", 
         caption="ASL Alphabet Reference", 
         use_container_width=True)