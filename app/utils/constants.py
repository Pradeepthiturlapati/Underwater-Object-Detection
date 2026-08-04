# --- MODEL CONFIGURATION ---
MODEL_FILE_PATH = '../hybrid_combined_model.keras'
IMG_SIZE = (224, 224)

# --- CLASS LABELS ---
# These must be in the exact order (index 0 to 31) that the model was trained on.
CLASS_NAMES = [
    'ArmyDiver1', 'ArmyDiver2', 'ArmyDiver3', 'Ballena', 'BlueFish1', 'BlueFish2', 
    'BoySwimming', 'CenoteAngelita', 'DeepSeaFish', 'Dolphin1', 'Dolphin2', 
    'FishFollowing', 'Fisherman', 'GarryFish', 'HoverFish1', 'HoverFish2', 
    'JerkbaitBites', 'MonsterCreature1', 'MonsterCreature2', 'Octopus1', 
    'Octopus2', 'PinkFish', 'SeaDiver', 'SeaDragon', 'SeaTurtle1', 'SeaTurtle2', 
    'SeaTurtle3', 'Steinlager', 'WhaleAtBeach1', 'WhaleAtBeach2', 'WhaleDiving', 
    'WhiteShark'
]

# --- TRAINING METRICS ---
FINAL_METRICS = {
    "Best Validation Accuracy": 0.9799,
    "Final Training Accuracy": 0.9774,
    "Final Training Loss": 0.0373,
    "Final Validation Loss": 0.0468,
}

# --- SAMPLE FILES (Relative to the streamlit_app.py) ---
SAMPLE_IMAGE_PATH = './sample/sea_diver.jpg'
SAMPLE_VIDEO_PATH = './sample/ballena.mp4'

# --- IMAGE PATHS (Relative to the streamlit_app.py) ---
ACCURACY_CURVE_PATH = 'images/accuracy_curve.png'
LOSS_CURVE_PATH = 'images/loss_curve.png'