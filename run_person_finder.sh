#!/bin/bash

# Run script for Person Finder Drone
# Usage: ./run_person_finder.sh [train|test|both] [image|text]

# Set default values
MODE="both"
GOAL_TYPE="image"
TARGET_IMAGE=""
TARGET_DESCRIPTION=""
TIMESTEPS=50000
MODEL_PATH="models/person_finder_model"
USE_GUI=true
SIMULATE_DETECTION=true

# Parse command line arguments
if [ $# -ge 1 ]; then
    MODE=$1
fi

if [ $# -ge 2 ]; then
    GOAL_TYPE=$2
fi

# Parse additional named arguments
for i in "$@"; do
    case $i in
        --target-image=*)
        TARGET_IMAGE="${i#*=}"
        shift
        ;;
        --target-description=*)
        TARGET_DESCRIPTION="${i#*=}"
        shift
        ;;
        --timesteps=*)
        TIMESTEPS="${i#*=}"
        shift
        ;;
        --model-path=*)
        MODEL_PATH="${i#*=}"
        shift
        ;;
        --no-gui)
        USE_GUI=false
        shift
        ;;
        --real-detection)
        SIMULATE_DETECTION=false
        shift
        ;;
    esac
done

# Create directories if needed
mkdir -p models
mkdir -p logs

# Build command arguments
COMMON_ARGS=""

if [ "$USE_GUI" = true ]; then
    COMMON_ARGS="$COMMON_ARGS --gui"
fi

if [ "$SIMULATE_DETECTION" = true ]; then
    COMMON_ARGS="$COMMON_ARGS --simulate_detection"
fi

COMMON_ARGS="$COMMON_ARGS --goal_type $GOAL_TYPE --model_path $MODEL_PATH"

if [ ! -z "$TARGET_IMAGE" ]; then
    COMMON_ARGS="$COMMON_ARGS --target_image $TARGET_IMAGE"
fi

if [ ! -z "$TARGET_DESCRIPTION" ]; then
    COMMON_ARGS="$COMMON_ARGS --target_description \"$TARGET_DESCRIPTION\""
fi

# Run based on mode
case $MODE in
    train)
        echo "Training person finder drone with $GOAL_TYPE goal..."
        python train_person_finder.py --train $COMMON_ARGS --timesteps $TIMESTEPS
        ;;
    test)
        echo "Testing person finder drone with $GOAL_TYPE goal..."
        python train_person_finder.py --test $COMMON_ARGS
        ;;
    both)
        echo "Training and testing person finder drone with $GOAL_TYPE goal..."
        python train_person_finder.py --train --test $COMMON_ARGS --timesteps $TIMESTEPS
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: ./run_person_finder.sh [train|test|both] [image|text]"
        echo "Additional options:"
        echo "  --target-image=PATH         Path to target person image"
        echo "  --target-description=TEXT   Text description of target person"
        echo "  --timesteps=N               Number of training timesteps"
        echo "  --model-path=PATH           Path to save/load model"
        echo "  --no-gui                    Run without GUI"
        echo "  --real-detection            Use real detection instead of simulation"
        exit 1
        ;;
esac

echo "Done!"
