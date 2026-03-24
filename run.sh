#!/bin/bash
# run.sh - Script to run Q-Eval tasks in separate tmux sessions
# Each session has two windows: one for "sbs" evaluation and one for "choice" evaluation

# Make sure the script is executable
# chmod +x run.sh

# Set the base directory
BASE_DIR="./"
PYTHON_CMD="python ${BASE_DIR}/main.py --model gpt-4o"

# Create a function to create a tmux session with two windows for sbs and choice evaluations
create_session() {
    local session_name=$1
    local task_name=$2
    
    # Check if the session already exists
    if tmux has-session -t $session_name 2>/dev/null; then
        echo "Session $session_name already exists. Killing it..."
        tmux kill-session -t $session_name
    fi
    
    # Create a new session with the first window running sbs evaluation
    echo "Creating session $session_name with two windows..."
    tmux new-session -d -s $session_name -n "sbs"
    
    # Send the sbs command to the first window
    local sbs_cmd="$PYTHON_CMD --task $task_name --evaluation_type sbs"
    echo "Running command in $session_name:sbs: $sbs_cmd"
    tmux send-keys -t $session_name:sbs "source ~/.bashrc" C-m
    tmux send-keys -t $session_name:sbs "$sbs_cmd" C-m
    
    # Create a second window for choice evaluation
    tmux new-window -t $session_name -n "choice"
    
    # Send the choice command to the second window
    local choice_cmd="$PYTHON_CMD --task $task_name --evaluation_type choice"
    echo "Running command in $session_name:choice: $choice_cmd"
    tmux send-keys -t $session_name:sbs "source ~/.bashrc" C-m
    tmux send-keys -t $session_name:choice "$choice_cmd" C-m
    
    echo "Session $session_name created with two windows (sbs and choice)."
}

# Create sessions for each task
echo "Setting up tmux sessions for Q-Eval tasks..."

# Image alignment task
create_session "image_alignment" "image_alignment"

# Image quality task
create_session "image_quality" "image_quality"

# Video alignment task
create_session "video_alignment" "video_alignment"

# Video quality task
create_session "video_quality" "video_quality"

echo "All sessions created. Use 'tmux attach-session -t <session_name>' to view a specific session."
echo "Available sessions:"
echo "  - image_alignment (windows: sbs, choice)"
echo "  - image_quality (windows: sbs, choice)"
echo "  - video_alignment (windows: sbs, choice)"
echo "  - video_quality (windows: sbs, choice)"
echo ""
echo "Tmux commands:"
echo "  - List all sessions: tmux list-sessions"
echo "  - Attach to a session: tmux attach-session -t <session_name>"
echo "  - Switch between windows: Ctrl+B, then window number (0, 1) or window name (n)"
echo "  - Detach from a session: Ctrl+B, then D"
echo "  - Kill a session: tmux kill-session -t <session_name>"
