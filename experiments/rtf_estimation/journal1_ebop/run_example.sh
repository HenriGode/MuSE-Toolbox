#!/bin/bash
# filepath: experiments/rtf_estimation/journal1_ebop/run_example.sh

# --- Configuration ---
# 1. Dynamic Workdir (Portable)
WORKDIR=$(pwd)
# 2. Environment Name (Portable - just change this string!)
ENV_NAME="j3"
# 3. Experiment Name (Passed as argument or default)
EXP_NAME="${1:-J1_EBOP_Default}" 
SESSION_NAME="muse_${EXP_NAME}"
# ---------------------

# Ensure we are in the root
if [ ! -f "setup.py" ]; then
    echo "Error: Please run this script from the project root (/data4/Henri/MuSE-Toolbox)"
    exit 1
fi

echo "--> Experiment: $EXP_NAME"
echo "--> Session:    $SESSION_NAME"

# Function to check if a process is running inside a screen session
is_screen_busy() {
    local session="$1"
    local screen_pid=$(screen -ls | grep "[0-9]*\.$session[[:space:]]" | awk '{print $1}' | cut -d. -f1)
    if [ -z "$screen_pid" ]; then return 1; fi
    local shell_pid=$(pgrep -P "$screen_pid")
    if [ ! -z "$shell_pid" ]; then
        local clean_shell_pids=$(echo "$shell_pid" | tr '\n' ',' | sed 's/,$//')
        if pgrep -P "$clean_shell_pids" > /dev/null; then return 0; fi
    fi
    return 1
}

# --- Main Logic ---

# 1. Check State
NEEDS_CREATION=0
READY_TO_RUN=0

if screen -list | grep -q "[0-9]*\.$SESSION_NAME[[:space:]]"; then
    if is_screen_busy "$SESSION_NAME"; then
        echo "❌ Error: Session '$SESSION_NAME' is currently running a job!"
        echo "   Attach manually: screen -r $SESSION_NAME"
        exit 1
    else
        echo "--> Session '$SESSION_NAME' exists and is idle."
        READY_TO_RUN=1
    fi
else
    echo "--> Session '$SESSION_NAME' not found. Creating..."
    NEEDS_CREATION=1
    READY_TO_RUN=1
fi

# 2. Execution Logic
if [ $READY_TO_RUN -eq 1 ]; then
    
    # If we need to create it first
    if [ $NEEDS_CREATION -eq 1 ]; then
        screen -dmS "$SESSION_NAME"
        # Initial setup for new session
        screen -S "$SESSION_NAME" -X stuff "cd $WORKDIR\n"
        sleep 1 # Wait a split second for CD to finish
    fi

    echo "--> Sending new command to session..."
    
    # Define the command
    CMD="conda run -n $ENV_NAME --no-capture-output python -m muse_toolbox.main experiment=rtf_estimation/journal1_ebop/example experiment_name=$EXP_NAME"
    
    # Send the command to the session
    # We add a clear command (Ctrl-C) just in case a prompt is weird, then run the command
    screen -S "$SESSION_NAME" -X stuff "^C" 
    screen -S "$SESSION_NAME" -X stuff "$CMD\n"
    
    # 3. Attach only if this isn't running inside another screen
    if [ -z "$STY" ]; then
        echo "--> Attaching to session..."
        exec screen -r "$SESSION_NAME"
    else
        echo "--> Already inside a screen session ($STY). Not attaching recursively."
    fi
fi