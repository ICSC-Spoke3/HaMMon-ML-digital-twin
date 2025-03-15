#!/bin/bash
# usage:
# nohup ./launcher.sh 88 training > ./.outputs/nohup.training 2>&1 &
# nohup ./launcher.sh 89 validation > ./.outputs/nohup.validation 2>&1 &
# nohup ./launcher.sh 94 other > ./.outputs/nohup.other 2>&1 &
# nohup ./launcher.sh 88 full > ./.outputs/nohup.full 2>&1 &

# Customizable waiting variable
WAIT_SECONDS=10

# Parameters must be given as input
if [ $# -lt 2 ]; then
  echo "Error: You must provide a number for the log file and a mode (training, validation, test, full)."
  exit 1
fi

# Assign parameters to variables
log_number="$1"
mode="$2"

# Check if the mode is valid
if [[ "$mode" != "training" && "$mode" != "validation" && "$mode" != "test" && "$mode" != "other" && "$mode" != "full" ]]; then
  echo "Error: The mode must be 'training', 'validation', 'test', 'other' or 'full'"
  exit 1
fi

# timestamp function
get_timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

# Check if the settings.yaml file exists
if [ ! -f settings.yaml ]; then
  echo "Error: The settings.yaml file does not exist."
  exit 1
fi

# Use Python to parse the YAML file and extract values
read_yaml_value() {
  # Run a Python command to read the value from the settings.yaml file
  python - <<EOF
import yaml

with open('settings.yaml', 'r') as file:
    settings = yaml.safe_load(file)
    value = settings.get('$1', None)
    if value:
        print(value)
EOF
}

# Store the current directory
initial_directory=$(pwd)

# Infinite loop to verify and launch the script
while true; do
  # Return to the initial directory
  cd "$initial_directory" || exit

  # Read paths from the settings.yaml file using yq
  scripts_root_folder=$(read_yaml_value 'scripts_root_folder')
  current_experiment=$(read_yaml_value 'current_experiment')  

  # Ensure the values were retrieved successfully
  if [ -z "$scripts_root_folder" ] || [ -z "$current_experiment" ]; then
    echo "Error: Failed to retrieve required values from settings.yaml."
    exit 1
  fi

  # Check if current_experiment is not "none"
  if [ "$current_experiment" = "none" ]; then
    sleep $WAIT_SECONDS
    continue
  fi

  # Define the full path of the Python script and the log folder
  experiment_folder="${scripts_root_folder}/${current_experiment}"
  log_folder="${experiment_folder}/logs"
  echo "current exp ${current_experiment}"

  # Check if the experiment folder exists
  if [ ! -d "$experiment_folder" ]; then
    echo "Errore: La cartella dell'esperimento '$experiment_folder' non esiste. Attesa di $WAIT_SECONDS secondi."
    sleep $WAIT_SECONDS
    continue
  fi

  # Create the log folder if it does not exist
  mkdir -p "$log_folder"

  # timestamp
  timestamp=$(get_timestamp)

  # Define the path of the log file including the provided number
  log_file="${log_folder}/${log_number}_${mode}.log"

  # Create the log file if it does not exist
  touch "$log_file"

  # Change the current directory to that of the experiment
  cd "$experiment_folder" || exit


  # Execute the Python script and redirect the output to the log file
  if [ "$mode" = "full" ]; then
    python -u "${experiment_folder}/script_training.py" >> "$log_file" 2>&1
    python -u "${experiment_folder}/script_validation.py" >> "$log_file" 2>&1
  else
    python -u "${experiment_folder}/script_${mode}.py" >> "$log_file" 2>&1
  fi

  # Calculate the elapsed time
  end_timestamp=$(get_timestamp)
  start_seconds=$(date -d "$timestamp" +%s)
  end_seconds=$(date -d "$end_timestamp" +%s)
  elapsed_seconds=$((end_seconds - start_seconds))

  # Wait until at least WAIT_SECONDS has passed
  if [ $elapsed_seconds -lt $WAIT_SECONDS ]; then
    sleep $((WAIT_SECONDS - elapsed_seconds))
  fi

done