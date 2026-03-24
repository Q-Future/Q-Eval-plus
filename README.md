# Q-Eval-plus
## Q-Eval-plus Evaluation Script

```
git clone https://github.com/Q-Future/Q-Eval-plus.git
```
Download info data from [HuggingFace](https://huggingface.co/datasets/q-future/q-eval-plus)

### Project Structure

```
Q-Eval-Plus/
├── main.py                # Main entry point
├── utils.py               # Utility functions
├── call.py                # API calling functions
├── api_config.json        # API configuration
├── tasks/                 # Task implementations
│   ├── base_task.py       # Base task class
│   ├── image_alignment.py # Image alignment task
│   ├── image_quality.py   # Image quality task
│   ├── video_alignment.py # Video alignment task
│   └── video_quality.py   # Video quality task
├── info/                  # Task implementations
│   ├── image_alignment_pairs_test.json       # Image alignment info
│   ├── image_quality_pairs_test.json         # Image quality info
│   ├── video_alignment_pairs_test.json       # Video alignment info
│   └── video_quality_pairs_test.json         # Video quality info
├── logs/                  # Log files
└── <model>/result/        # Results for each model
    └── <mission>.json     # Results for each mission
```


## Usage

### Running Evaluations

#### Option 1: Using main.py directly

To run evaluations, use the `main.py` script:

```bash
python main.py [options]
```

#### Option 2: Using run.sh (Parallel Execution with tmux)

For running multiple tasks in parallel, use the provided `run.sh` script which creates separate tmux sessions for each task, with each session having two windows for "sbs" and "choice" evaluations:

```bash
# Make the script executable (if not already)
chmod +x run.sh

# Run the script
./run.sh
```

This will create four tmux sessions, one for each evaluation task:
- image_alignment (windows: sbs, choice)
- image_quality (windows: sbs, choice)
- video_alignment (windows: sbs, choice)
- video_quality (windows: sbs, choice)

To view a specific session:
```bash
tmux attach-session -t <session_name>
```

Once attached to a session:
- Switch between windows: Press `Ctrl+B`, then window number (0, 1) or window name (n)
- Detach from a session: Press `Ctrl+B`, then `D`

### Command-line Options

- `--model`: Model to evaluate (default: "gpt-4o")
- `--task`: Task to run (choices: "image_alignment", "image_quality", "video_alignment", "video_quality", "all"; default: "all")
- `--evaluation_type`: Type of evaluation to run (choices: "sbs", "choice", "both"; default: "both")
- `--data_root`: Root directory for dataset (default: "/path/to/dataset/AGI-Eval/Q-Eval-100K/")
- `--image_alignment_data`: Path to image alignment test data
- `--image_quality_data`: Path to image quality test data
- `--video_alignment_data`: Path to video alignment test data
- `--video_quality_data`: Path to video quality test data

### Examples

Run all tasks with default settings:

```bash
python main.py
```

Run only image alignment task:

```bash
python main.py --task image_alignment
```

Run only side-by-side evaluations for all tasks:

```bash
python main.py --evaluation_type sbs
```

Run only choice evaluations for video quality:

```bash
python main.py --task video_quality --evaluation_type choice
```

## API Configuration

The API configuration is stored in `api_config.json`. This file is automatically created with default values if it doesn't exist. You can modify this file to use different API keys for different tasks.

Example configuration:

```json
{
    "default": {
        "base_url": "your.api.url",
        "api_key": "sk-yourkey"
    },
    "image_align": {
        "base_url": "your.api.url",
        "api_key": "sk-yourkey"
    },
    "image_quality": {
        "base_url": "your.api.url",
        "api_key": "sk-yourkey"
    },
    ...
}
```

## Results

Results are saved in JSON format at `./result/<model>/<mission>.json`. Each result file includes task attributes and detailed results for each evaluation item.

## Logs

Logs are saved at `./logs/`. Each task has its own log file, and there's also a main log file for the main script.

When running tasks in parallel using the `run.sh` script, each task will log to its own file, making it easier to monitor progress and debug issues for individual tasks. Since each task is now split into "sbs" and "choice" evaluations running in separate windows, you can monitor the progress of each evaluation type independently.
