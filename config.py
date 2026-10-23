"""Configuration shared by the Windows-APT next-technique forecasting scripts."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "Windows-APT 2025 A Dataset for APT-Inspired Attack"
DEFAULT_DATASET_ZIP = PROJECT_DIR / "dataset-archive" / "Windows-APT-2025-dataset.zip"
OUTPUT_DIR = PROJECT_DIR / "outputs"

RANDOM_SEED = 42
MAX_PREFIX_LENGTH = 15
MAX_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 3
EMBEDDING_DIM = 32
GRU_HIDDEN_DIM = 64
DROPOUT = 0.30
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
ATTENTION_CONFIDENCE_THRESHOLD = 0.50
SUSTAINED_WARNING_STEPS = 2
LEAD_TIME_TARGETS_PER_CLASS = 250
LEAD_TIME_CANDIDATE_POSITIONS = 20
FAITHFULNESS_SAMPLE_SIZE = 200

# The paper's Table I label space. IDs are the internal canonical labels.
SELECTED_TECHNIQUES = [
    ("T1105", "Ingress Tool Transfer"),
    ("T1059.003", "Windows Command Shell"),
    ("T1112", "Modify Registry"),
    ("T1087", "Account Discovery"),
    ("T1570", "Lateral Tool Transfer"),
    ("T1078", "Valid Accounts"),
    ("T1053.005", "Scheduled Task"),
    ("T1059", "Command and Scripting Interpreter"),
    ("T1055", "Process Injection"),
    ("T1565.001", "Stored Data Manipulation"),
    ("T1070.004", "File Deletion"),
    ("T1059.001", "PowerShell"),
]
CLASS_IDS = [item[0] for item in SELECTED_TECHNIQUES]
CLASS_NAMES = [item[1] for item in SELECTED_TECHNIQUES]
CLASS_TO_INDEX = {technique_id: idx for idx, technique_id in enumerate(CLASS_IDS)}
ID_TO_NAME = dict(SELECTED_TECHNIQUES)
NAME_TO_ID = {name.casefold(): technique_id for technique_id, name in SELECTED_TECHNIQUES}

# This split reproduces the file-level temporal partition stated in the paper.
TRAIN_FILES = [
    "28-October-01-November.csv",
    "1-11-November.csv",
    "11-16-November.csv",
    "23-30-November.csv",
    "30-November.csv",
]
TEST_FILES = [
    "01-03-December.csv",
    "03-04-December.csv",
    "04-07-December.csv",
    "07-10-December.csv",
    "10-December-P1.csv",
    "10-December-P2.csv",
    "11-12-December.csv",
    "12-13-December.csv",
    "13-14-December.csv",
    "14-17-December.csv",
    "22-December.csv",
]

# Dataset schema varied between collection periods. Keep aliases here for detection.
TIMESTAMP_COLUMNS = ["_source.@timestamp", "_source.timestamp", "Timestamp", "timestamp"]
AGENT_NAME_COLUMNS = ["_source.agent.name", "Agent-Name", "agent.name"]
AGENT_ID_COLUMNS = ["_source.agent.id", "Agent-ID", "agent.id"]
TECHNIQUE_NAME_COLUMNS = ["_source.rule.mitre.technique", "MITRE Technique", "Technique"]
TECHNIQUE_ID_COLUMNS = ["_source.rule.mitre.id", "MITRE Technique ID"]
TECHNIQUE_ID_FALLBACK_COLUMNS = ["_source.data.sca.check.compliance.mitre_techniques"]
PROCESS_IMAGE_COLUMNS = ["_source.data.win.eventdata.image", "_source.data.win.eventdata.imagePath", "_source.data.process.executable", "_source.data.process.name"]
PROCESS_ID_COLUMNS = ["_source.data.win.eventdata.processId", "_source.data.win.system.processID", "_source.data.win.eventdata.processID"]
COMMAND_COLUMNS = ["_source.data.win.eventdata.commandLine", "_source.data.win.eventdata.parentCommandLine", "_source.data.process.args"]

# Names and aliases that occur in the CSVs. Class IDs remain canonical.
TECHNIQUE_ALIASES = {
    "ingress tool transfer": "T1105",
    "windows command shell": "T1059.003",
    "modify registry": "T1112",
    "account discovery": "T1087",
    "lateral tool transfer": "T1570",
    "valid accounts": "T1078",
    "scheduled task/job": "T1053.005",
    "scheduled task": "T1053.005",
    "command and scripting interpreter": "T1059",
    "process injection": "T1055",
    "stored data manipulation": "T1565.001",
    "file deletion": "T1070.004",
    "file and directory discovery": "T1083",
    "powershell": "T1059.001",
}
