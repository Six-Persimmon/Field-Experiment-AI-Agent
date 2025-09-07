# Field-Experiment-AI-Agent

AI Agent framework for conducting consumer behavior experiments.

**Authors:**

* Xiao Liu (Associate Professor, NYU) [xl23@stern.nyu.edu](mailto:xl23@stern.nyu.edu)
* Shijian Liu (PhD, NYU) [sl9818@stern.nyu.edu](mailto:sl9818@stern.nyu.edu)
* Jiayu Li (PhD, NYU) [jl15681@stern.nyu.edu](mailto:jl15681@stern.nyu.edu)
* Manlu Ouyang (PhD, NYU) [mo2615@stern.nyu.edu](mailto:mo2615@stern.nyu.edu)
* Sichen Zhong (RA, NYU) [sz4972@nyu.edu](mailto:sz4972@nyu.edu)
* Qianyu Zhang (RA, NYU) [qz1104@nyu.edu](mailto:qz1104@nyu.edu)

# Survey Enhancement & Deployment System

## Overview

This project provides an interactive, AI-powered survey enhancement and deployment system. First-time users can:

* Convert raw text surveys into structured JSON
* Iteratively enhance surveys with AI feedback
* Deploy surveys to Qualtrics and optionally create MTurk HITs
* Collect both human and simulated survey responses
* Generate research papers from CSV data

The system uses Pydantic models, CrewAI agents, OpenAI, Qualtrics API, and MTurk API, offering a seamless end-to-end workflow for academic and market research.

## Quick Start (Recommended Entry & Flow)

- Terminal entry: `python survey.py`
  - Interactive menu to create/enhance surveys, deploy to Qualtrics/MTurk, collect simulated/human data, and generate papers. See “Terminal Usage” below.
  - API keys via `.env` (see “API Keys”). For simulation only, set `OPENAI_API_KEY`.

- Web UI: `python server.py`
  - Starts a local web app (URL shown in terminal) to process/enhance surveys, deploy, simulate responses, and debias via browser.

- Simulation only (quick demo): `python simulate_response/run_simulation.py`
  - Uses `simulate_response/participant_pool.csv`, `simulate_response/test_survey.json`, and `simulate_response/survey_response_template.txt` to produce `simulate_response/simulated_survey_responses.csv`.

---

## Prerequisites

* **Python 3.12+**
* A Qualtrics account with API token & data center information
* AWS credentials (if using MTurk) 
* Claude and OpenAI API accounts

---

## Key Directories & Files

```plaintext
config/                        # YAML configs for agents and tasks
  agents/
    survey_convert_agent.yaml  # Convert raw text → minimal JSON (cost-efficient agent)
    survey_editor.yaml         # Research enrichment + survey enhancement/editor
    econometrician_agent.yaml  # End-to-end paper agent (analysis/methodology/writing)
  tasks/
    convert_survey_to_json.yaml        # Conversion task template
    apply_survey_enhancements.yaml     # Research + improve survey task templates
    enhance_survey_iteratively.yaml    # Iterative enhancement task template (with placeholders)
    paper_tasks.yaml                   # Paper tasks (analysis, methodology, writing)
debias/                        # Debiasing pipeline module
knowledge/                     # Reference materials
simulate_response/             # Survey simulation scripts and templates
test_survey/                   # Sample survey JSON files
```

**Key files:**

```plaintext
survey.py                      # Final production code entry point
survey.html                    # Final HTML product
server.py                      # Backend server to run API calls
survey_logic.py                # Necessary logic from survey.py used for backend calls
requirements.txt               # Python dependencies list
README.md                      # Project overview and instructions
```

Agents
- survey_convert_agent.yaml: Converts raw text to minimal JSON using a cost-efficient model; avoids content rewriting.
- survey_editor.yaml: Enriches context via brief research and enhances/annotates surveys to meet academic standards.
- econometrician_agent.yaml: Executes analysis, methodology, and writing for research papers with journal-level rigor.

Tasks
- convert_survey_to_json.yaml: Converts raw text to a structured survey JSON schema.
- apply_survey_enhancements.yaml: Includes research_task (context enrichment) and improve_survey (annotated + revised survey).
- enhance_survey_iteratively.yaml: Iterative enhancement based on user feedback; outputs revised_survey and explanations.
- paper_tasks.yaml: Templates for analysis_task, methodology_task, and writing_task for end-to-end paper generation.

**Configuration Files:**

* **`config/agents/*.yaml`**: Defines roles and goals for each AI agent.
* **`config/tasks/*.yaml`**: Describes tasks for CrewAI (survey conversion, research, improvement).


---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Six-Persimmon/Field-Experiment-AI-Agent.git
   cd Field-Experiment-AI-Agent
   ```

2. **Create & activate a virtual environment**

   ```bash
   conda create -n venv python=3.12
   conda activate venv
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

# Terminal Usage

## API Keys

   Create a `.env` file in the root folder:

   ```
   # Qualtrics
   QUALTRICS_API_TOKEN=your_token_here
   QUALTRICS_DATA_CENTER=your_datacenter_id
   QUALTRICS_DIRECTORY_ID=your_directory_id

   # AWS (MTurk)
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_REGION=us-east-1
   MTURK_SANDBOX=True  # Set to True if you want to run a simulation; Set to False if you want to run a real experiment 

   #LLM Models
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_claude_key
   ```

## Running the Code

Run the following command in your terminal:

```bash
python survey.py
```

### Input File Formats

* **Copy and Pasting Your Original Survey Text**: Must include:

  ```text
  Topic: Your Survey Topic
  Purpose: Brief description
  Questions:
    - QID1: First question text...
    - QID2: Second question text...
  ```

* **Simulation**:

  * `participant_pool.csv`: CSV with simulated participant metadata. Eg:
  ```ParticipantID,Race,Gender,Age```
  * `test_survey.json`: Can be a default survey in JSON format or your newly created survey (please ensure that you save it as a JSON file)
  * `survey_response_template.txt`: A prompt template telling the LLM model to roleplay the specific demographics found in  `participant_pool.csv` for generating responses.

* **CSV for paper generation**: Any tabular data as long as it is in `.csv` file format; first row is the header.

### Menu Options

1. **Create and enhance a new survey**

   * Prompt: Enter raw survey text (must include `Topic:` and at least one `Questions:` line).
   * Result: Creates a survey in JSON format, allows AI and human enhancements, and deployable to Qualtrics and MTurk.

2. **Collect human data from existing survey**

   * Input: Qualtrics Survey ID and optional MTurk HIT ID.
   * Output: Formatted CSV with questions and responses.

3. **Collect simulated data from existing survey**

   * Input relative file paths to:

     * `survey_response_template.txt` (LLM template)
     * `test_survey.json` (survey content)
     * `participant_pool.csv` (participant metadata)
   * Output: JSON + CSV of simulated responses; optional debiasing.

4. **Generate research paper from CSV data**

   * Input: Path to CSV file.
   * Optional: Provide a research hypothesis.
   * Output: Markdown-formatted paper saved as `.md` file.

5. **Exit**

---

## Research Paper Agent

- An econometrician agent executes the full paper workflow end-to-end: analysis, methodology, and writing.
- Focuses on econometric rigor, clear exposition, appropriate visualization, and journal-ready structure aligned with top Economics (Top 5) and Management (UTD 24) venues.
- Tasks remain modular (analysis → methodology → writing) but are handled by a single, specialized agent for coherence and consistency.


# HTML Usage

## Running the Website

1. Run the following command in your terminal:

```bash
python server.py
```

2. You should see something of the following:
```
 * Serving Flask app 'server'
 * Debug mode: on
2025-07-11 20:09:13,297 - werkzeug - INFO - WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://{...}:5001
```

Please click the last link that says "Running on ..."

3. Now, the website will have opened up on your default browser for your usage.

### API Keys

   Enter the following information onto the website when prompted:

   ```
   # Qualtrics
   QUALTRICS_API_TOKEN=your_token_here
   QUALTRICS_DATA_CENTER=your_datacenter_id
   QUALTRICS_DIRECTORY_ID=your_directory_id

   # AWS (MTurk)
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_REGION=us-east-1
   MTURK_SANDBOX=True  # Set to True if you want to run a simulation; Set to False if you want to run a real experiment 

   #LLM Models
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_claude_key
   ```

### Input File Formats

* **Copy and Pasting Your Original Survey Text**: Must include:

  ```text
  Topic: Your Survey Topic
  Purpose: Brief description
  Questions:
    - QID1: First question text...
    - QID2: Second question text...
  ```

* **Simulation**:

  * `participant_pool.csv`: CSV with simulated participant metadata. Eg:
  ```ParticipantID,Race,Gender,Age```
  * `test_survey.json`: Can be a default survey in JSON format or your newly created survey (please ensure that you save it as a JSON file)
  * `survey_response_template.txt`: A prompt template telling the LLM model to roleplay the specific demographics found in  `participant_pool.csv` for generating responses.

* **CSV for paper generation**: Any tabular data as long as it is in `.csv` file format; first row is the header.

### Menu Options

1. **Create and enhance a new survey**

   * Prompt: Enter raw survey text (must include `Topic:` and at least one `Questions:` line).
   * Result: Creates a survey in JSON format, allows AI and human enhancements, and deployable to Qualtrics and MTurk.

2. **Collect human data from existing survey**

   * Input: Qualtrics Survey ID and optional MTurk HIT ID.
   * Output: Formatted CSV with questions and responses.

3. **Collect simulated data from existing survey**

   * Input relative file paths to:

     * `survey_response_template.txt` (LLM template)
     * `test_survey.json` (survey content)
     * `participant_pool.csv` (participant metadata)
   * Output: JSON + CSV of simulated responses; optional debiasing.

4. **Generate research paper from CSV data**

   * Input: Path to CSV file.
   * Optional: Provide a research hypothesis.
   * Output: Markdown-formatted paper saved as `.md` file.
