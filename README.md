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

## 📖 Overview

This project provides an interactive, AI-powered survey enhancement and deployment system. First-time users can:

* Convert raw text surveys into structured JSON
* Iteratively enhance surveys with AI feedback
* Deploy surveys to Qualtrics and optionally create MTurk HITs
* Collect both human and simulated survey responses
* Generate research papers from CSV data

The system uses Pydantic models, CrewAI agents, OpenAI, Qualtrics API, and MTurk API, offering a seamless end-to-end workflow for academic and market research.

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
debias/                        # Debiasing pipeline module
field_experiment_ai_agent/     # Field experiment agent code
knowledge/                     # Reference materials
simulate_response/             # Survey simulation scripts and templates
test_survey/                   # Sample survey JSON files
```

**Key files:**

```plaintext
version8.py                    # Final production code entry point
requirements.txt               # Python dependencies list
enhanced.json                  # Latest enhanced survey output
research_paper_20250627_222338.md  # Generated research paper
simulated_survey_responses.json     # Raw simulation output
simulated_survey_responses_debiased.csv # Debiased simulation results
survey_SV_6nkUCABm1WWeUiq_detailed_formatted.csv # Formatted survey responses
README.md                      # Project overview and instructions
```

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

4. **Environment variables**
   Create a `.env` file in the root folder:

   ```ini
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

---


## Input File Formats

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

---


## Terminal Usage

Run the following command in your terminal:

```bash
python version8.py
```

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
