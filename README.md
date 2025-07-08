# Field-Experiment-AI-Agent

AI Agent framework for conducting consumer behavior experiments.

**Authors:**

* Xiao Liu (Professor, NYU) [xl23@stern.nyu.edu](mailto:xl23@stern.nyu.edu)
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

The system leverages Pydantic models, CrewAI agents, OpenAI, Qualtrics API, and MTurk API, offering a seamless end-to-end workflow for academic and market research.

---

## ⚙️ Prerequisites

* **Python 3.8+**
* A Qualtrics account with API token & data center information
* AWS credentials (if using MTurk) in `.env`
* Internet connection for API calls

---

## 📥 Installation

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd <repo-directory>
   ```

2. **Create & activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Environment variables**
   Create a `.env` file in the project root:

   ```ini
   # Qualtrics
   QUALTRICS_API_TOKEN=your_token_here
   QUALTRICS_DATA_CENTER=your_datacenter_id
   QUALTRICS_DIRECTORY_ID=your_directory_id

   # AWS (MTurk)
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_REGION=us-east-1
   ```

---

## 📂 Key Directories & Files

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

---

## 📝 Configuration Files

* **`config/agents/*.yaml`**: Defines roles and goals for each AI agent.
* **`config/tasks/*.yaml`**: Describes tasks for CrewAI (survey conversion, research, improvement).

---

## 🚀 Usage

Run the main menu for interactive workflows:

```bash
python version8.py
```

### Menu Options

1. **Create and enhance a new survey**

   * Prompt: Enter raw survey text (must include `Topic:` and at least one `Questions:` line).
   * Result: Enhanced JSON, summary printed, and interactive enhancement menu.

2. **Collect human data from existing survey**

   * Input: Qualtrics Survey ID and optional MTurk HIT ID.
   * Output: Formatted CSV with interleaved questions and responses.

3. **Collect simulated data from existing survey**

   * Prompt for paths to:

     * `survey_response_template.txt` (LLM template)
     * `test_survey.json` (survey context)
     * `participant_pool.csv` (participant metadata)
   * Output: JSON + CSV of simulated responses; optional debiasing.

4. **Generate research paper from CSV data**

   * Input: Path to CSV file.
   * Optional: Provide a research hypothesis.
   * Output: Markdown-formatted paper and saved `.md` file.

5. **Exit**

---

## 📁 Input File Formats

* **Raw survey text**: Must include:

  ```text
  Topic: Your Survey Topic
  Purpose: Brief description
  Questions:
    - QID1: First question text...
    - QID2: Second question text...
  ```

* **Simulation**:

  * `survey_response_template.txt`: A prompt template for generating responses.
  * `test_survey.json`: Survey JSON with top-level `questions` or nested under `revised_survey`.
  * `participant_pool.csv`: CSV with participant metadata.

* **CSV for paper generation**: Any tabular data; first row is header.

---

