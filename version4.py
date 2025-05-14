import os
import yaml
import json
import warnings
import asyncio
import tempfile
import shutil
import copy
import logging
from datetime import datetime
from typing import Literal, Dict, List, Any, Union, Optional
from pydantic import BaseModel, ValidationError, Field, field_validator
from crewai import Agent, Task, Crew, Process
from crewai.tasks.task_output import OutputFormat

import requests
import zipfile
import io
import time
import pandas as pd
import boto3
from dotenv import load_dotenv
import logging
import re
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ========== Enhanced Pydantic Models with Validation ==========
class ChoiceOption(BaseModel):
    text: str
    value: str

class ChoiceConfig(BaseModel):
    options: List[str] = Field(default_factory=list)
    
    @field_validator('options')
    @classmethod
    def validate_options(cls, v):
        if not v:
            raise ValueError("Options cannot be empty for a choice-based question")
        return v

class SliderConfig(BaseModel):
    min: float = 0
    max: float = 100
    step: float = 1
    
    @field_validator('max')
    @classmethod
    def validate_max(cls, v, info):
        values = info.data
        if 'min' in values and v <= values['min']:
            raise ValueError("Max value must be greater than min value")
        return v
    
    @field_validator('step')
    @classmethod
    def validate_step(cls, v):
        if v <= 0:
            raise ValueError("Step value must be positive")
        return v

class TextInputConfig(BaseModel):
    placeholder: Optional[str] = None
    multiline: bool = False

class Question(BaseModel):
    question_id: str
    question_text: str
    input_type: Literal["multiple_choice", "single_choice", "slider", "text_input"]
    input_config: Dict[str, Any]
    
    @field_validator('question_text')
    @classmethod
    def validate_question_text(cls, v):
        if len(v.strip()) < 5:
            raise ValueError("Question text is too short")
        return v
    
    @field_validator('input_config')
    @classmethod
    def validate_input_config(cls, v, info):
        values = info.data
        if 'input_type' in values:
            input_type = values['input_type']
            if input_type in ['multiple_choice', 'single_choice']:
                if 'options' not in v or not v['options']:
                    raise ValueError(f"{input_type} must have options defined")
            elif input_type == 'slider':
                required_keys = ['min', 'max']
                for key in required_keys:
                    if key not in v:
                        raise ValueError(f"Slider configuration missing required key: {key}")
        return v

class Survey(BaseModel):
    theme: str
    purpose: str
    questions: List[Question]
    
    @field_validator('questions')
    @classmethod
    def validate_questions(cls, v):
        if not v:
            raise ValueError("Survey must have at least one question")
        # Ensure question_ids are unique
        ids = [q.question_id for q in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Question IDs must be unique")
        return v

class QuestionComment(BaseModel):
    question_id: str
    comment: str

class AnnotatedSurvey(BaseModel):
    survey: Survey
    question_comments: List[QuestionComment]
    overall_comment: Optional[str] = None

class SurveyImprovementResult(BaseModel):
    original_with_comments: AnnotatedSurvey
    revised_survey: Survey

# New model specifically for validating the conversion agent's output
class SurveyConversionOutput(BaseModel):
    """Model to validate the output of the survey conversion agent"""
    title: str
    fields: List[Dict[str, Any]]
    
    @field_validator('fields')
    @classmethod
    def validate_fields(cls, v):
        if not v:
            raise ValueError("Survey must have at least one field")
        
        for i, field in enumerate(v):
            if 'title' not in field:
                raise ValueError(f"Field {i+1} is missing 'title'")
            if 'type' not in field:
                raise ValueError(f"Field {i+1} is missing 'type'")
            
            if field['type'] in ['multiple_choice', 'single_choice']:
                if 'options' not in field or not field['options']:
                    raise ValueError(f"Field {i+1} ({field.get('title', 'Untitled')}) is missing options for {field['type']}")
        
        return v

# Enhanced validation function with detailed error reporting
def validate_conversion_output(raw_output: str) -> Dict:
    """
    Validates the output of the survey conversion agent using Pydantic with enhanced error reporting.
    
    Args:
        raw_output: The raw JSON string output from the agent
        
    Returns:
        The validated dictionary if successful
        
    Raises:
        ValidationError: If the output doesn't match the expected schema
        ValueError: If the output cannot be parsed as JSON
    """
    # Clean up the raw output - strip markdown code blocks if present
    cleaned_output = raw_output.strip()
    if cleaned_output.startswith("```json"):
        cleaned_output = cleaned_output.split("```json", 1)[1]
    elif cleaned_output.startswith("```"):
        cleaned_output = cleaned_output.split("```", 1)[1]
    
    if "```" in cleaned_output:
        cleaned_output = cleaned_output.rsplit("```", 1)[0]
    
    cleaned_output = cleaned_output.strip()
    
    try:
        # Parse the JSON
        parsed_dict = json.loads(cleaned_output)
        
        # Log the structure for debugging
        logger.debug(f"Parsed JSON structure: {json.dumps(parsed_dict, indent=2)}")
        
        # Validate using Pydantic model
        validated = SurveyConversionOutput(**parsed_dict)
        
        # Return the validated dict
        return validated.dict()
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse JSON output: {e}\nRaw output:\n{raw_output}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    except ValidationError as e:
        error_msg = f"Output validation failed: {e}\nRaw output:\n{raw_output}"
        logger.error(error_msg)
        raise ValidationError(e.errors(), SurveyConversionOutput)

def convert_to_question_format(conversion_output: Dict) -> List[Question]:
    """
    Converts the validated conversion output to a list of Question objects.
    
    Args:
        conversion_output: The validated conversion output
        
    Returns:
        List of Question objects
    """
    questions = []
    for i, field in enumerate(conversion_output["fields"]):
        question_id = f"q{i+1}"
        question_text = field["title"]
        
        # Determine input type and config
        if field["type"] == "multiple_choice":
            input_type = "multiple_choice"
            input_config = {"options": field.get("options", [])}
        elif field["type"] == "text_input":
            input_type = "text_input"
            input_config = {"multiline": False}
        elif field["type"] == "slider":
            input_type = "slider"
            input_config = {"min": 0, "max": 100, "step": 1}  # Default values
        else:
            # Default to single_choice for most survey questions with scales
            input_type = "single_choice"
            input_config = {"options": field.get("options", [])}
        
        # Create Question object
        try:
            question = Question(
                question_id=question_id,
                question_text=question_text,
                input_type=input_type,
                input_config=input_config
            )
            
            questions.append(question)
        except ValidationError as e:
            logger.warning(f"Question validation failed for question {i+1}: {e}")
            # Create a minimal valid question as a fallback
            fallback_question = Question(
                question_id=question_id,
                question_text=question_text if len(question_text) >= 5 else f"Question {i+1}",
                input_type="multiple_choice" if "options" in field else "text_input",
                input_config={"options": field.get("options", ["Yes", "No"])} if "options" in field else {"multiline": False}
            )
            questions.append(fallback_question)
    
    return questions

# ========== Interactive Survey Enhancement Flow ==========
# ========== Interactive Survey Enhancement Flow ==========
class SurveyEnhancementFlow:
    """Interactive flow for enhancing surveys with user feedback"""
    
    def __init__(self):
        """Initialize the enhancement flow"""
        self.convert_agent, self.editor_agent, self.enhancement_agent = self._load_agents()
        self.convert_task, self.research_task, self.improve_task, self.enhancement_task = self._load_tasks(
            self.convert_agent, self.editor_agent, self.enhancement_agent
        )
        self.survey_dict = None
        self.enhanced_dict = None
        self.temp_file = None
    
    def _load_agents(self):
        """Load the necessary agents for the survey enhancement flow"""
        # Load the conversion agent
        conv_cfg = self._load_yaml("config/agents/survey_convert_agent.yaml")["survey_convert_agent"]
        convert_agent = Agent(
            name="Survey Content Conversion Agent",
            role=conv_cfg["role"],
            goal=conv_cfg["goal"],
            backstory=conv_cfg["backstory"],
            verbose=conv_cfg["verbose"],
            allow_delegation=conv_cfg["allow_delegation"]
        )

        # Load the editor agent
        edit_cfg = self._load_yaml("config/agents/survey_editor.yaml")["survey_editor"]
        editor_agent = Agent(
            name="Academic Survey Designer",
            role=edit_cfg["role"],
            goal=edit_cfg["goal"],
            backstory=edit_cfg["backstory"],
            verbose=edit_cfg["verbose"],
            allow_delegation=edit_cfg["allow_delegation"]
        )
        
        # Create new enhancement agent for interactive feedback
        enhancement_agent = Agent(
            name="Survey Enhancement Agent",
            role="Interactive survey enhancer who refines surveys based on user feedback",
            goal="To improve survey design by incorporating user feedback in an iterative process",
            backstory="I am an AI assistant specialized in survey design and enhancement. I work iteratively with users to refine surveys until they perfectly match the user's needs and standards. I'm trained in best practices for survey design, question construction, and cognitive psychology to ensure surveys are effective, unbiased, and generate valuable data.",
            verbose=True,
            allow_delegation=False
        )
        
        return convert_agent, editor_agent, enhancement_agent
    
    def _load_yaml(self, path: str) -> dict:
        """Load YAML configuration file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # If the file doesn't exist, return an empty dict with expected structure
            if 'survey_convert_agent' in path:
                return {"survey_convert_agent": {
                    "role": "Survey conversion specialist",
                    "goal": "Convert text survey into structured JSON",
                    "backstory": "Expert in survey methodology and format conversion",
                    "verbose": True,
                    "allow_delegation": False
                }}
            elif 'survey_editor' in path:
                return {"survey_editor": {
                    "role": "Academic survey editor",
                    "goal": "Enhance survey quality and academic rigor",
                    "backstory": "Experienced researcher with expertise in survey methodology",
                    "verbose": True,
                    "allow_delegation": False
                }}
            else:
                return {}
    
    def _load_tasks(self, convert_agent, editor_agent, enhancement_agent):
        """Load the tasks for the survey enhancement flow"""
        # Load convert task
        conv_t = self._load_yaml("config/tasks/convert_survey_to_json.yaml").get("convert_survey_to_json", {})
        convert_task = Task(
            name="convert_survey_to_json",
            description=conv_t.get("description", "Convert the following survey (provided as raw text) into a structured JSON schema suitable for creating a survey in Qualtrics or similar platforms."),
            agent=convert_agent,
            tool=conv_t.get("tool"),
            expected_output=conv_t.get("expected_output", "A JSON object representing the survey"),
            output_format=OutputFormat.JSON,
            async_execution=True,
            validation_function=validate_conversion_output  # Add validation function
        )

        # Load research task 
        res_t = self._load_yaml("config/tasks/apply_survey_enhancements.yaml").get("research_task", {})
        # Remove placeholders by replacing them with actual values or generic text
        description = res_t.get("description", "Conduct a thorough research about the survey topic Make sure you find any interesting and relevant information given the current year is {current_year}.")
        description = description.replace("{topic}", "the survey topic").replace("{current_year}", str(datetime.now().year))
        
        expected_output = res_t.get("expected_output", "A bullet-point list of relevant information about {topic}.")
        expected_output = expected_output.replace("{topic}", "the survey topic")
        
        research_task = Task(
            name="research_task",
            description=description,
            agent=convert_agent,
            expected_output=expected_output,
            output_format=OutputFormat.JSON
        )

        # Load improve task
        imp_t = self._load_yaml("config/tasks/apply_survey_enhancements.yaml").get("improve_survey", {})
        default_description = """You will receive:
  - original_survey: the survey as a JSON object conforming to the Survey model  
  - comments: an array of feedback comments for each question  
  - suggestions: an array of proposed improvements for each question and for the survey as a whole

Your tasks:
  1. Produce `original_with_comments`: annotate the original_survey with comments only (do not include suggestions in this part).  
  2. Produce revised_survey: apply the suggestions to generate an updated survey JSON, ensuring that each question is rephrased to align with professional academic standards.
  3. Respond with exactly this JSON schema (no extra keys, no markdown fences):

  {{
    "original_with_comments": {{
      "survey": {{
        "theme": "<survey theme>",
        "purpose": "<survey purpose>",
        "questions": [
          {{
            "question_id": "<question_id>",
            "question_text": "<question text>",
            "input_type": "<multiple_choice|single_choice|text_input>",
            "input_config": {{ … }}
          }}
        ]
      }},
      "question_comments": [
        {{ "question_id": "<question_id>", "comment": "<comment>" }}
      ],
      "overall_comment": "<overall comment>"
    }},
    "revised_survey": {{
      "theme": "<survey theme>",
      "purpose": "<survey purpose>",
      "questions": [
        {{
          "question_id": "<question_id>",
          "question_text": "<question text>",
          "input_type": "<multiple_choice|single_choice|text_input>",
          "input_config": {{ … }}
        }}
      ]
    }}
  }}"""
        
        # Handle the JSON schema example carefully
        description = imp_t.get("description", default_description)
        description = description.replace("{", "{{").replace("}", "}}")
        
        expected_output = imp_t.get("expected_output", "A JSON object with original_with_comments and revised_survey")
        expected_output = expected_output.replace("{", "{{").replace("}", "}}")
        
        improve_task = Task(
            name="improve_survey",
            description=description,
            agent=editor_agent,
            expected_output=expected_output,
            output_format=OutputFormat.JSON
        )
        
        # Create enhanced enhancement task with stricter instructions
        enhancement_task = Task(
            name="enhance_survey_iteratively",
            description=(
                "Review and enhance the EXACT survey provided in the 'original_survey' field. "
                "DO NOT generate a new survey from scratch or use a default template. "
                "IMPORTANT: Your task is to improve the specific questions and structure of THIS EXACT PROVIDED SURVEY "
                "based on the user's feedback. Make specific modifications to improve question clarity, "
                "reduce bias, and align with best practices in survey methodology. "
                "Maintain the same general topic and purpose of the survey. "
                "Provide detailed explanations of changes made in an 'explanations' section."
            ),
            agent=enhancement_agent,
            expected_output=(
                "A JSON object containing the enhanced survey with the SAME STRUCTURE as the input but with "
                "improvements based on feedback. Include explanations for each change."
            ),
            output_format=OutputFormat.JSON
        )
        
        return convert_task, research_task, improve_task, enhancement_task
        
    def run(self, survey_text):
        """
        Run the initial survey conversion and enhancement flow
        
        Args:
            survey_text (str): Raw survey text input
            
        Returns:
            dict: Enhanced survey dictionary
        """
        print("Starting survey conversion and enhancement flow...")
        
        # Extract basic info
        first_line = survey_text.splitlines()[0]
        topic = first_line.replace('Topic:', '').strip()
        current_year = datetime.now().year
        
        # Create task input dictionary
        task_inputs = {
            'survey_text': survey_text,
            'topic': topic,
            'current_year': current_year
        }
        
        # Create and run the initial crew
        initial_crew = Crew(
            agents=[self.convert_agent, self.editor_agent],
            tasks=[self.convert_task, self.research_task, self.improve_task],
            process=Process.sequential,
            verbose=True
        )
        
        # Run the crew to process the survey
        print("\n=== Running Initial Survey Processing ===")
        crew_result = initial_crew.kickoff(inputs=task_inputs)
        
        # Parse the result
        raw = crew_result.raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        if raw.startswith("json"):
            raw = raw[4:].strip()
        
        try:
            self.survey_dict = json.loads(raw)
            
            # Check if we have a valid survey structure
            self._validate_survey_structure(self.survey_dict)
            
            self.enhanced_dict = self.survey_dict  # Start with the initial output
            
            # Print summary of the initial result
            self._print_survey_summary(self.survey_dict)
            
            return self.survey_dict
            
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parsing failed: {e}\nRaw output:\n{raw}")
        except ValueError as e:
            raise ValueError(f"Survey validation failed: {e}\nRaw output:\n{raw}")
    
    def _validate_survey_structure(self, survey_dict):
        """
        Validates that the survey has a proper structure
        
        Args:
            survey_dict: The survey dictionary to validate
            
        Raises:
            ValueError: If the survey structure is invalid
        """
        # Check if we have a revised_survey key
        if 'revised_survey' not in survey_dict:
            raise ValueError("Missing 'revised_survey' key in survey dictionary")
        
        revised = survey_dict['revised_survey']
        
        # Check if revised survey has questions
        if 'questions' not in revised or not isinstance(revised['questions'], list):
            raise ValueError("Missing or invalid 'questions' array in revised survey")
        
        # Check if we have at least one question
        if len(revised['questions']) == 0:
            raise ValueError("Survey must have at least one question")
        
        # All good!
        return True
    
    def _process_enhancement_result(self, result_raw, original_survey_dict):
        """
        Process and standardize the enhancement result from various possible formats
        
        Args:
            result_raw (str): Raw JSON string from enhancement agent
            original_survey_dict (dict): The original survey dictionary before enhancement
            
        Returns:
            dict: Standardized survey dictionary
        """
        try:
            enhanced_result = json.loads(result_raw)
            
            # If enhanced result looks like it might be a default survey rather than an enhancement
            # of the original, we'll flag it and try to fix it
            if self._is_default_survey(enhanced_result, original_survey_dict):
                logger.warning("Enhancement agent may have returned a default survey instead of enhancing the provided one")
                print("\nWarning: The enhancement appears to have used a default survey template rather than your inputted content.")
                print("Attempting to apply enhancements to your original survey content...\n")
                
                # Try to apply any explanations or improvement patterns to the original survey
                return self._merge_enhancements_with_original(enhanced_result, original_survey_dict)
            
            # The enhancement agent might return different structures
            # Case 1: Simple 'survey' key at root (most common from enhancement agent)
            if 'survey' in enhanced_result:
                survey_data = enhanced_result['survey']
                
                # Check if we need to convert the format
                if 'questions' in survey_data and isinstance(survey_data['questions'], list):
                    questions = []
                    
                    # Convert questions to expected format
                    for i, q in enumerate(survey_data['questions']):
                        # Handle different field naming conventions
                        question_id = q.get('question_id', q.get('id', f'q{i+1}'))
                        question_text = q.get('question_text', q.get('question', q.get('text', '')))
                        
                        # Determine input type
                        input_type = q.get('input_type', q.get('type', ''))
                        # Map common types to our expected types
                        if input_type in ['open_ended', 'open_text', 'text']:
                            input_type = 'text_input'
                        elif input_type in ['multiple_choice', 'single_choice']:
                            input_type = input_type
                        elif input_type == 'scale':
                            input_type = 'slider'
                        else:
                            # Default to multiple_choice for unknown types
                            input_type = 'multiple_choice'
                        
                        # Handle different options formats
                        input_config = {}
                        options = q.get('options', [])
                        if options and isinstance(options, list):
                            # If options are objects with value/label
                            if options and isinstance(options[0], dict) and ('label' in options[0] or 'text' in options[0]):
                                input_config['options'] = [opt.get('label', opt.get('text', '')) for opt in options]
                            else:
                                input_config['options'] = options
                        
                        # Handle slider/scale configuration
                        if input_type == 'slider' and ('scale_min' in q or 'scale_max' in q):
                            input_config['min'] = q.get('scale_min', 1)
                            input_config['max'] = q.get('scale_max', 10)
                            input_config['step'] = q.get('scale_step', 1)
                        
                        # Create standardized question
                        question = {
                            'question_id': str(question_id),
                            'question_text': question_text,
                            'input_type': input_type,
                            'input_config': input_config
                        }
                        
                        questions.append(question)
                    
                    # Create standardized survey structure
                    return {
                        'revised_survey': {
                            'theme': survey_data.get('title', survey_data.get('theme', 'Enhanced Survey')),
                            'purpose': survey_data.get('purpose', survey_data.get('description', '')),
                            'questions': questions
                        },
                        'explanations': enhanced_result.get('explanations', {})
                    }
                else:
                    # It's already in our expected format
                    return {'revised_survey': survey_data}
            
            # Case 2: 'revised_survey' key already exists
            elif 'revised_survey' in enhanced_result:
                return enhanced_result
            
            # Case 3: It's a completely different format, try to adapt it
            else:
                # Check if it looks like a survey with questions
                if 'questions' in enhanced_result and isinstance(enhanced_result['questions'], list):
                    return {'revised_survey': enhanced_result}
                else:
                    # Last resort - wrap the whole thing as revised_survey
                    return {'revised_survey': enhanced_result}
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse enhancement result: {e}")
            raise ValueError(f"Invalid JSON in enhancement result: {e}")
        except Exception as e:
            logger.error(f"Error processing enhancement result: {e}")
            raise ValueError(f"Error processing enhancement result: {e}")
    
    def _is_default_survey(self, enhanced_result, original_survey_dict):
        """
        Check if the enhanced result looks like a default survey rather than an enhancement of the original
        
        Args:
            enhanced_result (dict): The enhanced survey dictionary
            original_survey_dict (dict): The original survey dictionary
            
        Returns:
            bool: True if it appears to be a default survey
        """
        # Extract key elements from both original and enhanced surveys
        original_theme = None
        original_questions = []
        enhanced_theme = None
        enhanced_questions = []
        
        # Extract from original survey
        if 'revised_survey' in original_survey_dict:
            original_theme = original_survey_dict['revised_survey'].get('theme', '')
            original_questions = original_survey_dict['revised_survey'].get('questions', [])
        elif 'survey' in original_survey_dict:
            original_theme = original_survey_dict['survey'].get('theme', original_survey_dict['survey'].get('title', ''))
            original_questions = original_survey_dict['survey'].get('questions', [])
        
        # Extract from enhanced survey
        if 'survey' in enhanced_result:
            enhanced_theme = enhanced_result['survey'].get('theme', enhanced_result['survey'].get('title', ''))
            enhanced_questions = enhanced_result['survey'].get('questions', [])
        elif 'revised_survey' in enhanced_result:
            enhanced_theme = enhanced_result['revised_survey'].get('theme', '')
            enhanced_questions = enhanced_result['revised_survey'].get('questions', [])
        
        # If themes are completely different, that's suspicious
        if original_theme and enhanced_theme and original_theme.lower() != enhanced_theme.lower():
            if not (original_theme.lower() in enhanced_theme.lower() or enhanced_theme.lower() in original_theme.lower()):
                return True
        
        # If the number of questions changed dramatically, that's suspicious
        if len(original_questions) > 0 and len(enhanced_questions) > 0:
            # If question count differs by more than 50%, it might be a default survey
            if len(enhanced_questions) < len(original_questions) * 0.5 or len(enhanced_questions) > len(original_questions) * 1.5:
                return True
        
        # If original had questions but enhanced doesn't, that's suspicious
        if len(original_questions) > 0 and len(enhanced_questions) == 0:
            return True
            
        # If none of our checks were triggered, it's probably a legitimate enhancement
        return False
    
    def _merge_enhancements_with_original(self, enhanced_result, original_survey_dict):
        """
        Attempt to merge any valid enhancements with the original survey
        
        Args:
            enhanced_result (dict): The potentially default-based enhancement result
            original_survey_dict (dict): The original survey dictionary
            
        Returns:
            dict: The merged survey dictionary
        """
        # Start with a copy of the original survey
        merged_result = copy.deepcopy(original_survey_dict)
        
        # Extract any explanations to preserve them
        explanations = {}
        if 'explanations' in enhanced_result:
            explanations = enhanced_result['explanations']
        
        # Add the explanations to the merged result
        merged_result['explanations'] = explanations
        
        # Return the merged result with original structure plus any explanations
        return merged_result
            
    def interactive_enhancement(self):
        """
        Interactive enhancement loop that allows users to review, manually modify, 
        and request AI enhancements to the survey
        
        Returns:
            dict: Final enhanced survey dictionary
        """
        if not self.survey_dict:
            raise ValueError("No survey has been processed yet. Run the 'run' method first.")
        
        self.enhanced_dict = self.survey_dict
        iteration = 1
        
        while True:
            print("\n" + "="*50)
            print(f"ENHANCEMENT ITERATION #{iteration}")
            print("="*50)
            
            # Display options
            print("\nWhat would you like to do?")
            print("1. Review current survey")
            print("2. Request AI enhancement with feedback")
            print("3. Manually edit survey JSON")
            print("4. Save survey to file")
            print("5. Deploy survey to Qualtrics")
            print("6. Finish and exit")
            
            choice = input("\nEnter your choice (1-6): ")
            
            if choice == "1":
                # Review current survey
                self._print_survey_summary(self.enhanced_dict)
                
            elif choice == "2":
                # Get user feedback
                print("\nPlease provide your feedback/suggestions for the survey (enter a blank line to finish):")
                feedback_lines = []
                while True:
                    line = input()
                    if not line:
                        break
                    feedback_lines.append(line)
                
                user_feedback = "\n".join(feedback_lines)
                
                if not user_feedback:
                    print("No feedback provided. Returning to menu.")
                    continue
                
                # Prepare the enhancement task with additional context
                # Create a more specific directive to avoid default survey generation
                enhanced_task_description = (
                    "Review and enhance the EXACT survey provided in the input. "
                    "DO NOT generate a default or generic survey. "
                    "Your task is to improve the specific questions and structure of the provided survey "
                    "based on the user's feedback. Make specific modifications to improve question clarity, "
                    "reduce bias, and align with best practices in survey methodology. "
                    "Provide detailed explanations of changes made."
                )
                
                enhancement_task = Task(
                    name="enhance_survey_iteratively",
                    description=enhanced_task_description,
                    agent=self.enhancement_agent,
                    expected_output="A JSON object containing the enhanced survey and explanations of changes.",
                    output_format=OutputFormat.JSON
                )
                
                # Run the enhancement task with the feedback
                enhancement_crew = Crew(
                    agents=[self.enhancement_agent],
                    tasks=[enhancement_task],
                    process=Process.sequential,
                    verbose=True
                )
                
                # Create a clearer input with explicit structure to help guide the enhancement
                # Include the original structure and format to preserve
                survey_json = json.dumps(self.enhanced_dict, indent=2)
                enhancement_input = {
                    'original_survey': survey_json,
                    'survey_structure': f"The survey has the following structure:\nTheme: {self.enhanced_dict.get('revised_survey', {}).get('theme', 'Unknown')}\nNumber of questions: {len(self.enhanced_dict.get('revised_survey', {}).get('questions', []))}",
                    'user_feedback': user_feedback,
                    'instruction': "Please enhance this EXACT survey. Do not replace it with a default survey template. Preserve the general structure and theme, but improve the questions based on the feedback provided."
                }
                
                # Run enhancement
                print("\n=== Running AI Enhancement ===")
                enhancement_result = enhancement_crew.kickoff(inputs=enhancement_input)
                
                # Process the result
                result_raw = enhancement_result.raw.strip()
                if result_raw.startswith("```"):
                    result_raw = result_raw.split("\n", 1)[1].rsplit("```", 1)[0]
                if result_raw.startswith("json"):
                    result_raw = result_raw[4:].strip()
                
                try:
                    # Process the enhancement result using our helper method, passing original survey
                    enhanced_result = self._process_enhancement_result(result_raw, self.enhanced_dict)
                    self.enhanced_dict = enhanced_result
                    
                    print("\n=== Enhancement Complete ===")
                    self._print_survey_summary(self.enhanced_dict)
                    
                except json.JSONDecodeError as e:
                    print(f"Error processing enhancement result: {e}")
                    print("The original survey will be retained.")
                except ValueError as e:
                    print(f"Error: {e}")
                    print("The original survey will be retained.")
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    print("The original survey will be retained.")
                
            elif choice == "3":
                # Manually edit survey JSON
                self._manual_edit()
                
            elif choice == "4":
                # Save to file
                filename = input("Enter filename to save (default: enhanced_survey.json): ") or "enhanced_survey.json"
                with open(filename, 'w') as f:
                    json.dump(self.enhanced_dict, f, indent=2)
                print(f"Survey saved to {filename}")
                
            elif choice == "5":
                # Deploy to Qualtrics
                return self._deploy_to_qualtrics()
                
            elif choice == "6":
                # Finish and exit
                print("Enhancement process complete.")
                return self.enhanced_dict
                
            else:
                print("Invalid choice. Please enter a number from 1 to 6.")
            
            iteration += 1
    
    def _manual_edit(self):
        """Allow manual editing of the survey JSON with improved editor detection"""
        # Create a temporary file with the JSON
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as tf:
            json.dump(self.enhanced_dict, tf, indent=2)
            temp_filename = tf.name
        
        # Try to detect the best editor for the current environment
        editor = None
        
        # Check environment variable first
        if 'EDITOR' in os.environ:
            editor = os.environ['EDITOR']
        elif 'VISUAL' in os.environ:
            editor = os.environ['VISUAL']
        else:
            # Try to detect common editors
            if os.name == 'nt':  # Windows
                if shutil.which('notepad++'):
                    editor = 'notepad++'
                elif shutil.which('code'):
                    editor = 'code'
                else:
                    editor = 'notepad'
            else:  # Unix-like
                if shutil.which('nano'):
                    editor = 'nano'
                elif shutil.which('vim'):
                    editor = 'vim'
                elif shutil.which('vi'):
                    editor = 'vi'
                elif shutil.which('code'):
                    editor = 'code'
                elif shutil.which('gedit'):
                    editor = 'gedit'
                else:
                    editor = 'vi'  # Default to vi on Unix systems
        
        print(f"\nOpening temporary file {temp_filename} with {editor}...")
        print("Edit the JSON file, save, and close the editor to continue.")
        
        try:
            # Open the file with the editor
            if editor == 'code':  # VS Code needs special handling to wait
                subprocess.run([editor, '--wait', temp_filename])
            else:
                os.system(f'{editor} "{temp_filename}"')
            
            # Read the modified file
            with open(temp_filename, 'r') as f:
                modified_json = f.read()
            
            # Parse and validate the JSON
            try:
                modified_dict = json.loads(modified_json)
                
                # Validate the structure
                if 'revised_survey' in modified_dict:
                    revised = modified_dict['revised_survey']
                    # Basic validation of required fields
                    if not isinstance(revised, dict):
                        raise ValueError("revised_survey must be a dictionary")
                    if 'theme' not in revised:
                        raise ValueError("Survey must have a 'theme' field")
                    if 'questions' not in revised or not isinstance(revised['questions'], list):
                        raise ValueError("Survey must have a 'questions' array")
                
                self.enhanced_dict = modified_dict
                print("Successfully updated survey from your edits.")
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON. Your edits were not applied: {e}")
                print("The original survey will be retained.")
                
                # Provide the option to try again
                retry = input("Would you like to try editing again? (y/n): ").lower()
                if retry == 'y':
                    self._manual_edit()
            except ValueError as e:
                print(f"Error: {e}")
                print("The original survey will be retained.")
                
                # Provide the option to try again
                retry = input("Would you like to try editing again? (y/n): ").lower()
                if retry == 'y':
                    self._manual_edit()
        
        except Exception as e:
            print(f"Error during manual editing: {e}")
        
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_filename)
            except:
                pass
    
    def _deploy_to_qualtrics(self):
        """Deploy the survey to Qualtrics and optionally MTurk"""
        print("\n=== Deploying Survey to Qualtrics ===")
        
        # Check if QualtricsMTurkAutomation class exists
        if 'QualtricsAndMTurkAutomation' not in globals() and not hasattr(self, 'qualtrics_client'):
            print("Qualtrics deployment functionality not available.")
            print("Please ensure the QualtricsAndMTurkAutomation class is defined or implement it.")
            
            # Ask if user wants to just save the file instead
            save_option = input("Would you like to save the survey to a file instead? (y/n): ").lower()
            if save_option == 'y':
                filename = input("Enter filename to save (default: enhanced_survey.json): ") or "enhanced_survey.json"
                with open(filename, 'w') as f:
                    json.dump(self.enhanced_dict, f, indent=2)
                print(f"Survey saved to {filename}")
            
            return self.enhanced_dict
        
        # Convert to Qualtrics format
        try:
            qualtrics_payload = survey_dict_to_qualtrics_payload(self.enhanced_dict)
        except NameError:
            print("Function survey_dict_to_qualtrics_payload not found.")
            print("Please ensure the function is defined before attempting deployment.")
            return self.enhanced_dict
        
        # Ask about MTurk
        use_mturk = input("Would you like to also create an MTurk HIT? (y/n): ").lower() == 'y'
        
        # Create HIT configuration if MTurk is requested
        hit_config = None
        if use_mturk:
            survey_meta = self.enhanced_dict.get("revised_survey", {})
            hit_config = {
                'Title': f'Complete a survey on {survey_meta.get("theme", "research topic")}',
                'Description': survey_meta.get("purpose", "Complete a short research survey"),
                'Keywords': 'survey, research, feedback',
                'Reward': input("Enter reward amount (default: 0.75): ") or '0.75',
                'MaxAssignments': int(input("Enter max participants (default: 100): ") or '100'),
                'LifetimeInSeconds': 86400,
                'AssignmentDurationInSeconds': 1800,
                'AutoApprovalDelayInSeconds': 86400,
                'QualificationRequirements': []
            }
        
        # Run the automation
        try:
            if 'QualtricsAndMTurkAutomation' in globals():
                automation = QualtricsAndMTurkAutomation()
                
                if use_mturk:
                    results = automation.run(qualtrics_payload, hit_config)
                    print("\nDeployment Results:")
                    print(f"Survey ID: {results['survey_id']}")
                    print(f"Survey Link: {results['survey_link']}")
                    print(f"HIT ID: {results['hit_id']}")
                    return results
                else:
                    # Just create Qualtrics survey without MTurk
                    survey_name = qualtrics_payload.get("SurveyName", "New Survey")
                    survey_id = automation.qualtrics_client.create_survey(survey_name, qualtrics_payload)
                    automation.qualtrics_client.activate_survey(survey_id)
                    survey_link = automation.qualtrics_client.create_distribution_link(survey_id)
                    
                    print("\nDeployment Results:")
                    print(f"Survey ID: {survey_id}")
                    print(f"Survey Link: {survey_link}")
                    
                    return {
                        "survey_id": survey_id,
                        "survey_link": survey_link
                    }
            else:
                print("QualtricsAndMTurkAutomation class not found.")
                print("Please implement it or ensure it's defined in the global scope.")
                return self.enhanced_dict
                
        except Exception as e:
            print(f"Error during deployment: {e}")
            return None
    
    def _print_survey_summary(self, survey_dict):
        """Print a summary of the survey in a readable format"""
        print("\n=== SURVEY SUMMARY ===")
        
        try:
            # Try to get revised survey
            if 'revised_survey' in survey_dict:
                revised = survey_dict.get('revised_survey', {})
                
                # Print theme and purpose safely
                print(f"Theme:   {revised.get('theme', 'N/A')}")
                print(f"Purpose: {revised.get('purpose', 'N/A')}\n")
                
                # Safely iterate through questions
                for q in revised.get('questions', []):
                    if isinstance(q, dict):
                        qid = q.get('question_id', q.get('id', 'unknown'))
                        qtext = q.get('question_text', q.get('question', q.get('text', 'N/A')))
                        print(f"Q{qid}: {qtext}")
                        
                        # Safely get options
                        input_config = q.get('input_config', {})
                        if isinstance(input_config, dict):
                            opts = input_config.get('options', [])
                            if opts:
                                print("  Options:")
                                for o in opts:
                                    print(f"    - {o}")
                            elif 'min' in input_config and 'max' in input_config:
                                # Print slider info
                                print(f"  Slider: {input_config.get('min', 0)} to {input_config.get('max', 100)}, " +
                                      f"step: {input_config.get('step', 1)}")
                            elif input_config.get('multiline') is not None:
                                # Print text input info
                                multiline = "multiline" if input_config.get('multiline') else "single line"
                                print(f"  Text input: {multiline}")
                        
                        # Handle direct options in question
                        elif 'options' in q and isinstance(q['options'], list):
                            opts = q['options']
                            print("  Options:")
                            # Handle both simple strings and objects with label/value
                            for o in opts:
                                if isinstance(o, dict) and ('label' in o or 'text' in o):
                                    print(f"    - {o.get('label', o.get('text', ''))}")
                                else:
                                    print(f"    - {o}")
                        print()
            
            # Check for explanations in the result
            if 'explanations' in survey_dict:
                print("=== Explanations ===")
                explanations = survey_dict['explanations']
                if isinstance(explanations, dict):
                    for key, explanation in explanations.items():
                        print(f"{key}: {explanation}")
                print()
            
            # If original_with_comments exists, print it with annotations
            elif 'original_with_comments' in survey_dict:
                annotated = survey_dict.get('original_with_comments', {})
                survey = annotated.get('survey', {})
                
                print("=== Original Survey (with comments) ===")
                print(f"Theme: {survey.get('theme', 'N/A')}")
                print(f"Purpose: {survey.get('purpose', 'N/A')}\n")
                
                # Safely iterate through questions with comments
                comments = annotated.get('question_comments', [])
                for q in survey.get('questions', []):
                    if isinstance(q, dict):
                        qid = q.get('question_id', q.get('id', 'unknown'))
                        qtext = q.get('question_text', q.get('question', q.get('text', 'N/A')))
                        print(f"Question {qid}: {qtext}")
                        
                        # Find and print comment for this question
                        comment = next((c.get('comment', '') for c in comments if c.get('question_id') == qid), None)
                        if comment:
                            print(f"  Comment: {comment}")
                        
                        # Print options if available
                        input_config = q.get('input_config', {})
                        if isinstance(input_config, dict):
                            opts = input_config.get('options', [])
                            if opts:
                                print("  Options:")
                                for o in opts:
                                    print(f"    - {o}")
                        
                        # Handle direct options in question
                        elif 'options' in q and isinstance(q['options'], list):
                            opts = q['options']
                            print("  Options:")
                            for o in opts:
                                if isinstance(o, dict) and ('label' in o or 'text' in o):
                                    print(f"    - {o.get('label', o.get('text', ''))}")
                                else:
                                    print(f"    - {o}")
                        print()
                
                # Print overall comment if available
                overall = annotated.get('overall_comment')
                if overall:
                    print(f"Overall comment: {overall}\n")
            
            # If standard format is not found, check for simple survey structure
            elif 'survey' in survey_dict:
                survey = survey_dict.get('survey', {})
                print(f"Theme:   {survey.get('theme', survey.get('title', 'N/A'))}")
                print(f"Purpose: {survey.get('purpose', survey.get('description', 'N/A'))}\n")
                
                for q in survey.get('questions', []):
                    if isinstance(q, dict):
                        qid = q.get('question_id', q.get('id', 'unknown'))
                        qtext = q.get('question_text', q.get('question', q.get('text', 'N/A')))
                        print(f"Q{qid}: {qtext}")
                        
                        # Print options directly
                        if 'options' in q and isinstance(q['options'], list):
                            opts = q['options']
                            print("  Options:")
                            for o in opts:
                                if isinstance(o, dict) and ('label' in o or 'text' in o):
                                    print(f"    - {o.get('label', o.get('text', ''))}")
                                else:
                                    print(f"    - {o}")
                        
                        # Print input config if exists
                        input_config = q.get('input_config', {})
                        if isinstance(input_config, dict) and input_config:
                            opts = input_config.get('options', [])
                            if opts:
                                print("  Options:")
                                for o in opts:
                                    print(f"    - {o}")
                        print()
                        
            # Check for the SurveyConversionOutput format (directly from first conversion step)
            elif 'title' in survey_dict and 'fields' in survey_dict:
                print(f"Title: {survey_dict.get('title', 'N/A')}\n")
                
                for i, field in enumerate(survey_dict.get('fields', [])):
                    print(f"Q{i+1}: {field.get('title', 'N/A')}")
                    
                    # Print options if available
                    opts = field.get('options', [])
                    if opts:
                        print("  Options:")
                        for o in opts:
                            print(f"    - {o}")
                    print()
            
            # Check for the simplest possible format (just theme and questions)
            elif 'theme' in survey_dict and 'questions' in survey_dict:
                print(f"Theme: {survey_dict.get('theme', 'N/A')}")
                if 'purpose' in survey_dict:
                    print(f"Purpose: {survey_dict.get('purpose', 'N/A')}")
                print()
                
                for q in survey_dict.get('questions', []):
                    if isinstance(q, dict):
                        qid = q.get('question_id', q.get('id', 'unknown'))
                        qtext = q.get('question_text', q.get('question', q.get('text', 'N/A')))
                        print(f"Q{qid}: {qtext}")
                        
                        # Print options directly from the question
                        if 'options' in q and isinstance(q['options'], list):
                            opts = q['options']
                            print("  Options:")
                            for o in opts:
                                if isinstance(o, dict) and ('label' in o or 'text' in o):
                                    print(f"    - {o.get('label', o.get('text', ''))}")
                                else:
                                    print(f"    - {o}")
                        
                        # Print from input_config if exists
                        input_config = q.get('input_config', {})
                        if isinstance(input_config, dict):
                            opts = input_config.get('options', [])
                            if opts:
                                print("  Options:")
                                for o in opts:
                                    print(f"    - {o}")
                        print()
            
            else:
                # Fallback to just printing the JSON structure
                print("Survey structure doesn't match expected format. Raw structure:")
                print(json.dumps(survey_dict, indent=2))
                
        except Exception as e:
            print(f"Error parsing survey structure: {str(e)}")
            print("Raw survey data:")
            print(json.dumps(survey_dict, indent=2))

# ========== Survey to Qualtrics Conversion ==========
def survey_dict_to_qualtrics_payload(survey_dict: dict) -> dict:
    """
    Convert a custom survey dict to a Qualtrics v3 API survey-definitions payload
    Supports question types: multiple_choice, single_choice, slider, text_input
    
    Args:
        survey_dict: Dictionary with the survey data
        
    Returns:
        Dictionary formatted for Qualtrics API
    """
    # Extract the revised survey (works whether using original survey_dict or enhanced_dict)
    survey_meta = None
    if "revised_survey" in survey_dict:
        survey_meta = survey_dict["revised_survey"]
    elif "survey" in survey_dict:
        survey_meta = survey_dict["survey"]
    else:
        survey_meta = survey_dict  # Assume it's already the survey structure
        
    payload = {
        "SurveyName":      survey_meta.get("theme", "New Survey"),
        "Language":        "EN",
        "ProjectCategory": "CORE",
        "Questions":       {}
    }

    for q in survey_meta.get("questions", []):
        raw_id = q.get("question_id", "")               # e.g. "q4"
        num    = re.sub(r'\D+', '', raw_id) or "1"      # extract number "4"
        qid    = f"QID{num}"                            # assemble "QID4"

        qt  = q.get("question_text", "")
        it  = q.get("input_type", "")
        cfg = q.get("input_config", {})

        # ---- Common fields ----
        qobj = {
            "QuestionText":      qt,
            "DataExportTag":     qid,
            "Configuration":     {"QuestionDescriptionOption": "UseText"},
            "Validation":        {"Settings": {"ForceResponse": "OFF", "Type": "None"}}
        }

        # ---- Multiple/Single choice questions ----
        if it in ("multiple_choice", "single_choice"):
            choices = {}
            # Handle both list of strings and list of dict formats for options
            options = cfg.get("options", [])
            for i, opt in enumerate(options):
                if isinstance(opt, dict) and "text" in opt:
                    # If options is a list of dicts with text field
                    idx = str(i+1)
                    txt = opt["text"]
                elif isinstance(opt, str) and "=" in opt:
                    # If format is "1=Strongly disagree"
                    idx, txt = opt.split("=", 1)
                    idx, txt = idx.strip(), txt.strip()
                else:
                    # Default case: just a string or other format
                    idx = str(i+1)
                    txt = str(opt).strip()
                choices[idx] = {"Display": txt}

            qobj.update({
                "QuestionType": "MC",
                "Selector":     "SAVR" if it == "multiple_choice" else "SINGLE",
                "SubSelector":  "TX",
                "Choices":      choices
            })

        # ---- Slider questions ----
        elif it == "slider":
            # Safely read slider parameters (default 0-100, step 1)
            start = cfg.get("min", cfg.get("start", 0))
            end   = cfg.get("max", cfg.get("end", 100))
            step  = cfg.get("step", cfg.get("stepSize", 1))
        
            qobj.update({
                "QuestionType": "SL",
                "Selector":     "Slider",
                "SubSelector":  "SL",
                "SliderStart":  start,
                "SliderEnd":    end,
                "SliderStep":   step
            })
            
        # ---- Text input questions ----
        elif it == "text_input":   
            qobj.update({
                "QuestionType": "TE",
                "Selector":     "ML"   # Text Entry must use ML
            })
        else:
            # Default to multiple choice if input_type is unknown
            qobj.update({
                "QuestionType": "MC",
                "Selector":     "SINGLE",
                "SubSelector":  "TX",
                "Choices":      {"1": {"Display": "Yes"}, "2": {"Display": "No"}}
            })

        # Insert into final payload
        payload["Questions"][qid] = qobj

    return payload

# ========== Qualtrics API Client ==========
class QualtricsClient:
    """Handles all Qualtrics API interactions"""
    
    def __init__(self):
        """Initialize Qualtrics API client with credentials from .env file"""
        # Print current working directory to help debug file path issues
        print(f"Current working directory: {os.getcwd()}")
        
        # Check if .env file exists
        if os.path.exists('.env'):
            print("Found .env file in current directory")
        else:
            print("WARNING: No .env file found in current directory!")
            
        # Load environment variables
        load_dotenv(verbose=True)
        
        self.api_token = os.getenv('QUALTRICS_API_TOKEN')
        self.data_center = os.getenv('QUALTRICS_DATA_CENTER')
        self.directory_id = os.getenv('QUALTRICS_DIRECTORY_ID')
        
        # Print obfuscated token for debugging (only first/last 4 chars)
        if self.api_token:
            token_length = len(self.api_token)
            masked_token = self.api_token[:4] + '*' * (token_length - 8) + self.api_token[-4:] if token_length > 8 else "****"
            print(f"API Token loaded (masked): {masked_token}")
        else:
            print("WARNING: No API token found in environment variables!")
            
        if self.data_center:
            print(f"Data center: {self.data_center}")
        else:
            print("WARNING: No data center found in environment variables!")
        
        if not self.api_token or not self.data_center:
            raise ValueError("Missing Qualtrics API credentials in .env file")
            
        # Set up base URL for API requests
        self.base_url = f"https://{self.data_center}.qualtrics.com/API/v3/"
        self.headers = {
            "X-API-Token": self.api_token,
            "Content-Type": "application/json"
        }
        
        # Test connection
        print("Testing Qualtrics API connection...")
        try:
            test_url = f"{self.base_url}whoami"
            response = requests.get(test_url, headers=self.headers)
            if response.status_code == 200:
                user_info = response.json()["result"]
                print(f"Connection successful! Authenticated as: {user_info.get('firstName', '')} {user_info.get('lastName', '')}")
            else:
                print(f"Connection test failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
        except Exception as e:
            print(f"Error testing connection: {str(e)}")
    
    def create_survey(self, survey_name, survey_template=None):
        """
        Create a new survey in Qualtrics
        
        Args:
            survey_name (str): Name of the survey
            survey_template (dict, optional): Survey template JSON
            
        Returns:
            str: Survey ID of the created survey
        """
        print(f"Creating survey: {survey_name}")
        
        # If no template is provided, use a basic template
        if not survey_template:
            # Define the survey payload with required fields including ProjectCategory
            survey_payload = {
                "SurveyName": survey_name,
                "Language": "EN",
                "ProjectCategory": "CORE", # Required field
                "Questions": {
                    "QID1": {
                        "QuestionText": "What is your age?",
                        "QuestionType": "MC",
                        "Selector": "SAVR", # Required selector for multiple choice
                        "SubSelector": "TX", # Text selector
                        "Configuration": {
                            "QuestionDescriptionOption": "UseText"
                        },
                        "Validation": {
                            "Settings": {
                                "ForceResponse": "OFF",
                                "Type": "None"
                            }
                        },
                        "Choices": {
                            "1": {"Display": "18-24"},
                            "2": {"Display": "25-34"},
                            "3": {"Display": "35-44"},
                            "4": {"Display": "45-54"},
                            "5": {"Display": "55-64"},
                            "6": {"Display": "65+"}
                        }
                    }
                }
            }
        else:
            # If a template is provided, make sure it includes ProjectCategory
            survey_payload = survey_template
            if "ProjectCategory" not in survey_payload:
                survey_payload["ProjectCategory"] = "CORE"
        
        # Create survey
        url = f"{self.base_url}survey-definitions"
        payload = json.dumps(survey_payload)
        
        print(f"Sending payload to Qualtrics: {payload[:200]}...")
        
        response = requests.post(url, headers=self.headers, data=payload)
        
        if response.status_code != 200:
            print(f"Error response: {response.text}")
            raise Exception(f"Failed to create survey: {response.text}")
        
        result = response.json()
        survey_id = result["result"]["SurveyID"]
        print(f"Survey created successfully with ID: {survey_id}")
        
        return survey_id
    
    def activate_survey(self, survey_id):
        """
        Activate a survey to make it available for distribution
        
        Args:
            survey_id (str): ID of the survey to activate
            
        Returns:
            bool: True if successful
        """
        print(f"Activating survey: {survey_id}")
        
        url = f"{self.base_url}surveys/{survey_id}"
        payload = json.dumps({"isActive": True})
        
        response = requests.put(url, headers=self.headers, data=payload)
        
        if response.status_code != 200:
            raise Exception(f"Failed to activate survey: {response.text}")
        
        print(f"Survey activated successfully")
        return True
    
    def create_distribution_link(self, survey_id, link_type="Anonymous"):
        """
        Create a distribution link for a survey
        
        Args:
            survey_id (str): ID of the survey to distribute
            link_type (str): Type of link (Anonymous or Individual)
            
        Returns:
            str: Distribution link URL
        """
        print(f"Creating distribution link for survey: {survey_id}")
        
        # For anonymous links, we can construct the URL directly based on the standard pattern
        # https://DATACENTERID.qualtrics.com/jfe/form/SURVEYID
        if link_type == "Anonymous":
            survey_link = f"https://{self.data_center}.qualtrics.com/jfe/form/{survey_id}"
            print(f"Anonymous survey link created: {survey_link}")
            return survey_link
        
        # For other distribution types, we would use the API, but that's not implemented yet
        else:
            raise NotImplementedError(f"Distribution type '{link_type}' is not yet supported")
    
    def get_survey_responses(self, survey_id, file_format="csv"):
        """
        Download survey responses
        
        Args:
            survey_id (str): ID of the survey
            file_format (str): Format of the response file (csv, json, spss, etc.)
            
        Returns:
            pandas.DataFrame: Survey responses as a DataFrame
        """
        print(f"Downloading responses for survey: {survey_id}")
        
        # Step 1: Create the export
        export_url = f"{self.base_url}surveys/{survey_id}/export-responses"
        export_payload = json.dumps({
            "format": file_format,
            "useLabels": True
        })
        
        export_response = requests.post(export_url, headers=self.headers, data=export_payload)
        
        if export_response.status_code != 200:
            raise Exception(f"Failed to initiate export: {export_response.text}")
        
        progress_id = export_response.json()["result"]["progressId"]
        
        # Step 2: Check export progress
        progress_status = "inProgress"
        progress = 0
        
        while progress_status != "complete" and progress < 100:
            progress_url = f"{self.base_url}surveys/{survey_id}/export-responses/{progress_id}"
            progress_response = requests.get(progress_url, headers=self.headers)
            
            if progress_response.status_code != 200:
                raise Exception(f"Failed to check export progress: {progress_response.text}")
            
            progress_result = progress_response.json()["result"]
            progress_status = progress_result["status"]
            progress = progress_result.get("percentComplete", 0)
            
            print(f"Export progress: {progress}%")
            
            if progress_status != "complete" and progress < 100:
                time.sleep(2)
        
        # Step 3: Download the file
        file_id = progress_result["fileId"]
        download_url = f"{self.base_url}surveys/{survey_id}/export-responses/{file_id}/file"
        download_response = requests.get(download_url, headers=self.headers)
        
        if download_response.status_code != 200:
            raise Exception(f"Failed to download responses: {download_response.text}")
        
        # Step 4: Extract and parse the zip file
        with zipfile.ZipFile(io.BytesIO(download_response.content)) as zip_file:
            data_file = [f for f in zip_file.namelist() if f.endswith(f".{file_format}")][0]
            with zip_file.open(data_file) as file:
                if file_format == "csv":
                    df = pd.read_csv(file)
                elif file_format == "json":
                    df = pd.read_json(file)
                else:
                    raise ValueError(f"Unsupported file format: {file_format}")
        
        # Save to CSV with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"survey_responses_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Saved {len(df)} responses to {csv_filename}")
        
        return df

# ========== MTurk API Client ==========
class MTurkClient:
    """Handles all MTurk API interactions"""
    def __init__(self, 
                aws_access_key_id: str = None, 
                aws_secret_access_key: str = None, 
                use_sandbox: bool = True):  # Default to sandbox mode for safety
        """
        Initialize MTurk client
        
        Args:
            aws_access_key_id: Optional override for AWS access key
            aws_secret_access_key: Optional override for AWS secret key
            use_sandbox: Boolean for using sandbox (defaults to True for safety)
        """
        # Load from .env file
        load_dotenv()
        
        # Set AWS credentials (with optional overrides)
        self.aws_access_key_id = aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY')
        
        # Check if credentials are available
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            raise ValueError("Missing AWS credentials in .env file or constructor parameters")

        # Determine sandbox mode (with optional override)
        if use_sandbox is None:
            # Read from environment if not provided in constructor
            self.use_sandbox = os.getenv('MTURK_SANDBOX', 'True').lower() == 'true'
        else:
            self.use_sandbox = use_sandbox

        # Set endpoint based on sandbox mode
        region = os.getenv('AWS_REGION', 'us-east-1')
        endpoint = (
            'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
            if self.use_sandbox else
            'https://mturk-requester.us-east-1.amazonaws.com'
        )

        # Create boto3 client
        try:
            self.client = boto3.client(
                'mturk',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=region,
                endpoint_url=endpoint
            )
            print(f"MTurk client initialized in {'Sandbox' if self.use_sandbox else 'Production'} mode")
            
            # Verify connection by checking account balance
            self.get_account_balance()
            
        except Exception as e:
            print(f"Error initializing MTurk client: {str(e)}")
            print("Please verify your AWS credentials and MTurk account configuration.")
            print("For MTurk integration, you need:")
            print("1. Valid AWS credentials in your .env file")
            print("2. Your AWS account linked to your MTurk Requester account")
            print("3. Proper permissions for the MTurk API")
            
            # Create a dummy client for graceful degradation
            self.client = None
            self.connection_error = str(e)
            
    def get_account_balance(self):
        """Get the available MTurk account balance"""
        if not self.client:
            print(f"Cannot check balance: {self.connection_error}")
            return 0.0
            
        try:
            response = self.client.get_account_balance()
            balance = response['AvailableBalance']
            print(f"MTurk account balance: ${balance}")
            return float(balance)
        except Exception as e:
            print(f"Error checking balance: {str(e)}")
            return 0.0
    
    def create_hit_with_survey_link(self, survey_link, hit_config=None):
        """
        Create an MTurk HIT with a link to a Qualtrics survey
        
        Args:
            survey_link (str): URL to the Qualtrics survey
            hit_config (dict, optional): Custom configuration for the HIT
            
        Returns:
            str: HIT ID
        """
        print("Creating MTurk HIT with survey link")
        
        # Default HIT configuration
        if not hit_config:
            hit_config = {
                'Title': 'Complete a short survey',
                'Description': 'We need your input for a quick survey that should take less than 10 minutes',
                'Keywords': 'survey, research, opinion, feedback',
                'Reward': '0.50',
                'MaxAssignments': 100,
                'LifetimeInSeconds': 86400,  # 1 day
                'AssignmentDurationInSeconds': 1800,  # 30 minutes
                'AutoApprovalDelayInSeconds': 86400,  # 1 day
                'QualificationRequirements': []
            }
        
        # Create the HTML question with the survey link
        question_html = f"""
        <HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
            <HTMLContent><![CDATA[
                <!DOCTYPE html>
                <html>
                <head>
                    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8"/>
                    <script type='text/javascript' src='https://s3.amazonaws.com/mturk-public/externalHIT_v1.js'></script>
                </head>
                <body>
                    <form name='mturk_form' method='post' id='mturk_form' action='https://www.mturk.com/mturk/externalSubmit'>
                        <input type='hidden' value='' name='assignmentId' id='assignmentId'/>
                        <h1>Survey Task</h1>
                        <p>Please complete the survey at the following link:</p>
                        <p><a href='{survey_link}' target='_blank'>{survey_link}</a></p>
                        <p>After completing the survey, you will receive a completion code. Enter the code below:</p>
                        <p><input type='text' name='completion_code' id='completion_code' size='40'/></p>
                        <p><input type='submit' id='submitButton' value='Submit' /></p>
                    </form>
                    <script language='Javascript'>
                        turkSetAssignmentID();
                    </script>
                </body>
                </html>
            ]]></HTMLContent>
            <FrameHeight>600</FrameHeight>
        </HTMLQuestion>
        """
        
        # Create the HIT
        response = self.client.create_hit(
            Title=hit_config['Title'],
            Description=hit_config['Description'],
            Keywords=hit_config['Keywords'],
            Reward=hit_config['Reward'],
            MaxAssignments=hit_config['MaxAssignments'],
            LifetimeInSeconds=hit_config['LifetimeInSeconds'],
            AssignmentDurationInSeconds=hit_config['AssignmentDurationInSeconds'],
            AutoApprovalDelayInSeconds=hit_config['AutoApprovalDelayInSeconds'],
            Question=question_html,
            QualificationRequirements=hit_config['QualificationRequirements']
        )
        
        hit_id = response['HIT']['HITId']
        hit_type_id = response['HIT']['HITTypeId']
        
        print(f"HIT created successfully with ID: {hit_id}")
        
        # Print the HIT URL
        if self.use_sandbox:
            worker_url = f"https://workersandbox.mturk.com/mturk/preview?groupId={hit_type_id}"
        else:
            worker_url = f"https://worker.mturk.com/mturk/preview?groupId={hit_type_id}"
            
        print(f"Workers can access the HIT at: {worker_url}")
        
        return hit_id
    
    def get_hit_assignments(self, hit_id):
        """
        Get all assignments for a HIT
        
        Args:
            hit_id (str): ID of the HIT
            
        Returns:
            list: List of assignment dictionaries
        """
        print(f"Getting assignments for HIT: {hit_id}")
        
        # List to store all assignments
        all_assignments = []
        
        # Get assignments with pagination
        next_token = None
        
        while True:
            if next_token:
                response = self.client.list_assignments_for_hit(
                    HITId=hit_id,
                    NextToken=next_token,
                    MaxResults=100
                )
            else:
                response = self.client.list_assignments_for_hit(
                    HITId=hit_id,
                    MaxResults=100
                )
            
            all_assignments.extend(response['Assignments'])
            
            if 'NextToken' in response:
                next_token = response['NextToken']
            else:
                break
        
        print(f"Found {len(all_assignments)} assignments")
        return all_assignments
    
    def approve_assignments(self, assignments, feedback=None):
        """
        Approve multiple assignments
        
        Args:
            assignments (list): List of assignment dictionaries or IDs
            feedback (str, optional): Feedback to workers
            
        Returns:
            int: Number of successfully approved assignments
        """
        approved_count = 0
        
        for assignment in assignments:
            # Extract assignment ID if a dictionary was provided
            assignment_id = assignment['AssignmentId'] if isinstance(assignment, dict) else assignment
            
            try:
                self.client.approve_assignment(
                    AssignmentId=assignment_id,
                    RequesterFeedback=feedback if feedback else "Thank you for your participation!"
                )
                approved_count += 1
            except Exception as e:
                print(f"Error approving assignment {assignment_id}: {str(e)}")
        
        print(f"Successfully approved {approved_count} assignments")
        return approved_count
    
    def delete_hit(self, hit_id):
        """
        Delete a HIT
        
        Args:
            hit_id (str): ID of the HIT to delete
            
        Returns:
            bool: True if successful
        """
        try:
            # Get the HIT status
            hit = self.client.get_hit(HITId=hit_id)
            status = hit['HIT']['HITStatus']
            
            # If the HIT is reviewable, dispose of it
            if status == 'Reviewable':
                self.client.delete_hit(HITId=hit_id)
                print(f"HIT {hit_id} deleted successfully")
                return True
            
            # If the HIT is assignable, expire it first then delete it
            elif status == 'Assignable':
                self.client.update_expiration_for_hit(
                    HITId=hit_id,
                    ExpireAt=datetime(2015, 1, 1)  # Set to a past date to expire immediately
                )
                time.sleep(1)  # Give time for the HIT to update
                self.client.delete_hit(HITId=hit_id)
                print(f"HIT {hit_id} expired and deleted successfully")
                return True
                
            else:
                print(f"Cannot delete HIT {hit_id}, status is {status}")
                return False
                
        except Exception as e:
            print(f"Error deleting HIT {hit_id}: {str(e)}")
            return False
        
# ========== Qualtrics and MTurk Integration ==========
class QualtricsAndMTurkAutomation:
    """Handles the automation of creating Qualtrics surveys and MTurk HITs"""
    
    def __init__(self):
        """Initialize the automation with Qualtrics and MTurk clients"""
        self.qualtrics_client = QualtricsClient()
        self.mturk_client = MTurkClient()
    
    def run(self, qualtrics_payload, hit_config=None):
        """
        Run the automation to create a Qualtrics survey and MTurk HIT
        
        Args:
            qualtrics_payload (dict): Survey definition for Qualtrics
            hit_config (dict, optional): Configuration for MTurk HIT
            
        Returns:
            dict: Results including survey ID, survey link, and HIT ID
        """
        print("Starting Qualtrics and MTurk automation...")
        
        # Create Qualtrics survey
        survey_name = qualtrics_payload.get("SurveyName", "New Survey")
        survey_id = self.qualtrics_client.create_survey(survey_name, qualtrics_payload)
        
        # Activate the survey
        self.qualtrics_client.activate_survey(survey_id)
        
        # Get distribution link
        survey_link = self.qualtrics_client.create_distribution_link(survey_id)
        
        # Ensure hit_config has valid values
        if hit_config is None:
            hit_config = {}
            
        # Validate and provide defaults for required fields
        hit_config['Title'] = hit_config.get('Title') or f'Complete a survey on {survey_name}'
        hit_config['Description'] = hit_config.get('Description') or f'Complete a research survey about {survey_name}'
        hit_config['Keywords'] = hit_config.get('Keywords') or 'survey, research, feedback'
        hit_config['Reward'] = hit_config.get('Reward') or '0.75'
        hit_config['MaxAssignments'] = hit_config.get('MaxAssignments') or 100
        hit_config['LifetimeInSeconds'] = hit_config.get('LifetimeInSeconds') or 86400
        hit_config['AssignmentDurationInSeconds'] = hit_config.get('AssignmentDurationInSeconds') or 1800
        hit_config['AutoApprovalDelayInSeconds'] = hit_config.get('AutoApprovalDelayInSeconds') or 86400
        hit_config['QualificationRequirements'] = hit_config.get('QualificationRequirements') or []
        
        # Print the validated hit_config for debugging
        print("Validated HIT Configuration:")
        print(f"Title: {hit_config['Title']}")
        print(f"Description: {hit_config['Description']}")
        print(f"Reward: ${hit_config['Reward']}")
        print(f"Max Assignments: {hit_config['MaxAssignments']}")
        
        # Create MTurk HIT
        hit_id = self.mturk_client.create_hit_with_survey_link(survey_link, hit_config)
        
        # Return results
        return {
            "survey_id": survey_id,
            "survey_link": survey_link,
            "hit_id": hit_id
        }
    
    def collect_and_process_results(self, survey_id, hit_id, auto_approve=True):
        """
        Collect and process results from Qualtrics and MTurk
        
        Args:
            survey_id (str): Qualtrics survey ID
            hit_id (str): MTurk HIT ID
            auto_approve (bool): Whether to automatically approve assignments
            
        Returns:
            dict: Results including responses and assignment data
        """
        # Get Qualtrics responses
        responses = self.qualtrics_client.get_survey_responses(survey_id)
        
        # Save responses to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"survey_responses_{timestamp}.csv"
        responses.to_csv(csv_filename, index=False)
        print(f"Saved {len(responses)} responses to {csv_filename}")
        
        # Get MTurk assignments
        assignments = self.mturk_client.get_hit_assignments(hit_id)
        
        # Auto-approve assignments if requested
        approved_count = 0
        if auto_approve and assignments:
            approved_count = self.mturk_client.approve_assignments(assignments)
        
        # Return results
        return {
            "responses": responses,
            "csv_filename": csv_filename,
            "assignments": assignments,
            "approved_count": approved_count
        }
    
# ========== Main Function for Survey Processing ==========
def main():
    """Main function to run the enhanced survey processing and deployment flow"""
    # Load environment variables
    load_dotenv()
    
    # Display input requirements
    print("========================================")
    print("⚠️  INPUT REQUIREMENTS:")
    print("- You must include a line starting with 'Topic:'")
    print("- You must include at least one line starting with 'Questions:'")
    print("Otherwise, the survey cannot be processed.")
    print("========================================")
    
    # Get survey input
    print("\nPlease enter your survey content. Press Enter twice on an empty line when finished.")
    survey_lines = []
    
    while True:
        line = input()
        if not line and survey_lines and not survey_lines[-1]:  # Two empty lines in a row
            break
        survey_lines.append(line)
    
    survey_to_process = "\n".join(survey_lines).strip()
    
    if not survey_to_process:
        print("No survey text provided. Exiting.")
        return
    
    # Check for required elements
    if 'Topic:' not in survey_to_process:
        print("Error: Survey must include a line starting with 'Topic:'")
        return
    
    if 'Questions:' not in survey_to_process:
        print("Error: Survey must include at least one line starting with 'Questions:'")
        return
    
    # Initialize the enhanced survey flow
    enhancement_flow = SurveyEnhancementFlow()
    
    # Run the initial processing
    try:
        initial_result = enhancement_flow.run(survey_to_process)
        
        # Run the interactive enhancement loop
        final_result = enhancement_flow.interactive_enhancement()
        
        print("\n========================================")
        print("Survey enhancement and deployment complete!")
        print("========================================")
        
        return final_result
        
    except Exception as e:
        print(f"Error in survey processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Function to collect data from a previously deployed survey
def collect_survey_data():
    """
    Collect data from a completed survey in Qualtrics and MTurk
    
    Returns:
        dict: Collected data including responses and assignment information
    """
    # Get survey ID and HIT ID
    survey_id = input("Enter your Qualtrics Survey ID: ")
    hit_id = input("Enter your MTurk HIT ID (leave blank if not using MTurk): ")
    
    if not survey_id:
        print("Error: Qualtrics Survey ID is required")
        return None
    
    print(f"Ready to collect data for Survey ID: {survey_id}" + 
          (f" and HIT ID: {hit_id}" if hit_id else ""))

    # Create automation instance
    automation = QualtricsAndMTurkAutomation()
    
    try:
        # Collect and process results
        if hit_id:
            collected_data = automation.collect_and_process_results(
                survey_id=survey_id,
                hit_id=hit_id,
                auto_approve=input("Auto-approve MTurk assignments? (y/n): ").lower() == 'y'
            )
        else:
            # Just get Qualtrics data if no HIT ID
            responses = automation.qualtrics_client.get_survey_responses(survey_id)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"survey_responses_{timestamp}.csv"
            responses.to_csv(csv_filename, index=False)
            
            collected_data = {
                "responses": responses,
                "csv_filename": csv_filename,
                "assignments": [],
                "approved_count": 0
            }
        
        # Display response summary
        if 'responses' in collected_data and len(collected_data['responses']) > 0:
            print("\nResponse Preview:")
            print(collected_data['responses'].head())
            
            # Save to CSV with a more informative name
            detailed_csv = f"survey_{survey_id}_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            collected_data['responses'].to_csv(detailed_csv, index=False)
            print(f"Detailed responses saved to: {detailed_csv}")
        else:
            print("No responses collected or responses are empty.")
            
        return collected_data
        
    except Exception as e:
        print(f"Error collecting data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Entry point with options menu
if __name__ == "__main__":
    # Present options to the user
    print("====== Survey Processing System ======")
    print("1. Create and enhance a new survey")
    print("2. Collect data from existing survey")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == "1":
        # Run the main survey creation flow
        main()
    elif choice == "2":
        # Run the data collection flow
        collect_survey_data()
    elif choice == "3":
        print("Exiting...")
    else:
        print("Invalid choice. Exiting...")