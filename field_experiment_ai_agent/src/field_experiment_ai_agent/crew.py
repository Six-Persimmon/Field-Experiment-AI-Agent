from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from typing import List, Union, Literal, Optional
from pydantic import BaseModel, Field

#####记得后面call task的时候specify哪个output需要=Survey

# For multiple or single choice questions
class ChoiceOption(BaseModel):
    text: str
    value: str

class ChoiceConfig(BaseModel):
    options: List[ChoiceOption]

# For slider input
class SliderConfig(BaseModel):
    min: float
    max: float
    step: float

# For free text input
class TextInputConfig(BaseModel):
    placeholder: Optional[str] = None
    multiline: bool = False

# Question model
class Question(BaseModel):
    question_id: str
    question_text: str
    input_type: Literal["multiple_choice", "single_choice", "slider", "text_input"]
    input_config: Union[ChoiceConfig, SliderConfig, TextInputConfig]


# Survey model
class Survey(BaseModel):
    theme: str
    purpose: str
    questions: List[Question]

# Comment model
class QuestionComment(BaseModel):
    question_id: str
    comment: str

# Annoted Survey model
class AnnotatedSurvey(BaseModel):
    survey: Survey
    question_comments: List[QuestionComment]
    overall_comment: Optional[str]

# Improvement final output model
class SurveyImprovementResult(BaseModel):
    original_with_comments: AnnotatedSurvey
    revised_survey: Survey


# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class FieldExperimentAiAgent():
    """FieldExperimentAiAgent crew"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
        )

    
    @task
    def improve_survey_task(self) -> Task:
        return Task(
            config=self.tasks_config['improve_survey'],
            output_pydantic=SurveyImprovementResult
        )

    @crew
    def crew(self) -> Crew:
        """Creates the FieldExperimentAiAgent crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
