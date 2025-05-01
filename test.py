import os
import time
import json
import zipfile
import io
import csv
import pandas as pd
import requests
from datetime import datetime, timedelta
import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ----------------- Qualtrics API Configuration -----------------

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
                "ProjectCategory": "CORE", # This is the required field that was missing
                "Questions": {
                    "QID1": {
                        "QuestionText": "What is your age?",
                        "QuestionType": "MC",
                        "Selector": "SAVR", # Required selector for multiple choice questions
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
                    },
                    "QID2": {
                        "QuestionText": "How satisfied are you with our product?",
                        "QuestionType": "Likert",
                        "Selector": "LSL", # Likert scale
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
                            "1": {"Display": "Very dissatisfied"},
                            "2": {"Display": "Dissatisfied"},
                            "3": {"Display": "Neutral"},
                            "4": {"Display": "Satisfied"},
                            "5": {"Display": "Very satisfied"}
                        }
                    },
                    "QID3": {
                        "QuestionText": "Any additional comments?",
                        "QuestionType": "TE", # Text entry
                        "Selector": "ML", # Multi-line
                        "Configuration": {
                            "QuestionDescriptionOption": "UseText"
                        },
                        "Validation": {
                            "Settings": {
                                "ForceResponse": "OFF",
                                "Type": "None"
                            }
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
        
        print(f"Successfully downloaded {len(df)} responses")
        return df

# ----------------- Amazon MTurk Configuration -----------------

class MTurkClient:
    """Handles all MTurk API interactions"""
    
    def __init__(self):
        """Initialize MTurk API client with credentials from .env file"""
        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.use_sandbox = os.getenv('MTURK_SANDBOX', 'True').lower() == 'true'
        
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            raise ValueError("Missing AWS API credentials in .env file")
            
        # Set up MTurk client
        self.endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com' if self.use_sandbox else 'https://mturk-requester.us-east-1.amazonaws.com'
        
        self.client = boto3.client(
            'mturk',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name='us-east-1',
            endpoint_url=self.endpoint_url
        )
        
        env_name = "Sandbox" if self.use_sandbox else "Production"
        print(f"MTurk client initialized in {env_name} mode")
        
    def get_account_balance(self):
        """Get the available MTurk account balance"""
        response = self.client.get_account_balance()
        balance = response['AvailableBalance']
        print(f"MTurk account balance: ${balance}")
        return float(balance)
    
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
                'MaxAssignments': 10,
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

# ----------------- Main Application Logic -----------------

class QualtricsAndMTurkAutomation:
    """Main class that orchestrates the entire workflow"""
    
    def __init__(self):
        """Initialize the automation workflow"""
        self.qualtrics = QualtricsClient()
        self.mturk = MTurkClient()
        
    def run_complete_workflow(self, survey_name, hit_config=None, survey_template=None):
        """
        Run the complete workflow
        
        Args:
            survey_name (str): Name of the survey to create
            hit_config (dict, optional): Configuration for the MTurk HIT
            survey_template (dict, optional): Template for the Qualtrics survey
            
        Returns:
            dict: Results of the workflow
        """
        results = {}
        
        try:
            # Step 1: Check MTurk account balance
            balance = self.mturk.get_account_balance()
            results['mturk_balance'] = balance
            
            # Step 2: Create Qualtrics survey
            survey_id = self.qualtrics.create_survey(survey_name, survey_template)
            results['survey_id'] = survey_id
            
            # Step 3: Activate the survey
            self.qualtrics.activate_survey(survey_id)
            
            # Step 4: Create distribution link
            survey_link = self.qualtrics.create_distribution_link(survey_id)
            results['survey_link'] = survey_link
            
            # Step 5: Create MTurk HIT with the survey link
            hit_id = self.mturk.create_hit_with_survey_link(survey_link, hit_config)
            results['hit_id'] = hit_id
            
            print("\nWorkflow completed successfully!")
            print(f"Survey ID: {survey_id}")
            print(f"Survey Link: {survey_link}")
            print(f"HIT ID: {hit_id}")
            
            return results
            
        except Exception as e:
            print(f"Error in workflow: {str(e)}")
            return results
    
    def collect_and_process_results(self, survey_id, hit_id, auto_approve=True):
        """
        Collect results from Qualtrics and process MTurk assignments
        
        Args:
            survey_id (str): ID of the Qualtrics survey
            hit_id (str): ID of the MTurk HIT
            auto_approve (bool): Whether to automatically approve assignments
            
        Returns:
            dict: Results of the collection and processing
        """
        results = {}
        
        try:
            # Step 1: Get responses from Qualtrics
            responses_df = self.qualtrics.get_survey_responses(survey_id)
            results['responses'] = responses_df
            
            # Save responses to a CSV file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"survey_responses_{timestamp}.csv"
            responses_df.to_csv(csv_filename, index=False)
            results['csv_filename'] = csv_filename
            
            print(f"Saved {len(responses_df)} responses to {csv_filename}")
            
            # Step 2: Get MTurk assignments
            assignments = self.mturk.get_hit_assignments(hit_id)
            results['assignments'] = assignments
            
            # Step 3: Auto-approve assignments if requested
            if auto_approve and assignments:
                approved_count = self.mturk.approve_assignments(assignments)
                results['approved_count'] = approved_count
            
            return results
            
        except Exception as e:
            print(f"Error collecting results: {str(e)}")
            return results

# ----------------- Command Line Interface -----------------

def main():
    """Main function that demonstrates the workflow"""
    print("=== Qualtrics & MTurk Automation Tool ===")
    
    # Allow manual credential entry if .env file is not working
    use_manual_credentials = input("Do you want to enter API credentials manually? (y/n): ").lower() == 'y'
    
    if use_manual_credentials:
        qualtrics_api_token = input("Enter your Qualtrics API Token: ")
        qualtrics_data_center = input("Enter your Qualtrics Data Center (e.g., ca1, eu, co1): ")
        aws_access_key = input("Enter your AWS Access Key ID: ")
        aws_secret_key = input("Enter your AWS Secret Access Key: ")
        use_sandbox = input("Use MTurk Sandbox? (y/n): ").lower() == 'y'
        
        # Set environment variables manually
        os.environ['QUALTRICS_API_TOKEN'] = qualtrics_api_token
        os.environ['QUALTRICS_DATA_CENTER'] = qualtrics_data_center
        os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key
        os.environ['MTURK_SANDBOX'] = str(use_sandbox)
        
        print("Credentials set manually.")
    
    try:
        # Create the automation instance
        automation = QualtricsAndMTurkAutomation()
        
        # Check if we want to run the full workflow
        run_full_workflow = input("\nRun the full workflow? (y/n): ").lower() == 'y'
        
        if run_full_workflow:
            # Example survey name
            survey_name = f"Consumer Feedback Survey {datetime.now().strftime('%Y-%m-%d')}"
            
            # HIT configuration
            hit_config = {
                'Title': 'Complete a short consumer feedback survey',
                'Description': 'We need your input for a quick survey about consumer preferences that should take less than 10 minutes',
                'Keywords': 'survey, consumer, feedback, opinion',
                'Reward': '0.75',
                'MaxAssignments': 20,
                'LifetimeInSeconds': 86400,  # 1 day
                'AssignmentDurationInSeconds': 1800,  # 30 minutes
                'AutoApprovalDelayInSeconds': 86400,  # 1 day
                'QualificationRequirements': []
            }
            
            # Run the workflow
            print("\nRunning complete workflow...\n")
            results = automation.run_complete_workflow(survey_name, hit_config)
            
            # If you want to simulate waiting for responses
            if results.get('survey_id') and results.get('hit_id'):
                survey_id = results['survey_id']
                hit_id = results['hit_id']
                
                print("\nTo collect and process results later, run:")
                print(f"survey_id = '{survey_id}'")
                print(f"hit_id = '{hit_id}'")
                print("automation.collect_and_process_results(survey_id, hit_id)")
        else:
            print("\nSkipping full workflow. Just testing connection.")
        # To collect results from a previous run
        if input("\nDo you want to collect results from a previous survey? (y/n): ").lower() == 'y':
            survey_id = input("Enter survey ID (e.g., SV_e5tsBzNds9SuYUS): ")
            hit_id = input("Enter HIT ID (e.g., 33N1S8XHICID5WIXVXE0ZEAG55W1ZX): ")
            automation.collect_and_process_results(survey_id, hit_id)
    except Exception as e:
        print(f"\nError initializing automation: {str(e)}")
        print("\nPlease check your API credentials and try again.")

if __name__ == "__main__":
    main()