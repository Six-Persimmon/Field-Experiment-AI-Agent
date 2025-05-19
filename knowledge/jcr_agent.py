"""
JCR Journal Report Generation Agent

This script generates a research report based on OSF journal survey design and data.

Usage:
    python jcr_agent.py --paper_number 1
"""

import os
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv
import openai
from openai import OpenAI
import json

# Load environment variables
load_dotenv("/Users/princess/Documents/RA/Field-Experiment-AI-Agent/.env")

# Configure OpenAI
client = OpenAI()

# Paths
BASE_DIR = Path("JCR Papers")

def analyze_data(paper_number):
    """Analyze survey data."""
    paper_dir = BASE_DIR / f"Paper #{paper_number}"
    data_dir = paper_dir / "Data and Code"
    
    # Create sample data if it doesn't exist
    if not data_dir.exists():
        print(f"Creating sample data directory: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create more comprehensive sample data for better analysis
        with open(data_dir / "survey_data.csv", "w") as f:
            f.write("participant_id,age,gender,career_stage,discipline,impact_factor_importance,open_access_importance,editorial_reputation,peer_review_time,publishing_frequency\n")
            f.write("1,29,Female,Early-career,Biological Sciences,5,4,3,4,2\n")
            f.write("2,35,Male,Mid-career,Biological Sciences,4,3,5,3,3\n")
            f.write("3,42,Female,Senior,Social Sciences,5,2,4,2,4\n")
            f.write("4,31,Male,Early-career,Social Sciences,3,5,3,4,2\n")
            f.write("5,38,Non-binary,Mid-career,Humanities,2,5,4,3,1\n")
            f.write("6,45,Female,Senior,Biological Sciences,5,1,5,2,5\n")
            f.write("7,27,Male,Early-career,Physical Sciences,3,4,3,5,2\n")
            f.write("8,39,Female,Mid-career,Social Sciences,4,3,4,3,3\n")
            f.write("9,50,Male,Senior,Physical Sciences,5,2,5,1,4\n")
            f.write("10,33,Non-binary,Early-career,Humanities,2,5,3,4,1\n")
            f.write("11,37,Female,Mid-career,Biological Sciences,4,3,4,3,3\n")
            f.write("12,48,Male,Senior,Humanities,4,3,5,2,2\n")
            f.write("13,30,Female,Early-career,Physical Sciences,3,5,2,5,2\n")
            f.write("14,41,Male,Mid-career,Social Sciences,4,3,4,3,3\n")
            f.write("15,52,Female,Senior,Biological Sciences,5,1,5,1,5\n")
            
        # Create a more detailed survey design document
        with open(data_dir / "survey_design.txt", "w") as f:
            f.write("# OSF Journal Survey Design\n\n")
            f.write("## Purpose\n")
            f.write("To understand factors influencing journal selection among researchers across different career stages and disciplines.\n\n")
            f.write("## Survey Structure\n")
            f.write("Demographic Information:\n")
            f.write("- Age (numeric)\n")
            f.write("- Gender (categorical: Male, Female, Non-binary)\n")
            f.write("- Career Stage (categorical: Early-career, Mid-career, Senior)\n")
            f.write("- Discipline (categorical: Biological Sciences, Social Sciences, Physical Sciences, Humanities)\n\n")
            f.write("Journal Selection Factors (Likert scale 1-5, where 1=Not Important, 5=Very Important):\n")
            f.write("- Impact Factor Importance\n")
            f.write("- Open Access Importance\n")
            f.write("- Editorial Reputation\n")
            f.write("- Peer Review Time\n")
            f.write("- Publishing Frequency\n\n")
            f.write("## Methodology\n")
            f.write("The survey was distributed electronically to researchers at 15 universities across North America and Europe. Participation was voluntary and anonymous. Data was collected over a 6-week period.\n\n")
            f.write("## Data Analysis Plan\n")
            f.write("1. Descriptive statistics for all variables\n")
            f.write("2. Comparative analysis across demographic groups\n")
            f.write("3. Correlation analysis between journal selection factors\n")
            f.write("4. Regression analysis to identify predictors of journal selection preferences\n")
        
        # Create sample experiment data
        with open(data_dir / "experiment_results.csv", "w") as f:
            f.write("experiment_id,participant_id,condition,task_completion_time,satisfaction_score,accuracy_rate\n")
            f.write("1,1,Control,45,3,0.75\n")
            f.write("1,2,Control,52,4,0.82\n")
            f.write("1,3,Treatment,38,5,0.90\n")
            f.write("1,4,Treatment,42,4,0.88\n")
            f.write("1,5,Control,57,3,0.71\n")
            f.write("2,6,High_Prestige,31,5,0.92\n")
            f.write("2,7,Low_Prestige,48,2,0.63\n")
            f.write("2,8,High_Prestige,35,4,0.85\n")
            f.write("2,9,Low_Prestige,41,3,0.70\n")
            f.write("2,10,High_Prestige,33,5,0.89\n")
    
    # Create the prompt for data analysis
    prompt = f"""
    You are a Data Analyst examining OSF journal survey data for Paper #{paper_number}.
    
    Your task is to conduct a comprehensive analysis of the data files in the Paper #{paper_number} directory.
    
    IMPORTANT: DO NOT merely describe how to analyze the data. Actually perform the analysis on the following data:
    
    1. Survey Data (simplified for illustration):
    ```
    participant_id,age,gender,career_stage,discipline,impact_factor_importance,open_access_importance,editorial_reputation,peer_review_time,publishing_frequency
    1,29,Female,Early-career,Biological Sciences,5,4,3,4,2
    2,35,Male,Mid-career,Biological Sciences,4,3,5,3,3
    3,42,Female,Senior,Social Sciences,5,2,4,2,4
    4,31,Male,Early-career,Social Sciences,3,5,3,4,2
    5,38,Non-binary,Mid-career,Humanities,2,5,4,3,1
    6,45,Female,Senior,Biological Sciences,5,1,5,2,5
    7,27,Male,Early-career,Physical Sciences,3,4,3,5,2
    8,39,Female,Mid-career,Social Sciences,4,3,4,3,3
    9,50,Male,Senior,Physical Sciences,5,2,5,1,4
    10,33,Non-binary,Early-career,Humanities,2,5,3,4,1
    11,37,Female,Mid-career,Biological Sciences,4,3,4,3,3
    12,48,Male,Senior,Humanities,4,3,5,2,2
    13,30,Female,Early-career,Physical Sciences,3,5,2,5,2
    14,41,Male,Mid-career,Social Sciences,4,3,4,3,3
    15,52,Female,Senior,Biological Sciences,5,1,5,1,5
    ```
    
    2. Experiment Results:
    ```
    experiment_id,participant_id,condition,task_completion_time,satisfaction_score,accuracy_rate
    1,1,Control,45,3,0.75
    1,2,Control,52,4,0.82
    1,3,Treatment,38,5,0.90
    1,4,Treatment,42,4,0.88
    1,5,Control,57,3,0.71
    2,6,High_Prestige,31,5,0.92
    2,7,Low_Prestige,48,2,0.63
    2,8,High_Prestige,35,4,0.85
    2,9,Low_Prestige,41,3,0.70
    2,10,High_Prestige,33,5,0.89
    ```
    
    Provide a complete and comprehensive data analysis including:
    
    1. Descriptive Statistics:
       - Calculate mean, median, standard deviation for all numeric variables
       - Frequency distributions for categorical variables
       - Create tables showing these statistics
    
    2. Correlation Analysis:
       - Calculate correlations between impact factor, open access, editorial reputation, etc.
       - Identify significant relationships with correlation coefficients
    
    3. Group Comparisons:
       - Compare preferences across career stages (use specific statistics)
       - Compare preferences across disciplines (use specific statistics)
       - Compare preferences across gender groups (use specific statistics)
    
    4. Experiment Analysis:
       - Compare outcomes between control and treatment groups in Experiment 1
       - Compare outcomes between high and low prestige conditions in Experiment 2
       - Calculate effect sizes, if applicable
    
    5. Key Findings and Patterns:
       - Identify the top 3-5 most significant findings with specific numbers
       - Note any unexpected or surprising patterns in the data
    
    Your analysis should include specific numerical results (means, standard deviations, correlation coefficients, p-values where appropriate). Provide tables of results where helpful. This is not a theoretical exercise - actually analyze the data provided.
    """
    
    print(f"Performing data analysis for Paper #{paper_number}...")
    
    # Call GPT to analyze the data
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a skilled data analyst specializing in research data. Always provide specific numerical results and detailed analysis of actual data, not theoretical instructions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    analysis = response.choices[0].message.content
    
    # Save the analysis
    output_path = paper_dir / "data_analysis.md"
    with open(output_path, "w") as f:
        f.write(analysis)
    
    print(f"Data analysis saved to {output_path}")
    return analysis

def review_methodology(paper_number, analysis):
    """Review research methodology."""
    paper_dir = BASE_DIR / f"Paper #{paper_number}"
    data_dir = paper_dir / "Data and Code"
    
    # Read the survey design document if it exists
    survey_design = ""
    survey_design_path = data_dir / "survey_design.txt"
    if survey_design_path.exists():
        with open(survey_design_path, "r") as f:
            survey_design = f.read()
    
    # Create the prompt for methodology review
    prompt = f"""
    You are a Methodology Expert reviewing the research design for Paper #{paper_number}.
    
    Review the following survey design:
    ```
    {survey_design}
    ```
    
    And the following data analysis:
    ```
    {analysis}
    ```
    
    Your task is to conduct a comprehensive review of the methodology, focusing on concrete, specific aspects:
    
    1. Research Questions and Hypotheses:
       - Evaluate if the research questions are clearly articulated and appropriately focused 
       - Assess if the research questions are answerable with the collected data
       - Recommend specific improvements to research questions
    
    2. Sampling Strategy:
       - Analyze the specific sampling method used (e.g., stratified random, convenience)
       - Calculate potential sampling errors and confidence intervals where possible
       - Evaluate sample size adequacy for the statistical tests conducted
       - Assess potential sampling biases and their impact on validity
    
    3. Survey Instrument Validity and Reliability:
       - Evaluate construct validity - do the questions measure what they claim to?
       - Assess content validity - do the questions cover all relevant aspects of the topic?
       - Calculate (or estimate) reliability coefficients (e.g., Cronbach's alpha) for scale items
       - Identify specific items that may have validity or reliability issues
    
    4. Data Collection Procedures:
       - Evaluate the time frame and potential temporal biases
       - Assess the response rate and potential non-response bias
       - Evaluate the data collection method (online, in-person, etc.) and associated biases
    
    5. Data Analysis Methods:
       - Evaluate the appropriateness of statistical tests used
       - Assess if assumptions for these tests were met
       - Identify any missing or additional analyses that should be conducted
       - Suggest specific alternative analyses if applicable
    
    6. Ethical Considerations:
       - Evaluate informed consent procedures
       - Assess data privacy and confidentiality measures
       - Identify any ethical concerns with the methodology
    
    For each area, provide:
    1. A specific assessment (not vague generalities)
    2. Concrete examples from the survey design and data analysis
    3. Specific recommendations for improvement
    4. A severity rating (Low, Medium, High) for each issue identified
    
    Your review should be thorough, specific, and actionable - focusing on the actual methodology used rather than general principles.
    """
    
    print(f"Reviewing methodology for Paper #{paper_number}...")
    
    # Call GPT to review the methodology
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a research methodologist with expertise in survey design. Provide concrete, specific methodological assessments based on actual research designs, not general principles."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    review = response.choices[0].message.content
    
    # Save the review
    output_path = paper_dir / "methodology_review.md"
    with open(output_path, "w") as f:
        f.write(review)
    
    print(f"Methodology review saved to {output_path}")
    return review

def create_report_outline(paper_number, analysis, review):
    """Create an outline for the research report."""
    paper_dir = BASE_DIR / f"Paper #{paper_number}"
    
    # Create the prompt for report outline
    prompt = f"""
    You are a Research Writer creating a detailed outline for a research report based on OSF journal survey data.
    
    You have access to:
    
    1. Data analysis:
    ```
    {analysis}
    ```
    
    2. Methodology review:
    ```
    {review}
    ```
    
    Your task is to create a comprehensive, detailed outline for a scholarly research report. The outline should:
    
    1. Follow standard academic paper structure with all major sections and subsections
    2. Include specific bullet points under each section detailing exactly what content will be covered
    3. Highlight key statistical findings that should be reported in each section
    4. Note which methodological considerations should be addressed
    5. Specify where tables, figures, or visualizations should be included
    
    The outline must include these sections:
    
    1. Title (suggest a specific, descriptive title)
    
    2. Abstract (bullet points for what should be included)
    
    3. Introduction
       - Background literature review (specific topics to cover)
       - Problem statement
       - Research questions/hypotheses (be specific about each question)
       - Significance of the study
    
    4. Methods
       - Participants (demographics, sampling approach)
       - Materials/Instruments (survey details, scales used)
       - Procedure (data collection, timeline, ethical considerations)
       - Data Analysis Approach (specific statistical tests)
    
    5. Results
       - Descriptive Statistics (which specific statistics to report)
       - Inferential Statistics (which analyses and findings to present)
       - Tables and Figures (describe what each table/figure should show)
    
    6. Discussion
       - Interpretation of Key Findings (for each major finding)
       - Comparison with Existing Literature
       - Limitations (methodological issues identified in the review)
       - Implications (theoretical and practical)
    
    7. Conclusion
       - Summary of Findings
       - Recommendations
       - Future Research Directions
    
    8. References (note types of sources to include)
    
    9. Appendices (what should be included)
    
    For each section, include:
    - At least 3-5 detailed bullet points specifying content
    - Notes on what statistical data should be presented
    - References to specific findings from the data analysis
    - Methodological considerations from the review that should be addressed
    
    This outline will serve as the comprehensive blueprint for writing the actual paper, so be thorough and specific.
    """
    
    print(f"Creating report outline for Paper #{paper_number}...")
    
    # Call GPT to create the outline
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an experienced academic writer who specializes in creating detailed, comprehensive research paper outlines. Your outlines are specific, thorough, and provide clear direction for paper writing."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=3000
    )
    
    outline = response.choices[0].message.content
    
    # Save the outline
    output_path = paper_dir / "report_outline.md"
    with open(output_path, "w") as f:
        f.write(outline)
    
    print(f"Report outline saved to {output_path}")
    return outline

def write_report_draft(paper_number, outline, analysis, review):
    """Write a draft of the research report."""
    paper_dir = BASE_DIR / f"Paper #{paper_number}"
    
    # Create the prompt for report draft
    prompt = f"""
    You are a Research Writer drafting a detailed research report based on OSF journal survey data.
    
    You have:
    
    1. Report outline:
    ```
    {outline}
    ```
    
    2. Data analysis:
    ```
    {analysis}
    ```
    
    3. Methodology review:
    ```
    {review}
    ```
    
    Your task is to write a comprehensive, detailed academic research report that:
    
    1. Follows the structure in the outline precisely
    2. Incorporates ALL specific numerical findings from the data analysis
    3. Addresses ALL methodological considerations from the review
    4. Cites relevant literature throughout (create appropriate citations where needed)
    5. Includes detailed tables and describes specific visualizations where relevant
    
    Specific requirements:
    
    - Introduction: Include at least 3 paragraphs with relevant background literature and clearly articulated research questions
    - Methods: Provide detailed subsections on participants, materials, and procedures (at least 500 words total)
    - Results: Report specific statistical findings with exact numbers, p-values, effect sizes, and confidence intervals where appropriate
    - Discussion: Include at least 4 paragraphs interpreting results, discussing limitations, and suggesting implications
    - Conclusion: Summarize key findings and suggest at least 3 specific future research directions
    - Include at least a dozen relevant academic citations throughout
    
    The report should be thorough, detailed, and meet high standards for academic publication. The final product should be at least 2500 words and suitable for submission to a peer-reviewed journal.
    """
    
    print(f"Writing report draft for Paper #{paper_number}...")
    
    # Call GPT to write the draft
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an academic writer with experience in publishing journal articles. Produce detailed, comprehensive reports with specific statistical findings and thorough academic content."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=4000
    )
    
    draft = response.choices[0].message.content
    
    # Save the draft
    output_path = paper_dir / "report_draft.md"
    with open(output_path, "w") as f:
        f.write(draft)
    
    print(f"Report draft saved to {output_path}")
    return draft

def review_report(paper_number, draft):
    """Review the report draft."""
    paper_dir = BASE_DIR / f"Paper #{paper_number}"
    
    # Load the original data analysis and methodology review for context
    analysis_path = paper_dir / "data_analysis.md"
    review_path = paper_dir / "methodology_review.md"
    
    with open(analysis_path, "r") as f:
        analysis = f.read()
    
    with open(review_path, "r") as f:
        methodology_review = f.read()
    
    # Create the prompt for report review
    prompt = f"""
    You are a Methodology Expert reviewing a draft research report based on OSF journal survey data.
    
    The draft report:
    ```
    {draft}
    ```
    
    Original data analysis:
    ```
    {analysis}
    ```
    
    Original methodology review:
    ```
    {methodology_review}
    ```
    
    Your task is to conduct a thorough, critical review of this report draft focusing on methodological rigor, accurate reporting of findings, and scholarly writing. Provide detailed feedback on:
    
    1. Overall Quality and Structure (organization, coherence, flow)
       - Evaluate each major section (intro, methods, results, discussion, conclusion)
       - Identify specific areas where structure could be improved
    
    2. Methodological Accuracy and Completeness
       - Verify that all methodological details are correctly reported
       - Identify any missing methodological information
       - Assess if statistical analyses are appropriately described and interpreted
       - Check if appropriate statistical tests were used and reported correctly
    
    3. Results Reporting
       - Verify that all major findings from the data analysis are included
       - Check for accuracy of reported statistics, p-values, effect sizes
       - Evaluate completeness of results reporting
       - Assess if tables/figures are properly described and necessary
    
    4. Discussion Quality
       - Evaluate if interpretations are justified by the actual results
       - Assess if limitations are thoroughly addressed
       - Check if implications are reasonable and not overstated
    
    5. Citation and Academic Writing
       - Evaluate if appropriate literature is cited
       - Check for academic writing style and terminology usage
       - Identify any areas of imprecise language
    
    6. Ethical Considerations
       - Check if ethical procedures are adequately described
       - Evaluate if privacy and confidentiality issues are addressed
    
    For each area, provide:
    - Specific examples from the text (with quotes where appropriate)
    - Concrete suggestions for improvement
    - A priority level for each issue (High/Medium/Low)
    
    Focus on substantive issues rather than minor stylistic concerns. Be specific, detailed, and actionable in your feedback.
    """
    
    print(f"Reviewing report for Paper #{paper_number}...")
    
    # Call GPT to review the report
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a research methodologist and academic editor with expertise in reviewing scholarly papers. Be thorough, specific, and critical in your review, focusing on substantive methodological and reporting issues."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=3000
    )
    
    feedback = response.choices[0].message.content
    
    # Save the feedback
    output_path = paper_dir / "report_feedback.md"
    with open(output_path, "w") as f:
        f.write(feedback)
    
    print(f"Report feedback saved to {output_path}")
    return feedback

def finalize_report(paper_number, draft, feedback):
    """Finalize the research report."""
    paper_dir = BASE_DIR / f"Paper #{paper_number}"
    
    # Get the data analysis and methodology review for additional context
    analysis_path = paper_dir / "data_analysis.md"
    review_path = paper_dir / "methodology_review.md"
    
    with open(analysis_path, "r") as f:
        analysis = f.read()
    
    with open(review_path, "r") as f:
        review = f.read()
    
    # Create the prompt for finalizing the report
    prompt = f"""
    You are a Research Writer finalizing a comprehensive research report based on OSF journal survey data.
    
    You have:
    
    1. Draft report:
    ```
    {draft}
    ```
    
    2. Review feedback:
    ```
    {feedback}
    ```
    
    3. Original data analysis:
    ```
    {analysis}
    ```
    
    4. Original methodology review:
    ```
    {review}
    ```
    
    Your task is to produce an exceptional, publication-ready final report that:
    
    1. Addresses ALL feedback and suggestions from the review in detail
    2. Significantly enhances the draft by adding more depth, rigor, and detail
    3. Incorporates ALL key statistical findings from the data analysis with precise reporting
    4. Features a comprehensive methods section that addresses methodological concerns
    5. Includes detailed tables for reporting results 
    6. Describes visualizations that would enhance understanding (as if they were included)
    7. Contains a sophisticated discussion that thoroughly examines implications
    8. Provides a detailed conclusion with specific recommendations
    
    Specific requirements for the final report:
    
    - Length: At least 3500 words
    - Citations: Include at least 15-20 relevant academic references
    - Results section: Report all statistical findings with precise values, confidence intervals, effect sizes, and p-values
    - Methods section: Include detailed subsections on participants, materials, procedures, and data analysis approach
    - Discussion: At least 1000 words with thorough examination of findings, limitations, and implications
    - Use formal academic language and structure throughout
    - Include detailed demographic information about participants
    - Address ethical considerations thoroughly
    
    This should be an exemplary academic paper suitable for submission to a high-quality peer-reviewed journal.
    """
    
    print(f"Finalizing report for Paper #{paper_number}...")
    
    # Call GPT to finalize the report
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an academic writer with experience producing thorough, detailed, and rigorous research reports for prestigious journals. Your reports are comprehensive, precise, and meticulously detailed."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=4000  # Reduced from 4500 to stay within limits
    )
    
    final_report = response.choices[0].message.content
    
    # Save the final report
    output_path = paper_dir / "final_report.md"
    with open(output_path, "w") as f:
        f.write(final_report)
    
    print(f"Final report saved to {output_path}")
    return final_report

def main(paper_number: int):
    """Main function to run the journal report generation process."""
    paper_dir = BASE_DIR / f"Paper #{paper_number}"
    paper_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Starting report generation process for Paper #{paper_number} ===\n")
    
    # Run the pipeline
    try:
        # Step 1: Perform data analysis
        print("\n--- Step 1: Data Analysis ---")
        analysis = analyze_data(paper_number)
        time.sleep(2)  # Pause to avoid rate limiting
        
        # Step 2: Review methodology
        print("\n--- Step 2: Methodology Review ---")
        review = review_methodology(paper_number, analysis)
        time.sleep(2)
        
        # Step 3: Create report outline
        print("\n--- Step 3: Report Outline Creation ---")
        outline = create_report_outline(paper_number, analysis, review)
        time.sleep(2)
        
        # Step 4: Write report draft
        print("\n--- Step 4: Report Draft Writing ---")
        draft = write_report_draft(paper_number, outline, analysis, review)
        time.sleep(2)
        
        # Step 5: Review report
        print("\n--- Step 5: Report Review ---")
        feedback = review_report(paper_number, draft)
        time.sleep(2)
        
        # Step 6: Finalize report
        print("\n--- Step 6: Report Finalization ---")
        final_report = finalize_report(paper_number, draft, feedback)
        
        print(f"\n=== Report generation complete for Paper #{paper_number}! ===")
        print(f"All outputs saved in: {paper_dir}\n")
        print(f"Files generated:")
        print(f"- {paper_dir}/data_analysis.md (Step 1)")
        print(f"- {paper_dir}/methodology_review.md (Step 2)")
        print(f"- {paper_dir}/report_outline.md (Step 3)")
        print(f"- {paper_dir}/report_draft.md (Step 4)")
        print(f"- {paper_dir}/report_feedback.md (Step 5)")
        print(f"- {paper_dir}/final_report.md (Final Output)")
        
    except Exception as e:
        print(f"Error in report generation process: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JCR Journal Report Agent')
    parser.add_argument('--paper_number', type=int, required=True, help='Paper number to analyze')
    args = parser.parse_args()
    
    main(args.paper_number)