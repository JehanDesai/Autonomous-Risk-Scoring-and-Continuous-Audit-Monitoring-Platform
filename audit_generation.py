import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from datetime import datetime
import os
from typing import Dict, List, Optional, Union, Tuple
import openai
from pathlib import Path
from google import genai
from dotenv import load_dotenv

load_dotenv()

class AuditReportGenerator:
    # Generates audit reports based on transaction data and risk scores.
    # Integrates with LLMs to provide narrative insights about anomalies.
    def __init__(self, llm_provider: str = "google-gemini", api_key: Optional[str] = None, llm_config: Optional[Dict] = None):
        self.llm_provider = llm_provider
        self.api_key = api_key or os.environ.get(f"{llm_provider.upper()}_API_KEY", "")
        self.llm_config = llm_config or {}
        self.report_dir = Path("audit_reports")
        self.report_dir.mkdir(exist_ok=True)
        # Configure LLM client
        if llm_provider == "openai" and self.api_key:
            openai.api_key = self.api_key
        
    def _generate_plots(self, data: pd.DataFrame) -> Dict[str, str]:
        # Generate plots for the audit report.
        plots = {}
        # Set style
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({'font.size': 12})
        # Risk distribution plot
        plt.figure(figsize=(10, 6))
        sns.histplot(data['risk_score'], bins=20, kde=True)
        plt.title('Distribution of Risk Scores')
        plt.xlabel('Risk Score')
        plt.ylabel('Frequency')
        plt.axvline(x=80, color='r', linestyle='--', label='High Risk Threshold')
        plt.legend()
        # Save to bytes buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plots['risk_distribution'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        # Risk by department
        plt.figure(figsize=(12, 6))
        dept_risk = data.groupby('department')['risk_score'].mean().sort_values(ascending=False)
        sns.barplot(x=dept_risk.index, y=dept_risk.values)
        plt.title('Average Risk Score by Department')
        plt.xlabel('Department')
        plt.ylabel('Average Risk Score')
        plt.xticks(rotation=45)
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plots['dept_risk'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        # Risk by vendor (top 10)
        plt.figure(figsize=(12, 6))
        vendor_risk = data.groupby('vendor')['risk_score'].mean().sort_values(ascending=False).head(10)
        sns.barplot(x=vendor_risk.index, y=vendor_risk.values)
        plt.title('Average Risk Score by Vendor (Top 10)')
        plt.xlabel('Vendor')
        plt.ylabel('Average Risk Score')
        plt.xticks(rotation=45)
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plots['vendor_risk'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return plots
    
    def _get_llm_insights(self, anomalies: pd.DataFrame) -> str:
        # Get narrative insights about anomalies using an LLM.
        # Prepare anomaly data for LLM prompt
        anomaly_summary = []
        for _, row in anomalies.iterrows():
            # Handle date formatting safely whether it's a string or datetime
            date_value = row['date']
            if isinstance(date_value, str):
                formatted_date = date_value  # Use as is if it's already a string
            else:
                # Format it if it's a datetime object
                formatted_date = date_value.strftime('%Y-%m-%d')
            anomaly_summary.append(
                f"Transaction ID: {row['transaction_id']}, "
                f"Amount: ${row['amount']:.2f}, "
                f"Department: {row['department']}, "
                f"Vendor: {row['vendor']}, "
                f"Risk Score: {row['risk_score']:.2f}, "
                f"Date: {formatted_date}, "
                f"Payment Method: {row['payment_method']}"
            )
        anomaly_text = "\n".join(anomaly_summary)
        # Create prompt for the LLM
        prompt = f"""
        You are a forensic financial auditor with expertise in detecting fraud patterns.
        
        Below is a list of financial transactions that have been flagged as potentially anomalous:
        
        {anomaly_text}
        
        Please provide a concise analysis of these transactions, identifying:
        1. The most serious potential issues and why they are concerning
        2. Common patterns among the flagged transactions
        3. Recommendations for further investigation
        4. Possible explanations for the anomalies that might not indicate fraud
        
        Format your response as a professional audit finding.
        """
        try:
            if self.llm_provider == "google-gemini":
                import asyncio
                import nest_asyncio
                # Apply nest_asyncio to allow running asyncio in Streamlit environment
                try:
                    nest_asyncio.apply()
                except:
                    pass
                # Create a new event loop if needed
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                # Create Gemini client
                client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
                # For synchronous operation
                response = client.models.generate_content(model="gemini-2.0-flash",contents=prompt)
                return response.text     
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return f"Error generating LLM insights: {str(e)}\n\nDetails: {error_details}"
    
    def generate_csv_report(self, data: pd.DataFrame, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, report_name: Optional[str] = None) -> str:
        # Generate a CSV report of high-risk transactions.
        if start_date is None:
            start_date = data['date'].min()
        if end_date is None:
            end_date = data['date'].max()
        # Filter data for the specified period
        period_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        # Get high-risk transactions
        high_risk = period_data[period_data['risk_score'] >= 80].sort_values('risk_score', ascending=False)
        # Generate report name
        report_date = datetime.now().strftime('%Y-%m-%d')
        if report_name is None:
            report_name = f"high_risk_transactions_{report_date}"
        # Save the report
        report_path = self.report_dir / f"{report_name}.csv"
        high_risk.to_csv(report_path, index=False)
        return str(report_path)
    
    def flag_transactions(self, data: pd.DataFrame, flags: List[Dict]) -> pd.DataFrame:
        # Flag transactions based on audit findings.
        # Create a copy of the data to avoid modifying the original
        result = data.copy()
        # If 'flag_type' and 'flag_comment' columns don't exist, create them
        if 'flag_type' not in result.columns:
            result['flag_type'] = None
        if 'flag_comment' not in result.columns:
            result['flag_comment'] = None
        # Apply flags
        for flag in flags:
            transaction_id = flag.get('transaction_id')
            if transaction_id in result['transaction_id'].values:
                idx = result[result['transaction_id'] == transaction_id].index
                result.loc[idx, 'flag_type'] = flag.get('flag_type')
                result.loc[idx, 'flag_comment'] = flag.get('comment')
        return result
    
    def save_report_data(self, data: pd.DataFrame, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, report_name: Optional[str] = None) -> str:
       #  Save the report data for future reference. 
        if start_date is None:
            start_date = data['date'].min()
        if end_date is None:
            end_date = data['date'].max()
        # Filter data for the specified period
        period_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        # Generate report name
        report_date = datetime.now().strftime('%Y-%m-%d')
        if report_name is None:
            report_name = f"report_data_{report_date}"
        # Save the data
        report_path = self.report_dir / f"{report_name}.pkl"
        period_data.to_pickle(report_path)
        return str(report_path)