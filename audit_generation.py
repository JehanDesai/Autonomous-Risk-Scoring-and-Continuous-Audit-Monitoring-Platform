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
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                return response.text
                    
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return f"Error generating LLM insights: {str(e)}\n\nDetails: {error_details}"
    
    def generate_html_report(self, data: pd.DataFrame, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, report_name: Optional[str] = None) -> str:
        # Generate an HTML audit report with visualizations and LLM insights.
        if start_date is None:
            start_date = data['date'].min()
        if end_date is None:
            end_date = data['date'].max()
            
        # Filter data for the specified period
        period_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        
        # Get high-risk transactions
        high_risk = period_data[period_data['risk_score'] >= 80].sort_values('risk_score', ascending=False)
        
        # Generate plots
        plots = self._generate_plots(period_data)
        
        # Get LLM insights on anomalies
        if not high_risk.empty:
            llm_insights = self._get_llm_insights(high_risk.head(10))
        else:
            llm_insights = "No high-risk transactions identified during this period."
        
        # Prepare report statistics
        stats = {
            'total_transactions': len(period_data),
            'total_amount': period_data['amount'].sum(),
            'avg_risk_score': period_data['risk_score'].mean(),
            'high_risk_count': len(high_risk),
            'high_risk_amount': high_risk['amount'].sum(),
            'period_start': start_date.strftime('%Y-%m-%d'),
            'period_end': end_date.strftime('%Y-%m-%d'),
        }
        
        # Generate HTML report
        report_date = datetime.now().strftime('%Y-%m-%d')
        if report_name is None:
            report_name = f"audit_report_{report_date}"
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Financial Audit Report - {report_date}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .header {{ background-color: #003366; color: white; padding: 20px; text-align: center; }}
                .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .summary-box {{ background-color: #f7f7f7; border-radius: 5px; padding: 15px; width: 22%; text-align: center; }}
                .insights {{ background-color: #f7f7f7; padding: 20px; margin: 20px 0; border-radius: 5px; }}
                .visualization {{ margin: 30px 0; text-align: center; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #003366; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .risk-very-high {{ background-color: #ffcccc; }}
                .risk-high {{ background-color: #ffddcc; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Financial Transactions Audit Report</h1>
                <p>Period: {stats['period_start']} to {stats['period_end']}</p>
                <p>Generated on: {report_date}</p>
            </div>
            
            <div class="summary">
                <div class="summary-box">
                    <h3>Total Transactions</h3>
                    <p>{stats['total_transactions']:,}</p>
                </div>
                <div class="summary-box">
                    <h3>Total Amount</h3>
                    <p>${stats['total_amount']:,.2f}</p>
                </div>
                <div class="summary-box">
                    <h3>High Risk Transactions</h3>
                    <p>{stats['high_risk_count']:,} ({stats['high_risk_count']/stats['total_transactions']*100:.1f}%)</p>
                </div>
                <div class="summary-box">
                    <h3>High Risk Amount</h3>
                    <p>${stats['high_risk_amount']:,.2f}</p>
                </div>
            </div>
            
            <div class="insights">
                <h2>AI-Generated Insights</h2>
                <div style="white-space: pre-line">{llm_insights}</div>
            </div>
            
            <div class="visualization">
                <h2>Risk Score Distribution</h2>
                <img src="data:image/png;base64,{plots['risk_distribution']}" alt="Risk Score Distribution">
            </div>
            
            <div class="visualization">
                <h2>Average Risk by Department</h2>
                <img src="data:image/png;base64,{plots['dept_risk']}" alt="Risk by Department">
            </div>
            
            <div class="visualization">
                <h2>Average Risk by Vendor (Top 10)</h2>
                <img src="data:image/png;base64,{plots['vendor_risk']}" alt="Risk by Vendor">
            </div>
            
            <h2>High Risk Transactions</h2>
            <table>
                <tr>
                    <th>Transaction ID</th>
                    <th>Date</th>
                    <th>Amount</th>
                    <th>Department</th>
                    <th>Vendor</th>
                    <th>Employee</th>
                    <th>Risk Score</th>
                </tr>
        """
        
        # Add high risk transactions to the table
        for _, row in high_risk.head(20).iterrows():
            risk_class = "risk-very-high" if row['risk_score'] > 90 else "risk-high"
            html += f"""
                <tr class="{risk_class}">
                    <td>{row['transaction_id']}</td>
                    <td>{row['date'].strftime('%Y-%m-%d')}</td>
                    <td>${row['amount']:,.2f}</td>
                    <td>{row['department']}</td>
                    <td>{row['vendor']}</td>
                    <td>{row['employee']}</td>
                    <td>{row['risk_score']:.1f}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <div style="text-align: center; margin-top: 50px;">
                <p>This report was automatically generated by the Autonomous Audit Agent</p>
            </div>
        </body>
        </html>
        """
        
        # Save the report
        report_path = self.report_dir / f"{report_name}.html"
        with open(report_path, 'w') as f:
            f.write(html)
            
        return str(report_path)
    
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