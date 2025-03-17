import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import schedule
import time
import requests
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("audit_agent.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('AuditAgent')

class LLMIntegration:
    def __init__(self, model_name="openai/gpt-4", api_key=None):
        """
        Initialize LLM integration for generating narrative insights
        
        Parameters:
        model_name (str): Name of the LLM model to use
        api_key (str): API key for accessing the LLM service
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        
        # Validate we have what we need
        if self.api_key is None and "openai" in model_name.lower():
            logger.warning("OpenAI API key not provided. Set LLM_API_KEY environment variable.")
    
    def generate_insights(self, flagged_transactions, risk_factors, summary_stats):
        """
        Generate narrative insights for flagged transactions using LLM
        
        Parameters:
        flagged_transactions (pd.DataFrame): DataFrame of flagged transactions
        risk_factors (dict): Risk factors for each flagged transaction
        summary_stats (dict): Summary statistics for the audit period
        
        Returns:
        str: Narrative insights about the anomalies
        """
        if len(flagged_transactions) == 0:
            return "No anomalies detected in the analyzed transactions."
        
        # Prepare the prompt for the LLM
        prompt = self._prepare_prompt(flagged_transactions, risk_factors, summary_stats)
        
        if "openai" in self.model_name.lower():
            return self._query_openai(prompt)
        elif "llama" in self.model_name.lower():
            return self._query_llama(prompt)
        else:
            # Default to a simplified approach if no LLM is available
            return self._generate_basic_insights(flagged_transactions, risk_factors, summary_stats)
    
    def _prepare_prompt(self, flagged_transactions, risk_factors, summary_stats):
        """Prepare a prompt for the LLM based on the flagged transactions"""
        transaction_examples = []
        
        # Add up to 5 example transactions with their risk factors
        for i, (idx, tx) in enumerate(flagged_transactions.iterrows()):
            if i >= 5:  # Limit to 5 examples to keep prompt size reasonable
                break
                
            tx_id = tx.get('transaction_id', f"TX{i}")
            tx_type = tx.get('transaction_type', 'Unknown')
            amount = tx.get('amount', 0)
            risk_score = tx.get('risk_score', 0)
            
            tx_factors = risk_factors.get(tx_id, [])
            factor_text = "\n".join([f"- {f['factor']}: {f['details']}" for f in tx_factors])
            
            transaction_examples.append(
                f"Transaction {tx_id} ({tx_type}, ${amount:,.2f}, Risk Score: {risk_score:.2f}):\n{factor_text}"
            )
        
        # Format summary statistics
        stats_text = "\n".join([f"- {k}: {v}" for k, v in summary_stats.items()])
        
        # Build the full prompt
        prompt = f"""
        As a financial auditor, analyze these flagged transactions and provide insights.
        
        SUMMARY STATISTICS:
        {stats_text}
        
        FLAGGED TRANSACTIONS:
        {chr(10).join(transaction_examples)}
        
        Based on the above information, provide:
        1. A summary of the key risk patterns detected
        2. Potential explanations for these anomalies
        3. Recommended follow-up actions for the audit team
        4. Potential improvements to internal controls
        
        Format your response in clear, professional language suitable for an audit report.
        """
        
        return prompt
    
    def _query_openai(self, prompt):
        """Query OpenAI API for generating insights"""
        try:
            import openai
            openai.api_key = self.api_key
            
            response = openai.chat.completions.create(
                model="gpt-4", 
                messages=[
                    {"role": "system", "content": "You are a financial audit expert analyzing transaction anomalies."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error querying OpenAI: {str(e)}")
            return self._generate_basic_insights(None, None, None)
    
    def _query_llama(self, prompt):
        """Query Llama model for generating insights"""
        try:
            # Example implementation using an API endpoint
            API_URL = f"https://api-inference.huggingface.co/models/{self.model_name}"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 1000,
                    "temperature": 0.3
                }
            }
            
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()[0]["generated_text"]
            
        except Exception as e:
            logger.error(f"Error querying Llama: {str(e)}")
            return self._generate_basic_insights(None, None, None)
    
    def _generate_basic_insights(self, flagged_transactions, risk_factors, summary_stats):
        """Generate basic insights without using an LLM"""
        # Fallback when no LLM is available or there's an error
        insights = []
        insights.append("# Financial Audit Insights")
        insights.append("\n## Key Risk Patterns Detected")
        
        if flagged_transactions is not None and not flagged_transactions.empty:
            num_flagged = len(flagged_transactions)
            avg_risk = flagged_transactions['risk_score'].mean() if 'risk_score' in flagged_transactions.columns else 0
            
            insights.append(f"\nDetected {num_flagged} potentially suspicious transactions with an average risk score of {avg_risk:.2f}.")
            
            # Count transaction types
            if 'transaction_type' in flagged_transactions.columns:
                type_counts = flagged_transactions['transaction_type'].value_counts()
                insights.append("\nSuspicious transactions by type:")
                for tx_type, count in type_counts.items():
                    insights.append(f"- {tx_type}: {count}")
            
            # Summarize risk factors
            if risk_factors:
                factor_types = {}
                for tx_id, factors in risk_factors.items():
                    for factor in factors:
                        factor_name = factor['factor']
                        factor_types[factor_name] = factor_types.get(factor_name, 0) + 1
                
                insights.append("\nCommon risk factors:")
                for factor, count in sorted(factor_types.items(), key=lambda x: x[1], reverse=True):
                    insights.append(f"- {factor}: found in {count} transactions")
        
        insights.append("\n## Recommended Actions")
        insights.append("\n1. Review all flagged transactions, starting with those having the highest risk scores")
        insights.append("2. Investigate unusual transaction patterns, particularly those occurring outside normal business hours")
        insights.append("3. Review approval processes for high-value transactions")
        insights.append("4. Consider additional controls around transactions with round number amounts")
        
        return "\n".join(insights)

class AuditReport:
    def __init__(self, report_id=None, period_start=None, period_end=None):
        """
        Initialize an audit report
        
        Parameters:
        report_id (str): Unique identifier for the report
        period_start (datetime): Start of the audit period
        period_end (datetime): End of the audit period
        """
        self.report_id = report_id or f"AR{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.period_start = period_start or datetime.now() - timedelta(days=7)
        self.period_end = period_end or datetime.now()
        self.creation_date = datetime.now()
        self.transactions = pd.DataFrame()
        self.flagged_transactions = pd.DataFrame()
        self.risk_factors = {}
        self.summary_stats = {}
        self.narrative_insights = ""
        
    def add_transactions(self, transactions_df):
        """Add transactions to the report"""
        self.transactions = transactions_df
        
    def add_flagged_transactions(self, flagged_df):
        """Add flagged transactions to the report"""
        self.flagged_transactions = flagged_df
        
    def add_risk_factors(self, risk_factors):
        """Add risk factors to the report"""
        self.risk_factors = risk_factors
        
    def add_summary_stats(self, summary_stats):
        """Add summary statistics to the report"""
        self.summary_stats = summary_stats
        
    def add_narrative_insights(self, insights):
        """Add narrative insights to the report"""
        self.narrative_insights = insights
        
    def to_dict(self):
        """Convert report to dictionary"""
        return {
            'report_id': self.report_id,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'creation_date': self.creation_date.isoformat(),
            'summary_stats': self.summary_stats,
            'flagged_transactions_count': len(self.flagged_transactions),
            'narrative_insights': self.narrative_insights
        }
        
    def to_markdown(self):
        """Generate a markdown representation of the report"""
        md = []
        md.append(f"# Audit Report {self.report_id}")
        md.append(f"\nGenerated on: {self.creation_date.strftime('%Y-%m-%d %H:%M:%S')}")
        md.append(f"\nPeriod: {self.period_start.strftime('%Y-%m-%d')} to {self.period_end.strftime('%Y-%m-%d')}")
        
        md.append("\n## Summary")
        md.append(f"\nTotal Transactions: {len(self.transactions)}")
        md.append(f"Flagged Transactions: {len(self.flagged_transactions)}")
        
        md.append("\n## Statistics")
        for key, value in self.summary_stats.items():
            md.append(f"\n- {key}: {value}")
        
        md.append("\n## Risk Analysis Insights")
        md.append(f"\n{self.narrative_insights}")
        
        md.append("\n## Flagged Transactions")
        if not self.flagged_transactions.empty:
            # Convert top 10 flagged transactions to markdown table
            top_flagged = self.flagged_transactions.sort_values('risk_score', ascending=False).head(10)
            
            cols_to_show = ['transaction_id', 'transaction_type', 'amount', 'risk_score', 'risk_level']
            cols_to_show = [col for col in cols_to_show if col in top_flagged.columns]
            
            md.append("\n")
            md.append("| " + " | ".join(cols_to_show) + " |")
            md.append("| " + " | ".join(["---" for _ in cols_to_show]) + " |")
            
            for _, row in top_flagged.iterrows():
                values = []
                for col in cols_to_show:
                    val = row.get(col, '')
                    if col == 'amount' and isinstance(val, (int, float)):
                        val = f"${val:,.2f}"
                    elif col == 'risk_score' and isinstance(val, float):
                        val = f"{val:.2f}"
                    values.append(str(val))
                md.append("| " + " | ".join(values) + " |")
        else:
            md.append("\nNo transactions were flagged in this period.")
        
        return "\n".join(md)
        
    def save(self, output_dir="reports"):
        """Save the report to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        json_path = os.path.join(output_dir, f"{self.report_id}.json")
        with open(json_path, 'w') as f:
            # Convert DataFrames to dictionaries for JSON serialization
            report_data = self.to_dict()
            report_data['transactions'] = self.transactions.to_dict('records') if not self.transactions.empty else []
            report_data['flagged_transactions'] = self.flagged_transactions.to_dict('records') if not self.flagged_transactions.empty else []
            report_data['risk_factors'] = self.risk_factors
            
            json.dump(report_data, f, indent=2, default=str)
        
        # Save as Markdown
        md_path = os.path.join(output_dir, f"{self.report_id}.md")
        with open(md_path, 'w') as f:
            f.write(self.to_markdown())
          
        return {'json': json_path, 'markdown': md_path}