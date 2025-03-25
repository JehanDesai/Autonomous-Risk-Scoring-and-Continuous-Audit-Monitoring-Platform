import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import schedule
import threading
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import streamlit as st

class AuditAgent:
    # Autonomous agent for continuous monitoring of financial transactions.
    # Integrates data connector, risk scoring model, and report generation.
    
    def __init__(self, data_connector, risk_model, report_generator, config: Dict = None):
        self.data_connector = data_connector
        self.risk_model = risk_model
        self.report_generator = report_generator
        self.config = config or {}
        self.data_dir = Path("audit_data")
        self.data_dir.mkdir(exist_ok=True)
        self.processed_data = None
        self.schedule_thread = None
        self.running = False
        
    def get_monitoring_status(self):
        """Return the current monitoring status."""
        return self.running

    def set_monitoring_config(self, config):
        """Update the monitoring configuration."""
        self.config.update(config)
        return True

    def get_departments(self):
        """Get list of available departments from the data.
        This needs to be added since it's called in the dashboard."""
        if self.processed_data is not None and 'department' in self.processed_data.columns:
            return self.processed_data['department'].unique().tolist()
        return ["Sales", "Marketing", "Finance", "IT", "HR"]  # Default departments if no data

    def process_transactions(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, save_data: bool = True) -> pd.DataFrame:
        # Process transactions from the data source and assign risk scores.
        if "processed_data" in st.session_state and st.session_state["processed_data"] is not None:
            print("Using cached processed data from session state.")
            return st.session_state["processed_data"]
        # Extract transactions
        transactions = self.data_connector.extract_transactions(start_date=start_date, end_date=end_date,limit=self.config.get('transaction_limit', 10000))
        model_path = self.data_dir / "risk_model.joblib"
        if model_path.exists():
            print("Loading existing risk model...")
            self.risk_model.load_model(str(model_path))
        # If the model is not yet trained we train it
        if not hasattr(self.risk_model, 'model') or self.risk_model.model is None:
            print("Training risk model...")
            self.risk_model.train(transactions)
            # Save the trained model
            if self.config.get('save_model', True):
                model_path = self.data_dir / "risk_model.joblib"
                self.risk_model.save_model(str(model_path))
                print(f"Risk model saved to {model_path}")
        # Assign risk scores
        scored_transactions = self.risk_model.predict_risk_scores(transactions)
        # Save processed data if requested
        if save_data:
            data_path = self.data_dir / "processed_transactions_latest.pkl"
            scored_transactions.to_pickle(data_path)
            if not data_path.exists():
                print(f"Error: Processed data file was not saved properly at {data_path}")
            print(f"Processed transactions saved to {data_path}")
        self.processed_data = scored_transactions
        st.session_state["audit_config"] = {"start_date": start_date,"end_date": end_date,"rules": self.config.get('audit_rules', {})}
        return scored_transactions
    
    def generate_report(self, data: Optional[pd.DataFrame] = None, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> str:
        #Generate an audit report.
        if data is None:
            if self.processed_data is None:
                raise ValueError("No data available. Process transactions first.")
            data = self.processed_data
        report_path = self.data_dir / "latest_audit_report.json"
        # Save the data in JSON format for consistency
        data.to_json(report_path, orient="records", date_format="iso")
        print(f"Audit report generated: {report_path}")
        return report_path
    
    def _weekly_audit(self):
        # Perform weekly audit tasks
        print(f"Starting weekly audit at {datetime.now()}")
        # Calculate date range for the past week
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        try:
            # Process transactions
            transactions = self.process_transactions(start_date=start_date,end_date=end_date,save_data=True)
            # Generate report
            report_path = self.generate_report(data=transactions,start_date=start_date,end_date=end_date)
            # Flag high risk transactions
            high_risk = transactions[transactions['risk_score'] >= 90]
            flags = [
                {
                    'transaction_id': row['transaction_id'],
                    'flag_type': 'High Risk',
                    'comment': f"Flagged by weekly audit with risk score {row['risk_score']:.1f}"
                }
                for _, row in high_risk.iterrows()
            ]
            flagged_transactions = self.report_generator.flag_transactions(transactions, flags)
            # Save flagged transactions
            timestamp = datetime.now().strftime('%Y%m%d')
            flagged_path = self.data_dir / f"flagged_transactions_{timestamp}.pkl"
            flagged_transactions.to_pickle(flagged_path)
            print(f"Weekly audit completed. Flagged {len(flags)} high-risk transactions.")
            print(f"Report generated at {report_path}")
        except Exception as e:
            print(f"Error in weekly audit: {str(e)}")
    
    def start_monitoring(self):
        # Start the autonomous monitoring process
        if self.running:
            print("Monitoring is already running.")
            return
        self.running = True
        # Schedule weekly audit for Friday
        schedule.every().friday.at("09:00").do(self._weekly_audit)
        # Start the scheduler in a separate thread
        def run_scheduler():
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        self.schedule_thread = threading.Thread(target=run_scheduler)
        self.schedule_thread.daemon = True
        self.schedule_thread.start()
        print("Autonomous monitoring started.")
        
    def stop_monitoring(self):
        # Stop the autonomous monitoring process
        if not self.running:
            print("Monitoring is not running.")
            return
        self.running = False
        if self.schedule_thread:
            self.schedule_thread.join(timeout=5)
        print("Autonomous monitoring stopped.")

    def run_manual_audit(self, start_date=None, end_date=None, audit_type="Comprehensive", department=None, risk_threshold=75, sample_size=100, rules=None):
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        print(f"Running {audit_type} audit from {start_date} to {end_date}")
        # Process transactions
        if self.processed_data is not None:
            transactions = self.processed_data
            print("Using existing processed data.")
        else:
            transactions = self.process_transactions(start_date=start_date, end_date=end_date, save_data=True)
            print("Generating new processed data.")
        # Apply department filter if specified
        if department and department != "All Departments":
            if 'department' in transactions.columns:
                transactions = transactions[transactions['department'] == department]
                print(f"Filtered to {department} department: {len(transactions)} transactions")
        # Apply sample size
        if sample_size < 100:
            sample_count = int(len(transactions) * sample_size / 100)
            transactions = transactions.sample(n=min(sample_count, len(transactions)))
            print(f"Sampled {sample_size}% of transactions: {len(transactions)} transactions")
        # Apply rules if specified
        flagged_transactions = pd.DataFrame()
        if rules:
            # Flag round number transactions
            if rules.get('round_numbers'):
                round_amount = transactions[transactions['amount'] % 100 == 0]
                flagged_transactions = pd.concat([flagged_transactions, round_amount])
            # Flag transactions just below approval thresholds
            if rules.get('below_threshold'):
                # Assuming approval thresholds are at 1000, 5000, 10000
                thresholds = [1000, 5000, 10000]
                threshold_margin = 50  # Amount below threshold to flag
                below_threshold = transactions[transactions['amount'].apply(
                    lambda x: any(abs(x - t) < threshold_margin and x < t for t in thresholds)
                )]
                flagged_transactions = pd.concat([flagged_transactions, below_threshold])
            # Flag weekend/holiday transactions
            if rules.get('weekend_holiday'):
                if 'date' in transactions.columns:
                    weekend_txns = transactions[transactions['date'].dt.dayofweek >= 5]
                    flagged_transactions = pd.concat([flagged_transactions, weekend_txns])
            # Flag transactions without proper documentation
            if rules.get('documentation'):
                if 'documentation_complete' in transactions.columns:
                    undocumented = transactions[transactions['documentation_complete'] == False]
                    flagged_transactions = pd.concat([flagged_transactions, undocumented])
            # Apply custom rule if provided
            if rules.get('custom'):
                try:
                    # Use safer eval approach
                    custom_mask = transactions.eval(rules['custom'])
                    custom_flagged = transactions[custom_mask]
                    flagged_transactions = pd.concat([flagged_transactions, custom_flagged])
                except Exception as e:
                    print(f"Error applying custom rule: {str(e)}")
        # Flag high risk transactions based on risk_threshold
        high_risk = transactions[transactions['risk_score'] >= risk_threshold]
        flagged_transactions = pd.concat([flagged_transactions, high_risk])
        # Remove duplicates
        flagged_transactions = flagged_transactions.drop_duplicates()
        # Generate report
        report_path = self.generate_report(data=transactions, start_date=start_date, end_date=end_date)
        print(f"Manual audit completed. Report generated at {report_path}")
        return {
            'total_transactions': len(transactions),
            'flagged_transactions': flagged_transactions,
            'total_risk_amount': flagged_transactions['amount'].sum() if 'amount' in flagged_transactions.columns else 0,
            'audit_type': audit_type,
            'department': department,
            'risk_threshold': risk_threshold,
            'sample_size': sample_size,
            'findings': []  # Add basic findings structure
        }