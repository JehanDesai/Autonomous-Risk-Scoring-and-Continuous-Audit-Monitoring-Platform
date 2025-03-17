import pandas as pd
import numpy as np
import datetime
import schedule
import time
import threading
import streamlit as st
from datetime import datetime, timedelta

class AuditAgent:
    def __init__(self, model, llm_integration, risk_threshold=0.7, feature_cols=None): # Add feature_cols as a parameter
        """
        Initialize the Audit Agent.
        
        Parameters:
        model: The trained anomaly detection model.
        llm_integration: The LLM integration for narrative insights.
        risk_threshold: The threshold for flagging transactions as high risk.
        feature_cols: The list of features used for model training.
        """
        self.model = model
        self.llm_integration = llm_integration
        self.risk_threshold = risk_threshold
        self.feature_cols = feature_cols or []  # Initialize as an empty list if not provided
        self.flagged_transactions = pd.DataFrame()
        self.reports = []  # Initialize reports list here
        self.is_running = False
        
    def evaluate_transaction(self, transaction):
        """
        Evaluate a single transaction and assign a risk score
        """
        # Extract features for the model
        transaction_features = self.get_transaction_features(transaction)  # Define this helper function

        # Make prediction
        # Assuming IsolationForest model, using decision_function to get anomaly score
        score = self.model.decision_function([transaction_features])[0] 

        # Create result with risk score
        result = transaction.to_dict()
        result['risk_score'] = float(score)
        result['is_flagged'] = score >= self.risk_threshold

        if result['is_flagged']:
            # Add flagged transactions to a list
            self.flagged_transactions = pd.concat([self.flagged_transactions, pd.DataFrame([result])], ignore_index=True)

        return result
    
    def evaluate_batch(self, transactions_df):
        """
        Evaluate a batch of transactions and assign risk scores.
        
        Args:
            transactions_df: DataFrame of transactions
            
        Returns:
            DataFrame with original transactions and added risk scores
        """
        results = []
        for _, transaction in transactions_df.iterrows():
            results.append(self.evaluate_transaction(transaction))
        
        return pd.DataFrame(results)
        
    def get_transaction_features(self, transaction):
        """
        Extract features from a transaction for model prediction.
        """
        # Select only the features used during training
        features = transaction[self.feature_cols]
        return features.values  # Return as a NumPy array for model input
    

    def _prepare_features(self, transaction):
        """
        Extract and prepare features for model prediction.
        
        Args:
            transaction: Transaction data
            
        Returns:
            Features array for model input
        """
        # Implementation depends on your specific model requirements
        # This is a placeholder - you'll need to extract the relevant features
        # based on your model training
        
        # Example: Extract numerical features and normalize
        numeric_cols = [col for col in transaction.index 
                       if pd.api.types.is_numeric_dtype(transaction[col])]
        
        # Skip any known non-feature columns
        exclude_cols = ['id', 'transaction_id', 'date', 'timestamp']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        return transaction[feature_cols].values
    
    def _add_to_flagged(self, transaction):
        """Add a transaction to the flagged transactions list"""
        transaction_df = pd.DataFrame([transaction])
        self.flagged_transactions = pd.concat([self.flagged_transactions, transaction_df])
    
    def generate_weekly_report(self):
        """
        Generate the weekly audit report and reset flagged transactions.
        
        Returns:
                Report dictionary with summary and details
        """
        report_date = datetime.now()
        
        # Skip if no flagged transactions
        if self.flagged_transactions.empty:
            report = {
                'report_id': f"AR-{report_date.strftime('%Y%m%d')}",
                'date': report_date,
                'status': 'No anomalies detected',
                'summary': 'No transactions were flagged during this period.',
                'flagged_count': 0,
                'details': None,
                # Ensure 'insights' is an empty string when no anomalies are detected.
                'insights': ''  
            }
        else:
            # Generate insights using LLM
            insights = self.llm_integration.generate_insights(self.flagged_transactions)
            
            # Create report
            report = {
                'report_id': f"AR-{report_date.strftime('%Y%m%d')}",
                'date': report_date,
                'status': 'Anomalies detected',
                'summary': f"Detected {len(self.flagged_transactions)} potentially fraudulent transactions.",
                'flagged_count': len(self.flagged_transactions),
                'details': self.flagged_transactions.copy(),
                'insights': insights
            }
                
            # Print notification
            print(f"Audit report generated on {report_date.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Flagged {len(self.flagged_transactions)} transactions")
        
        # Store report
        self.reports.append(report)
        
        # Reset flagged transactions for next period
        self.flagged_transactions = pd.DataFrame()
        
        return report
    
    def _scheduler_job(self):
        """Scheduler function that runs continuously to check for report day"""
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def start_monitoring(self):
        """Start the autonomous monitoring and reporting"""
        if self.is_running:
            print("Monitoring is already active")
            return
        
        # Schedule weekly report generation
        schedule.every().friday.at("17:00").do(self.generate_weekly_report)
        
        # Start the scheduler in a separate thread
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_job)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        print("Autonomous monitoring started. Reports will be generated every Friday at 5:00 PM.")
    
    def stop_monitoring(self):
        """Stop the autonomous monitoring"""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=2)
        schedule.clear()
        print("Autonomous monitoring stopped.")
    
    def get_latest_report(self):
        """Get the most recent audit report"""
        if not self.reports:
            return None
        return self.reports[-1]
    
    def get_all_reports(self):
        """Get all historical audit reports"""
        return self.reports
    
    def create_dashboard(self):
        """
        Create and display an interactive Streamlit dashboard
        
        This method should be called from a Streamlit app
        """
        st.title("Financial Transaction Audit Dashboard")
        
        # Sidebar for filtering and controls
        st.sidebar.header("Controls")
        if st.sidebar.button("Generate Report Now"):
            report = self.generate_weekly_report()
            st.sidebar.success(f"Report generated: {report['report_id']}")
        
        # Report selection
        st.header("Audit Reports")
        if not self.reports:
            st.info("No audit reports have been generated yet.")
        else:
            report_options = [f"{r['report_id']} ({r['date'].strftime('%Y-%m-%d')})" for r in self.reports]
            selected_report_idx = st.selectbox("Select a report:", range(len(report_options)), 
                                             format_func=lambda x: report_options[x])
            
            report = self.reports[selected_report_idx]
            
            # Display report details
            st.subheader(f"Report: {report['report_id']}")
            st.write(f"Status: {report['status']}")
            st.write(f"Date: {report['date'].strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"Summary: {report['summary']}")
            
            # Show insights
            if report['insights']:
                st.subheader("AI-Generated Insights")
                st.write(report['insights'])
            
            # Display flagged transactions if any
            if report['details'] is not None and not report['details'].empty:
                st.subheader("Flagged Transactions")
                st.dataframe(report['details'])
                
                # Add visualization of risk scores
                st.subheader("Risk Score Distribution")
                hist_values = np.histogram(report['details']['risk_score'], bins=10, range=(0, 1))[0]
                st.bar_chart(pd.DataFrame(hist_values))
        
        # Real-time monitoring section
        st.header("Real-time Monitoring")
        st.write("Status: " + ("Active" if self.is_running else "Inactive"))
        
        # Display recent flagged transactions
        if not self.flagged_transactions.empty:
            st.subheader("Recently Flagged Transactions (Current Period)")
            st.dataframe(self.flagged_transactions)