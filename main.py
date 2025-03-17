import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from audit_agent import AuditAgent

# Page configuration
st.set_page_config(
    page_title="Financial Audit Monitor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to generate sample data
def generate_sample_transactions(n=1000):
    np.random.seed(42)
    
    dates = [datetime.now() - timedelta(days=random.randint(0, 30)) for _ in range(n)]
    transaction_ids = [f"TXN-{i+10000}" for i in range(n)]
    departments = np.random.choice(['Finance', 'Marketing', 'Operations', 'IT', 'HR'], n)
    amounts = np.random.lognormal(mean=6, sigma=1, size=n)
    payment_methods = np.random.choice(['Credit Card', 'Wire Transfer', 'Check', 'ACH'], n)
    
    # Inject some anomalies
    anomaly_indices = np.random.choice(n, size=int(n*0.05), replace=False)
    for idx in anomaly_indices:
        amounts[idx] = amounts[idx] * random.uniform(5, 10)
    
    # Create risk scores
    risk_scores = np.random.beta(2, 5, size=n)  # Beta distribution for risk scores
    # Make anomalies have higher risk scores
    risk_scores[anomaly_indices] = np.random.uniform(0.7, 0.95, size=len(anomaly_indices))
    
    is_flagged = risk_scores >= 0.7
    
    df = pd.DataFrame({
        'transaction_id': transaction_ids,
        'date': dates,
        'department': departments,
        'amount': amounts,
        'payment_method': payment_methods,
        'risk_score': risk_scores,
        'is_flagged': is_flagged
    })
    
    return df

# Mock class for demo purposes
class MockAuditAgent:
    def __init__(self):
        self.is_running = False
        self.flagged_transactions = generate_sample_transactions(100)
        self.flagged_transactions = self.flagged_transactions[self.flagged_transactions['is_flagged']]
        self.reports = self._generate_mock_reports()
    
    def _generate_mock_reports(self):
        reports = []
        for i in range(5):
            report_date = datetime.now() - timedelta(days=i*7)
            transactions = generate_sample_transactions(50)
            flagged = transactions[transactions['is_flagged']]
            
            reports.append({
                'report_id': f"AR-{report_date.strftime('%Y%m%d')}",
                'date': report_date,
                'status': 'Anomalies detected',
                'summary': f"Detected {len(flagged)} potentially fraudulent transactions.",
                'flagged_count': len(flagged),
                'details': flagged,
                'insights': self._generate_mock_insights(flagged)
            })
        return reports
    
    def _generate_mock_insights(self, transactions_df):
        n_transactions = len(transactions_df)
        
        insights = f"""
        # Transaction Analysis Insights
        
        Analysis of {n_transactions} flagged transactions has revealed the following patterns:
        
        1. **Unusual Transaction Timing**: {random.randint(20, 50)}% of flagged transactions occurred outside normal business hours.
        
        2. **Recurring Patterns**: Detected {random.randint(1, 5)} clusters of similar transaction patterns that suggest systematic anomalies.
        
        3. **Amount Distribution**: The majority of flagged transactions fall within the {random.choice(['low', 'medium', 'high'])} value range, which is unusual compared to historical patterns.
        
        4. **Key Risk Indicators**: Most frequent anomalies involve {random.choice(['unusual vendors', 'duplicate payments', 'round amount figures', 'bypassed approval processes'])}.
        
        5. **Recommended Actions**: Consider detailed review of transactions from department {random.choice(['A', 'B', 'C'])}, which shows the highest concentration of anomalies.
        """
        
        return insights
    
    def start_monitoring(self):
        self.is_running = True
    
    def stop_monitoring(self):
        self.is_running = False
    
    def evaluate_batch(self, transactions):
        # Just for demo
        return transactions
    
    def generate_weekly_report(self):
        report_date = datetime.now()
        transactions = generate_sample_transactions(50)
        flagged = transactions[transactions['is_flagged']]
        
        report = {
            'report_id': f"AR-{report_date.strftime('%Y%m%d')}",
            'date': report_date,
            'status': 'Anomalies detected',
            'summary': f"Detected {len(flagged)} potentially fraudulent transactions.",
            'flagged_count': len(flagged),
            'details': flagged,
            'insights': self._generate_mock_insights(flagged)
        }
        
        self.reports.insert(0, report)
        return report
    
    def get_all_reports(self):
        return self.reports

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = MockAuditAgent()
    st.session_state.transactions = generate_sample_transactions(1000)

# Ensure reports are initialized
if 'reports' not in st.session_state:
    st.session_state.reports = st.session_state.agent.get_all_reports()
# Get agent from session state
agent = st.session_state.agent

# Sidebar
st.sidebar.title("Audit Control Panel")
st.sidebar.markdown("---")

monitoring_status = "Active" if agent.is_running else "Inactive"
status_color = "green" if agent.is_running else "red"
st.sidebar.markdown(f"### Monitoring Status: <span style='color:{status_color}'>{monitoring_status}</span>", unsafe_allow_html=True)

if not agent.is_running:
    if st.sidebar.button("Start Monitoring"):
        agent.start_monitoring()
        st.sidebar.success("Monitoring started")
        st.session_state['monitoring_status'] = "Active"  # Store status
        st.rerun()
else:
    if st.sidebar.button("Stop Monitoring"):
        agent.stop_monitoring()
        st.sidebar.error("Monitoring stopped")
        st.session_state['monitoring_status'] = "Inactive"  # Store status
        st.rerun()


st.sidebar.markdown("---")

if st.sidebar.button("Generate Report Now"):
    with st.sidebar:
        with st.spinner("Generating report..."):
            report = agent.generate_weekly_report()
            
            # Ensure reports list exists and append to it
            if 'reports' not in st.session_state:
                st.session_state.reports = []
                
            st.session_state.reports.insert(0, report)  
            st.success(f"Report generated: {report['report_id']}")
            time.sleep(1)
            st.rerun()  # Force UI refresh


st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Controls")

if st.sidebar.button("Simulate New Transactions"):
    with st.sidebar:
        with st.spinner("Processing new transactions..."):
            new_transactions = generate_sample_transactions(200)
            results = agent.evaluate_batch(new_transactions)
            st.success(f"Processed 200 new transactions")
            st.info(f"Flagged {sum(new_transactions['is_flagged'])} potential anomalies")
            # Update session state with new transactions
            st.session_state.transactions = pd.concat([st.session_state.transactions, new_transactions])
            time.sleep(1)
            st.rerun()  # Force UI refresh


# Main page content
st.title("🔍 Financial Transaction Audit Dashboard")
st.markdown("## Autonomous Risk Scoring and Continuous Audit Monitoring")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📝 Audit Reports", "⚙️ System Status"])

with tab1:
    # Dashboard tab
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Score Distribution")
        # Use Plotly for interactive charts
        flagged_df = agent.flagged_transactions
        if len(flagged_df) > 0:
            fig = px.histogram(
                flagged_df, 
                x="risk_score",
                nbins=20,
                color_discrete_sequence=['#0068c9'],
                opacity=0.75,
                marginal="rug"
            )
            fig.update_layout(
                xaxis_title="Risk Score",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No flagged transactions in current period")
    
    with col2:
        st.subheader("Department Risk Analysis")
        if len(flagged_df) > 0 and 'department' in flagged_df.columns:
            dept_risk = flagged_df.groupby('department')['risk_score'].agg(['mean', 'count']).reset_index()
            dept_risk.columns = ['Department', 'Avg. Risk Score', 'Count']
            
            fig = px.scatter(
                dept_risk,
                x='Department',
                y='Avg. Risk Score',
                size='Count',
                color='Avg. Risk Score',
                color_continuous_scale='Reds',
                hover_data=['Count']
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No department data available for flagged transactions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Score Timeline")
        # Create timeline of risk scores
        timeline_data = flagged_df.copy()
        timeline_data['date'] = pd.to_datetime(timeline_data['date'])
        timeline_data = timeline_data.sort_values('date')
        
        fig = px.scatter(
            timeline_data,
            x='date',
            y='risk_score',
            color='risk_score',
            color_continuous_scale='Reds',
            size='amount',
            hover_data=['transaction_id', 'department', 'amount']
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Risk Score",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Amount vs. Risk Score")
        fig = px.scatter(
            flagged_df,
            x='amount',
            y='risk_score',
            color='department',
            hover_data=['transaction_id', 'date', 'payment_method']
        )
        fig.update_layout(
            xaxis_title="Transaction Amount",
            yaxis_title="Risk Score",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Recently Flagged Transactions")
    if len(flagged_df) > 0:
        # Display the most recent flagged transactions
        display_cols = ['transaction_id', 'date', 'department', 'amount', 'payment_method', 'risk_score']
        display_df = flagged_df[display_cols].sort_values('risk_score', ascending=False)
        
        # Create a custom dataframe with color highlighting
        st.dataframe(
            display_df,
            height=300,
            column_config={
                "risk_score": st.column_config.ProgressColumn(
                    "Risk Score",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                    help="Risk score (higher is riskier)"
                ),
                "amount": st.column_config.NumberColumn(
                    "Amount",
                    format="$%.2f"
                ),
                "date": st.column_config.DateColumn(
                    "Date",
                    format="MMM DD, YYYY"
                )
            }
        )
    else:
        st.info("No transactions have been flagged during the current monitoring period")

with tab2:
    # Reports tab
    reports = agent.get_all_reports()
    if not reports:
        st.info("No audit reports have been generated yet.")
    else:
        report_options = [f"{r['report_id']} ({r['date'].strftime('%Y-%m-%d')})" for r in reports]
        selected_report_idx = st.selectbox("Select a report:", range(len(report_options)), 
                                         format_func=lambda x: report_options[x])
        
        report = reports[selected_report_idx]
        
        # Report header
        st.markdown(f"## Report: {report['report_id']}")
        
        # Create columns for metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Date:** {report['date'].strftime('%Y-%m-%d %H:%M:%S')}")
        with col2:
            st.markdown(f"**Status:** {report['status']}")
        with col3:
            st.markdown(f"**Flagged Transactions:** {report['flagged_count']}")
        
        st.markdown("---")
        
        # Insights
        if report['insights']:
            st.markdown("### AI-Generated Insights")
            st.markdown(report['insights'])
            st.markdown("---")
        
        # Flagged transactions visualization
        if report['details'] is not None and not report['details'].empty:
            st.markdown("### Risk Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # Department distribution
                dept_counts = report['details']['department'].value_counts().reset_index()
                dept_counts.columns = ['Department', 'Count']
                
                fig = px.pie(
                    dept_counts, 
                    values='Count', 
                    names='Department',
                    title='Flagged Transactions by Department',
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Payment method distribution
                payment_counts = report['details']['payment_method'].value_counts().reset_index()
                payment_counts.columns = ['Payment Method', 'Count']
                
                fig = px.bar(
                    payment_counts,
                    x='Payment Method',
                    y='Count',
                    title='Flagged Transactions by Payment Method',
                    color='Payment Method'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Transactions table
            st.markdown("### Flagged Transactions")
            st.dataframe(
                report['details'],
                height=300,
                column_config={
                    "risk_score": st.column_config.ProgressColumn(
                        "Risk Score",
                        format="%.2f",
                        min_value=0,
                        max_value=1
                    ),
                    "amount": st.column_config.NumberColumn(
                        "Amount",
                        format="$%.2f"
                    ),
                    "date": st.column_config.DateColumn(
                        "Date",
                        format="MMM DD, YYYY"
                    )
                }
            )

with tab3:
    # System Status tab
    st.subheader("System Performance")
    
    # Create some mock metrics
    col1, col2, col3, col4 = st.columns(4)
    daily_count = 200  # Mock data
    flagged_today = 15  # Mock data
    risk_score_avg = 0.65  # Mock data

    total_reports = len(agent.get_all_reports())  # Number of reports generated

    with col1:
        st.metric("Processed Transactions", f"{daily_count}", "+15% from last week")

    with col2:
        st.metric("Flagged Transactions", f"{flagged_today}", "-5% from last week")

    with col3:
        st.metric("Average Risk Score", f"{risk_score_avg:.2f}", "Stable")

    with col4:
        st.metric("Audit Reports Generated", f"{total_reports}")

st.markdown("---")

# System Logs
st.subheader("System Logs")
mock_logs = [
    "[INFO] Monitoring started at 10:00 AM",
    "[INFO] 200 transactions processed",
    "[WARNING] 15 anomalies detected",
    "[INFO] Report AR-20250314 generated successfully",
    "[INFO] Monitoring stopped at 5:00 PM"
]

log_display = "\n".join(mock_logs)
st.text_area("System Logs", log_display, height=200)
