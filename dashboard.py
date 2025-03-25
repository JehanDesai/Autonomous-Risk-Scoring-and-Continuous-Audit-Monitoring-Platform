import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import base64
from io import BytesIO

from audit_agent import AuditAgent
from data_connector import DataConnector
from audit_generation import AuditReportGenerator
from risk_scoring_model import RiskScoringModel

class AuditDashboard:
    if "audit_config" not in st.session_state:
        st.session_state["audit_config"] = {}
    if "processed_data" not in st.session_state:
        st.session_state["processed_data"] = None

    def __init__(self, audit_agent):
        self.audit_agent = audit_agent
        self.data_dir = Path("audit_data")
        self.report_dir = Path("audit_data")
    
    def run(self):
        # Run the Streamlit dashboard
        st.set_page_config(page_title="Financial Audit Dashboard",page_icon="🔍",layout="wide")
        st.markdown(
            """
            <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
            <style>
                .error-message {
                    font-family: 'Material Icons', sans-serif;
                    font-size: 20px;
                    color: red;
                    display: flex;
                    align-items: center;
                }
                .error-icon {
                    font-family: 'Material Icons';
                    font-size: 24px;
                    margin-right: 8px;
                }
                [data-testid="stSidebar"] {
                    background-color: #aceaff; /* Light Blue*/
                    padding: 20px;
                    border-right: 2px solid #ddd;
                }
                [data-testid="stAppViewContainer"] {
                    font-weight: bold;
                    # background-color: #0c233c; /* Dark blue */
                    color: black;
                }
                [data-testid="stSidebar"] h1 {
                    color: #1e48e2;
                    font-size: 26px;
                    font-weight: bold;
                }
                label[data-testid="stWidgetLabel"] {
                    color: white;
                    font-size: 16px;
                    font-weight: bold;
                }
                html, body, [class*="st-"] {
                    font-family: 'Arial', sans-serif;
                }
                div.stButton > button {
                    background-color: #1e48e2; 
                    color: white;
                    border-radius: 10px;
                    padding: 10px 20px;
                    font-size: 16px;
                    font-weight: bold;
                    transition: 0.3s;
                }
                div.stButton > button:hover {
                    background-color: #00b8f5;
                    color: white !important;
                    border: 2px solid #45a049
                }
                div.stButton > button:active {
                    background-color: #;
                    transform: scale(0.98);
                    border: 2px solid #45a049
                }

                div[data-testid="stMetric"] {
                    background-color: #00348d;  /* Dark background */
                    padding: 15px;
                    border-radius: 10px;
                    border: 2px solid #00b8f5;
                    text-align: center;
                }

                div[data-testid="stMetric"] > label {
                    color: white;
                    font-size: 18px;
                    font-weight: bold;
                }

                div[data-testid="stMetric"] > div {
                    color: white;
                    font-size: 30px;
                    font-weight: bold;
                }
                /* Style the select box container */
                div[data-baseweb="select"] > div {
                    background: linear-gradient(to right, rgb(114, 19, 234), rgb(30, 73, 226)) !important; 
                    border-radius: 8px !important;
                    border: 2px solid white !important;
                    color: white !important;
                    font-size: 16px !important;
                    padding: 5px !important;
                    height: 50px;
                }
                /* Style the selected text inside select box */
                div[data-baseweb="select"] span {
                    color: white !important;
                }
                /* Style the dropdown menu background */
                div[data-testid="stSelectboxOptionsContainer"] {
                    background-color: rgb(30, 73, 226) !important; /* Blue dropdown */
                    border-radius: 8px !important;
                }
                /* Style the dropdown options text */
                div[data-testid="stSelectboxOptionsContainer"] div {
                    color: white !important;
                    font-size: 14px !important;
                }
                /* Hover effect for dropdown options */
                div[data-testid="stSelectboxOptionsContainer"] div:hover {
                    background: rgb(114, 19, 234);
                    color: white;
                }
                .stSlider label {
                color: black !important;
                font-size: 16px !important;
                font-weight: bold !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.sidebar.image("logo-removebg-preview.png", use_container_width=True)  # Update path
        st.title("Financial Audit Dashboard")
        st.sidebar.title("Controls")
        # Sidebar controls
        action = st.sidebar.selectbox("Navigate", ["View Data", "Run Manual Audit", "View Reports", "Configure Monitoring"])
        if action == "View Data":
            self._view_data_page()
        elif action == "Run Manual Audit":
            self._run_audit_page()
        elif action == "View Reports":
            self._view_reports_page()
        elif action == "Configure Monitoring":
            self._configure_monitoring_page()
    
    def _view_data_page(self):
        # Display the data viewing page
        st.header("View Transaction Data")
        # Initialize session state if it doesn't exist
        if "processed_data" not in st.session_state:
            st.session_state["processed_data"] = None
        # Data source selection
        data_source = st.selectbox("Select Data Source",["Current Data", "Load Saved Data"],index=0)
        if data_source == "Current Data":
            # Get current data or process new data
            if st.session_state["processed_data"] is not None:
                data = st.session_state["processed_data"]
            else:
                if st.button("Process New Data"):
                    with st.spinner("Processing transactions..."):
                        data = self.audit_agent.process_transactions()
                        st.session_state["processed_data"] = data  # Store processed data in session state
                        st.success("Data processed successfully")
                        st.rerun()  # Force refresh with the new data
                    return
                else:
                    st.info("No data available. Click 'Process New Data' to extract and process transactions.")
                    return
        else:
            # Load saved data
            data_files = [f for f in self.data_dir.glob("*.pkl") if f.is_file()]
            if not data_files:
                st.warning("No saved data files found.")
                return
            selected_file = st.selectbox("Select Data File",data_files,format_func=lambda x: x.name)
            data = pd.read_pickle(selected_file)
            st.session_state["processed_data"] = data  # Store loaded data in session state too
        # Display data and visualizations
        st.subheader("Transaction Data")
        # Filter controls
        col1, col2, col3 = st.columns(3)
        with col1:
            min_risk = st.slider("Min Risk Score", 0, 100, st.session_state.get("min_risk", 0))
            st.session_state["min_risk"] = min_risk
        with col2:
            max_risk = st.slider("Max Risk Score", 0, 100, st.session_state.get("max_risk", 100))
        with col3:
            department = st.selectbox("Department", ["All"] + sorted(data["department"].unique().tolist()), index=st.session_state.get("department_index", 0))
        # Apply filters
        filtered_data = data
        filtered_data = filtered_data[(filtered_data["risk_score"] >= min_risk) & (filtered_data["risk_score"] <= max_risk)]
        if department != "All":
            filtered_data = filtered_data[filtered_data["department"] == department]
        # Show data stats
        st.subheader("Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", f"{len(filtered_data):,}")
        with col2:
            st.metric("Total Amount", f"${filtered_data['amount'].sum():,.2f}")
        with col3:
            st.metric("Average Risk Score", f"{filtered_data['risk_score'].mean():.2f}")
        with col4:
            high_risk = len(filtered_data[filtered_data['risk_score'] > 75])
            st.metric("High Risk Transactions", f"{high_risk:,}")
        # Data table with pagination
        st.subheader("Transaction Details")
        page_size = 20
        page_num = st.number_input("Page", min_value=1, max_value=max(1, len(filtered_data) // page_size + 1), value=1)
        start_idx = (page_num - 1) * page_size
        end_idx = min(start_idx + page_size, len(filtered_data))
        # Check available columns and only display those that exist
        base_display_cols = ["transaction_id", "date", "description", "amount", "department", "risk_score", "flagged"]
        display_cols = [col for col in base_display_cols if col in filtered_data.columns]
        st.dataframe(filtered_data[display_cols].iloc[start_idx:end_idx], use_container_width=True)
        # Visualizations
        st.subheader("Visualizations")
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Risk Distribution", "Departments", "Time Series"])
        with viz_tab1:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(filtered_data["risk_score"], bins=20, kde=True, ax=ax)
            ax.set_title("Risk Score Distribution")
            ax.set_xlabel("Risk Score")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        with viz_tab2:
            dept_data = filtered_data.groupby("department").agg({"amount": "sum", "risk_score": "mean", "transaction_id": "count"}).rename(columns={"transaction_id": "count"}).reset_index()
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(dept_data["amount"], labels=dept_data["department"], autopct='%1.1f%%')
                ax.set_title("Transaction Amount by Department")
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(x="department", y="risk_score", data=dept_data, ax=ax)
                ax.set_title("Average Risk Score by Department")
                ax.set_xlabel("Department")
                ax.set_ylabel("Average Risk Score")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
        with viz_tab3:
            time_data = filtered_data.copy()
            time_data['date'] = pd.to_datetime(time_data['date'])
            time_data = time_data.set_index('date')
            
            time_agg = time_data.resample('D').agg({'amount': 'sum','risk_score': 'mean','transaction_id': 'count'}).rename(columns={'transaction_id': 'count'}).reset_index()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(time_agg['date'], time_agg['amount'], marker='o')
            ax.set_title('Daily Transaction Volume')
            ax.set_xlabel('Date')
            ax.set_ylabel('Total Amount ($)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)    
        # Export options
        st.subheader("Export Data")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export to CSV"):
                tmp_download_link = self._get_download_link(filtered_data, "audit_data.csv", "csv")
                st.markdown(tmp_download_link, unsafe_allow_html=True)
        with col2:
            if st.button("Export to Excel"):
                tmp_download_link = self._get_download_link(filtered_data, "audit_data.xlsx", "excel")
                st.markdown(tmp_download_link, unsafe_allow_html=True)
    
    def _run_audit_page(self):
        # Display the manual audit page
        st.header("Run Manual Audit")
        # Audit configuration
        st.subheader("Audit Configuration")
        col1, col2 = st.columns(2)
        with col1:
            audit_type = st.selectbox("Audit Type",["Comprehensive", "Risk-Based", "Department-Specific", "Custom"])
            if audit_type == "Department-Specific":
                departments = ["All Departments"] + sorted(self.audit_agent.get_departments())
                selected_dept = st.selectbox("Select Department", departments)
            risk_threshold = st.slider("Risk Threshold", 0, 100, 0)
        with col2:
            time_period = st.selectbox("Time Period",["Last 7 Days", "Last 30 Days", "Last Quarter", "Last Year", "Custom Range"])
            if time_period == "Custom Range":
                col_a, col_b = st.columns(2)
                with col_a:
                    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
                with col_b:
                    end_date = st.date_input("End Date", datetime.now())
            sample_size = st.slider("Sample Size (% of transactions)", 1, 100, 100)
        st.subheader("Additional Rules")
        with st.expander("Configure Audit Rules"):
            rule1 = st.checkbox("Flag round number transactions (ending in 00)", value=True)
            rule2 = st.checkbox("Flag transactions just below approval thresholds", value=True)
            rule3 = st.checkbox("Flag weekend/holiday transactions", value=True)
            rule4 = st.checkbox("Flag transactions without proper documentation", value=True)
            custom_rule = st.text_area("Add custom rule (Python expression)","transaction['amount'] > 10000 and transaction['approvals'] < 2")
        # Run audit button
        if st.button("Run Audit", type="primary"):
            # Collect audit parameters
            audit_params = {"audit_type": audit_type,"risk_threshold": risk_threshold,"sample_size": sample_size, "rules": {"round_numbers": rule1,"below_threshold": rule2,"weekend_holiday": rule3,"documentation": rule4,"custom": custom_rule if custom_rule else None}}
            if "processed_data" not in st.session_state:
                st.session_state["processed_data"] = self.audit_agent.process_transactions(start_date=audit_params["start_date"], end_date=audit_params["end_date"], save_data=True)
            # Use the stored data
            try:
                audit_results = st.session_state["processed_data"]
                if audit_type == "Department-Specific":
                    audit_params["department"] = None if selected_dept == "All Departments" else selected_dept
                    print(f"Department filter set to: {audit_params.get('department')}")
                if time_period == "Custom Range":
                    audit_params["start_date"] = start_date
                    audit_params["end_date"] = end_date
                else:
                    # Convert time period to dates
                    end_date = datetime.now()
                    if time_period == "Last 7 Days":
                        start_date = end_date - timedelta(days=7)
                    elif time_period == "Last 30 Days":
                        start_date = end_date - timedelta(days=30)
                    elif time_period == "Last Quarter":
                        start_date = end_date - timedelta(days=90)
                    elif time_period == "Last Year":
                        start_date = end_date - timedelta(days=365)
                        
                    audit_params["start_date"] = start_date
                    audit_params["end_date"] = end_date
                
                # Run the audit
                with st.spinner("Running audit analysis..."):
                    try:                        
                        department_value = audit_params.get("department", None)

                        audit_results = self.audit_agent.run_manual_audit(start_date=audit_params["start_date"],end_date=audit_params["end_date"],audit_type=audit_params["audit_type"],department=department_value,risk_threshold=audit_params["risk_threshold"],sample_size=audit_params["sample_size"],rules=audit_params["rules"])
                        if isinstance(audit_results, bool):
                            st.error(f"Unexpected result type from audit function: {type(audit_results)}")
                            return
                        st.session_state["audit_results"] = audit_results
                        st.success(f"Audit completed successfully. Found {len(audit_results.get('flagged_transactions', []))} issues.")
                    except Exception as e:
                        import traceback
                        error_details = traceback.format_exc()
                        st.error(f"Error running audit: {str(e)}")
                        st.expander("Error Details").code(error_details)
                        return
                # Display audit results
                self._display_audit_results(audit_results)
                # Save audit results button
                if st.button("Save Audit Report"):
                    report_name = f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    report_path = self.report_dir / report_name
                    # Ensure directory exists
                    self.report_dir.mkdir(exist_ok=True, parents=True)
                    # Convert dates to strings in the audit results
                    audit_results_copy = audit_results.copy()
                    audit_results_copy["audit_date"] = datetime.now().isoformat()
                    audit_results_copy["start_date"] = audit_params["start_date"].isoformat()
                    audit_results_copy["end_date"] = audit_params["end_date"].isoformat()
                    # Convert dataframe to dictionary for json serialization
                    if "flagged_transactions" in audit_results_copy:
                        audit_results_copy["flagged_transactions"] = audit_results_copy["flagged_transactions"].to_dict(orient="records")
                    # Save as json
                    with open(report_path, "w") as f:
                        json.dump(audit_results_copy, f, indent=2)
                    st.success(f"Audit report saved as {report_name}")
            except Exception as e:
                st.error(f"Error running audit: {str(e)}")
                return

    def _display_audit_results(self, audit_results):
        # Display audit results
        st.subheader("Audit Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Transactions Analyzed", f"{audit_results['total_transactions']:,}")
        with col2:
            st.metric("Flagged Transactions", f"{len(audit_results['flagged_transactions']):,}")
        with col3:
            flag_rate = len(audit_results['flagged_transactions']) / audit_results['total_transactions'] * 100
            st.metric("Flag Rate", f"{flag_rate:.1f}%")
        with col4:
            st.metric("Total Risk Amount", f"${audit_results['total_risk_amount']:,.2f}")
        # Flagged transactions
        st.subheader("Flagged Transactions")
        flagged_df = audit_results['flagged_transactions']
        # Sort by risk score descending
        flagged_df = flagged_df.sort_values(by="risk_score", ascending=False)
        # Display flagged transactions
        st.dataframe(flagged_df, use_container_width=True)
        # Risk findings
        st.subheader("Risk Findings")
        findings = audit_results.get('findings', [])
        for i, finding in enumerate(findings):
            with st.expander(f"Finding {i+1}: {finding['title']}"):
                st.write(f"**Severity:** {finding['severity']}")
                st.write(f"**Description:** {finding['description']}")
                st.write(f"**Recommendation:** {finding['recommendation']}")

    def _view_reports_page(self):
        # Display the audit reports viewing page
        generator = AuditReportGenerator()
        st.header("View Audit Reports")
        # Ensure report directory exists
        self.report_dir.mkdir(exist_ok=True, parents=True)
        # Get all report files
        report_files = sorted(self.report_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not report_files:
            st.info("No audit reports found. Run an audit first to generate reports.")
            return
        # Report selection
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_report = st.selectbox("Select Report",report_files,format_func=lambda x: f"{x.stem} ({datetime.fromtimestamp(x.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})")
        with col2:
            st.write("")
            st.write("")
            if st.button("🗑️ Delete Report"):
                try:
                    os.remove(selected_report)
                    st.success(f"Report {selected_report.name} deleted successfully")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting report: {str(e)}")
        # Load and display report
        try:
            with open(selected_report, "r") as f:
                transactions = json.load(f)
            # Check if the data is a list of transactions
            if isinstance(transactions, list):
                # Create a report data structure from the list of transactions
                report_data = {
                    "audit_date": datetime.fromtimestamp(selected_report.stat().st_mtime).isoformat(),
                    "audit_type": "Manual",  # Assuming a default since it's not in the data
                    "department": "All",
                    "start_date": min([datetime.fromisoformat(t["date"]) for t in transactions if "date" in t]).isoformat() if transactions else "",
                    "end_date": max([datetime.fromisoformat(t["date"]) for t in transactions if "date" in t]).isoformat() if transactions else "",
                    "flagged_transactions": pd.DataFrame(transactions),
                    "llm_insights": ""  # Add empty LLM insights for old reports
                }
            else:
                # Already in the expected format
                report_data = transactions
                # Convert flagged transactions back to dataframe if it exists
                if "flagged_transactions" in report_data and isinstance(report_data["flagged_transactions"], list):
                    report_data["flagged_transactions"] = pd.DataFrame(report_data["flagged_transactions"])
                # Handle missing LLM insights in old reports
                if "llm_insights" not in report_data:
                    report_data["llm_insights"] = ""
            # Display report metadata
            st.subheader("Report Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Report ID:** {selected_report.stem}")
                st.write(f"**Audit Date:** {datetime.fromisoformat(report_data.get('audit_date', '')).strftime('%Y-%m-%d %H:%M')}")
            with col2:
                st.write(f"**Audit Type:** {report_data.get('audit_type', 'Unknown')}")
                st.write(f"**Department:** {report_data.get('department', 'All')}")
            with col3:
                st.write(f"**Period:** {datetime.fromisoformat(report_data.get('start_date', '')).strftime('%Y-%m-%d')} to {datetime.fromisoformat(report_data.get('end_date', '')).strftime('%Y-%m-%d')}")
            # Display AI Insights if available
            if "llm_insights" in report_data and report_data["llm_insights"]:
                st.subheader("AI Analysis Insights")
                st.markdown(report_data["llm_insights"])
                # Option to regenerate insights
                if st.button("Regenerate AI Insights"):
                    if "flagged_transactions" in report_data and isinstance(report_data["flagged_transactions"], pd.DataFrame) and not report_data["flagged_transactions"].empty:
                        with st.spinner("Generating insights..."):
                            report_data["llm_insights"] = generator._get_llm_insights(report_data["flagged_transactions"])
                            # Update the saved report
                            with open(selected_report, "w") as f:
                                # Convert DataFrame to list for JSON serialization
                                report_data_json = report_data.copy()
                                if isinstance(report_data_json["flagged_transactions"], pd.DataFrame):
                                    report_data_json["flagged_transactions"] = report_data_json["flagged_transactions"].to_dict(orient="records")
                                json.dump(report_data_json, f, indent=2)
                            st.success("AI insights regenerated successfully")
                    else:
                        st.warning("No flagged transactions available to generate insights")
            elif "flagged_transactions" in report_data and isinstance(report_data["flagged_transactions"], pd.DataFrame) and not report_data["flagged_transactions"].empty:
                # No insights but we have flagged transactions - show option to generate
                if st.button("Generate AI Insights"):
                    with st.spinner("Generating insights..."):
                        report_data["llm_insights"] = generator._get_llm_insights(report_data["flagged_transactions"])
                        # Update the saved report
                        with open(selected_report, "w") as f:
                            # Convert DataFrame to list for JSON serialization
                            report_data_json = report_data.copy()
                            if isinstance(report_data_json["flagged_transactions"], pd.DataFrame):
                                report_data_json["flagged_transactions"] = report_data_json["flagged_transactions"].to_dict(orient="records")
                            json.dump(report_data_json, f, indent=2)
                        st.success("AI insights generated successfully")
                        st.rerun()  # Reload the page to show new insights
            # Display transactions
            if "flagged_transactions" in report_data and isinstance(report_data["flagged_transactions"], pd.DataFrame) and not report_data["flagged_transactions"].empty:
                st.subheader("Transactions")
                st.dataframe(report_data["flagged_transactions"])
            elif isinstance(transactions, list) and transactions:
                st.subheader("Transactions")
                st.dataframe(pd.DataFrame(transactions))
            # Generate PDF report
            if st.button("Generate PDF Report"):
                try:
                    pdf_path = self._generate_pdf_report(report_data, selected_report.stem)
                    with open(pdf_path, "rb") as f:
                        pdf_data = f.read()
                    # Create download link
                    b64_pdf = base64.b64encode(pdf_data).decode()
                    pdf_filename = f"{selected_report.stem}.pdf"
                    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{pdf_filename}">Download PDF Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
        except Exception as e:
            st.error(f"Error loading report: {str(e)}")
            st.exception(e)  # Show the full exception for debugging

    def _configure_monitoring_page(self):
        # Display the monitoring configuration page
        st.header("Configure Audit Monitoring")
        # Monitoring status
        current_status = self.audit_agent.get_monitoring_status() if hasattr(self.audit_agent, "get_monitoring_status") else False
        status_col1, status_col2 = st.columns([3, 1])
        with status_col1:
            st.subheader("Monitoring Status")
            status_text = "Active" if current_status else "Inactive"
            status_color = "green" if current_status else "red"
            st.markdown(f"<h3 style='color: {status_color};'>● {status_text}</h3>", unsafe_allow_html=True)
            if status_text == "Inactive":
                st.info("ERP needs to be configured!", icon = "⚠️")
        with status_col2:
            st.write("")
            st.write("")
            if current_status:
                if st.button("Stop Monitoring"):
                    self.audit_agent.stop_monitoring()
                    st.success("Monitoring stopped successfully")
                    st.rerun()
            else:
                if st.button("Start Monitoring", type="primary"):
                    self.audit_agent.start_monitoring()
                    st.success("Monitoring started successfully")
                    st.rerun()
        # Monitoring configuration
        st.subheader("Monitoring Configuration")
        config_tabs = st.tabs(["Schedule", "Triggers", "Notifications", "Advanced"])
        
        with config_tabs[0]:  # Schedule
            st.subheader("Monitoring Schedule")
            
            schedule_type = st.radio("Schedule Type",["Continuous", "Daily", "Weekly", "Monthly"],horizontal=True)
            
            if schedule_type == "Continuous":
                interval_minutes = st.slider("Check Interval (minutes)", 5, 60, 15)
                st.info(f"System will check for new transactions every {interval_minutes} minutes")
            
            elif schedule_type == "Daily":
                time_of_day = st.time_input("Time of Day", datetime.now().time().replace(hour=0, minute=0, second=0))
                st.info(f"System will run daily audit at {time_of_day.strftime('%H:%M')}")
            
            elif schedule_type == "Weekly":
                day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                time_of_day = st.time_input("Time of Day", datetime.now().time().replace(hour=0, minute=0, second=0))
                st.info(f"System will run weekly audit on {day_of_week} at {time_of_day.strftime('%H:%M')}")
            
            elif schedule_type == "Monthly":
                day_of_month = st.slider("Day of Month", 1, 28, 1)
                time_of_day = st.time_input("Time of Day", datetime.now().time().replace(hour=0, minute=0, second=0))
                st.info(f"System will run monthly audit on day {day_of_month} at {time_of_day.strftime('%H:%M')}")
        with config_tabs[1]:  # Triggers
            st.subheader("Monitoring Triggers")
            
            col1, col2 = st.columns(2)
            with col1:
                trigger_risk_score = st.checkbox("Risk Score Threshold", value=True)
                if trigger_risk_score:
                    risk_threshold = st.slider("Risk Score Threshold", 0, 100, 80)
                    st.info(f"Alert when risk score exceeds {risk_threshold}")
                
                trigger_amount = st.checkbox("Transaction Amount", value=True)
                if trigger_amount:
                    amount_threshold = st.number_input("Amount Threshold", value=10000.0, min_value=0.0)
                    st.info(f"Alert when transaction amount exceeds ${amount_threshold:,.2f}")
            
            with col2:
                trigger_velocity = st.checkbox("Transaction Velocity", value=True)
                if trigger_velocity:
                    velocity_threshold = st.number_input("Transactions per Hour", value=100, min_value=1)
                    st.info(f"Alert when transaction rate exceeds {velocity_threshold} per hour")
                
                trigger_pattern = st.checkbox("Suspicious Patterns", value=True)
                if trigger_pattern:
                    pattern_options = st.multiselect("Select Patterns to Monitor",["Round Numbers", "Split Transactions", "After Hours", "Duplicate Transactions"],default=["Round Numbers", "Split Transactions"])
                    st.info(f"Alert on suspicious patterns: {', '.join(pattern_options)}")
        with config_tabs[2]:  # Notifications
            st.subheader("Notification Settings")
            
            notification_type = st.multiselect("Notification Channels",["Email", "SMS", "Dashboard", "Webhook"],default=["Email", "Dashboard"])
            
            if "Email" in notification_type:
                st.subheader("Email Notifications")
                recipients = st.text_area("Email Recipients (one per line)", "auditor@company.com\nfinance@company.com")
                email_subject = st.text_input("Email Subject", "Audit Alert: Suspicious Transaction Detected")
            
            if "SMS" in notification_type:
                st.subheader("SMS Notifications")
                phone_numbers = st.text_area("Phone Numbers (one per line)", "+1234567890")
            
            if "Webhook" in notification_type:
                st.subheader("Webhook Notifications")
                webhook_url = st.text_input("Webhook URL", "https://api.example.com/webhook")
                webhook_auth = st.text_input("Webhook Authorization", "Bearer ", type="password")
        with config_tabs[3]:  # Advanced
            st.subheader("Advanced Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                log_level = st.selectbox("Log Level", ["INFO", "DEBUG", "WARNING", "ERROR"], index=0)
                retention_days = st.slider("Data Retention (days)", 30, 365, 90)
            
            with col2:
                max_alerts = st.number_input("Maximum Alerts per Day", value=20, min_value=1)
                cooldown_period = st.slider("Alert Cooldown (minutes)", 5, 60, 15)
            
            st.subheader("Custom Rules")
            custom_rules = st.text_area(
                "Custom Monitoring Rules (Python expressions, one per line)",
                "transaction['amount'] > 9000 and transaction['amount'] < 10000\ntransaction['department'] == 'IT' and transaction['amount'] > 5000"
            )
        # Save configuration button
        if st.button("Save Configuration", type="primary"):
            # Collect all configuration settings
            config = {
                "schedule": {"type": schedule_type,"interval_minutes": interval_minutes if schedule_type == "Continuous" else None,"time_of_day": time_of_day.strftime("%H:%M") if schedule_type in ["Daily", "Weekly", "Monthly"] else None,"day_of_week": day_of_week if schedule_type == "Weekly" else None,"day_of_month": day_of_month if schedule_type == "Monthly" else None},
                "triggers": {"risk_score": {"enabled": trigger_risk_score,"threshold": risk_threshold if trigger_risk_score else None},"amount": {"enabled": trigger_amount,"threshold": amount_threshold if trigger_amount else None},"velocity": {"enabled": trigger_velocity,"threshold": velocity_threshold if trigger_velocity else None},"patterns": {"enabled": trigger_pattern,"selected_patterns": pattern_options if trigger_pattern else []}},
                "notifications": {"channels": notification_type,"email": {"recipients": recipients.split("\n") if "Email" in notification_type else [],"subject": email_subject if "Email" in notification_type else ""},"sms": {"phone_numbers": phone_numbers.split("\n") if "SMS" in notification_type else []},"webhook": {"url": webhook_url if "Webhook" in notification_type else "","auth": webhook_auth if "Webhook" in notification_type else ""}},
                "advanced": {"log_level": log_level,"retention_days": retention_days,"max_alerts": max_alerts,"cooldown_period": cooldown_period,"custom_rules": custom_rules.split("\n") if custom_rules else []}
            }
            # Save configuration
            try:
                # Ensure directory exists
                config_dir = Path("audit_config")
                config_dir.mkdir(exist_ok=True, parents=True)
                # Save config as json
                config_path = config_dir / "monitoring_config.json"
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                # Update agent configuration
                if hasattr(self.audit_agent, "set_monitoring_config"):
                    self.audit_agent.set_monitoring_config(config)
                st.success("Monitoring configuration saved successfully")
            except Exception as e:
                st.error(f"Error saving configuration: {str(e)}")

    def _get_download_link(self, df, filename, file_type):
        # Generate a download link for a DataFrame
        if file_type == "csv":
            # Generate CSV
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
        elif file_type == "excel":
            # Generate Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')
            excel_data = output.getvalue()
            b64 = base64.b64encode(excel_data).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download {filename}</a>'
        else:
            return None
        
        return href

    def _generate_pdf_report(self, report_data, report_name):
        # Generate a PDF report from audit results
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        except ImportError:
            st.error("ReportLab is required for PDF generation. Install it with 'pip install reportlab'")
            return None
        # Create PDF file
        pdf_path = self.report_dir / f"{report_name}.pdf"
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        # Create styles
        styles = getSampleStyleSheet()
        title_style = styles['Heading1']
        subtitle_style = styles['Heading2']
        normal_style = styles['Normal']
        # Build PDF content
        elements = []
        # Title
        elements.append(Paragraph(f"Audit Report: {report_name}", title_style))
        elements.append(Spacer(1, 12))
        # Report metadata
        elements.append(Paragraph("Report Information", subtitle_style))
        metadata = [
            ["Report ID:", report_name],
            ["Audit Date:", datetime.fromisoformat(report_data.get('audit_date', '')).strftime('%Y-%m-%d %H:%M')],
            ["Audit Type:", report_data.get('audit_type', 'Unknown')],
            ["Department:", report_data.get('department', 'All')],
            ["Period:", f"{datetime.fromisoformat(report_data.get('start_date', '')).strftime('%Y-%m-%d')} to {datetime.fromisoformat(report_data.get('end_date', '')).strftime('%Y-%m-%d')}"]
        ]
        meta_table = Table(metadata, colWidths=[100, 300])
        meta_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('RIGHTPADDING', (0, 0), (0, -1), 12),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 20))
        # Summary section
        elements.append(Paragraph("Audit Summary", subtitle_style))
        if "summary" in report_data:
            for key, value in report_data["summary"].items():
                if isinstance(value, (int, float)) and key not in ["start_date", "end_date", "audit_date"]:
                    formatted_key = " ".join(word.capitalize() for word in key.split("_"))
                    elements.append(Paragraph(f"{formatted_key}: {value}", normal_style))
        elements.append(Spacer(1, 20))
        # AI-generated insights section
        if "llm_insights" in report_data and report_data["llm_insights"]:
            elements.append(Paragraph("AI Analysis Insights", subtitle_style))
            # Process the LLM insights - split by newlines to create paragraphs
            insights_text = report_data["llm_insights"]
            for paragraph in insights_text.split('\n\n'):
                if paragraph.strip():
                    elements.append(Paragraph(paragraph.strip(), normal_style))
                    elements.append(Spacer(1, 6))
            elements.append(Spacer(1, 12))
        # Flagged transactions
        if "flagged_transactions" in report_data and isinstance(report_data["flagged_transactions"], pd.DataFrame) and not report_data["flagged_transactions"].empty:
            elements.append(Paragraph("Flagged Transactions", subtitle_style))
            # Convert DataFrame to list for Table
            df = report_data["flagged_transactions"]
            # Limit columns for PDF readability
            display_columns = ['transaction_id', 'date', 'amount', 'department', 'risk_score', 'flag_reason']
            available_cols = [col for col in display_columns if col in df.columns]
            # Create header row
            header = [col.replace('_', ' ').title() for col in available_cols]
            data = [header]
            # Add data rows (limit to 50 records for PDF performance)
            max_rows = min(50, len(df))
            for i in range(max_rows):
                row = [str(df.iloc[i][col]) for col in available_cols]
                data.append(row)
            # Add note if truncated
            if len(df) > max_rows:
                elements.append(Paragraph(f"Showing {max_rows} of {len(df)} flagged transactions", normal_style))
            # Create table
            table = Table(data)
            # Style the table
            table_style = [('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),('TEXTCOLOR', (0, 0), (-1, 0), colors.black),('ALIGN', (0, 0), (-1, 0), 'CENTER'),('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),('BOTTOMPADDING', (0, 0), (-1, 0), 12),('GRID', (0, 0), (-1, -1), 1, colors.black),('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),]
            # Add alternating row colors
            for i in range(1, len(data)):
                if i % 2 == 0:
                    table_style.append(('BACKGROUND', (0, i), (-1, i), colors.whitesmoke))
            table.setStyle(TableStyle(table_style))
            elements.append(table)
        else:
            elements.append(Paragraph("No flagged transactions found in this report.", normal_style))
        # Add footer with timestamp
        elements.append(Spacer(1, 30))
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ParagraphStyle("Footer", parent=normal_style, fontSize=8, textColor=colors.grey)))
        # Build PDF
        doc.build(elements)
        return pdf_path

my_data_connector = DataConnector()
my_risk_scoring_model = RiskScoringModel()
my_report_generator = AuditReportGenerator()
my_audit_agent = AuditAgent(my_data_connector, my_risk_scoring_model, my_report_generator)
dashboard = AuditDashboard(my_audit_agent)
dashboard.run()