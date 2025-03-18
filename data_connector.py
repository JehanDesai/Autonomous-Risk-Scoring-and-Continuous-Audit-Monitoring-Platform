import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import json
from typing import Dict, List, Union, Optional, Tuple

class DataConnector:
    # Handles connecting to and extracting data from various sources. 
    # For prototype purposes, this will generate synthetic financial transaction data.
    # In production, this would connect to ERP systems.
    
    def __init__(self, data_source: str = "synthetic", config: Optional[Dict] = None):
        # Type of data source ('synthetic', 'csv', 'database', 'sap', 'oracle')
        self.data_source = data_source
        self.config = config or {}
        self.connection = None
        
    def connect(self) -> bool:
        # Establish connection to the data source
        if self.data_source == "synthetic":
            self.connection = True
            return True
        elif self.data_source == "csv":
            # Would implement file path validation here
            return os.path.exists(self.config.get("file_path", ""))
        elif self.data_source in ["database", "sap", "oracle"]:
            return False
        return False
        
    def extract_transactions(self, start_date: Optional[datetime] = None, 
                           end_date: Optional[datetime] = None, 
                           limit: int = 1000) -> pd.DataFrame:

        # Extract financial transactions from the data source.
        # 1. Start date for transaction extraction
        # 2. End date for transaction extraction
        # 3. Maximum number of transactions to extract

        if not self.connection and not self.connect():
            raise ConnectionError(f"Failed to connect to data source: {self.data_source}")
            
        if self.data_source == "synthetic":
            return self._generate_synthetic_data(start_date, end_date, limit)
        elif self.data_source == "csv":
            return pd.read_csv(self.config.get("file_path", ""))
        else:
            return pd.DataFrame()
            
    def _generate_synthetic_data(self, start_date: Optional[datetime] = None, 
                              end_date: Optional[datetime] = None, 
                              num_records: int = 1000) -> pd.DataFrame:
        # Generate synthetic financial transaction data for prototyping
        # Set default dates if not provided
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
            
        # Create random dates between start and end
        date_range = (end_date - start_date).days
        transaction_dates = [start_date + timedelta(days=random.randint(0, date_range)) 
                            for _ in range(num_records)]
        
        # Define the categories
        departments = ['Finance', 'IT', 'HR', 'Marketing', 'Operations', 'Sales']
        expense_types = ['Travel', 'Office Supplies', 'Services', 'Equipment', 'Software', 'Consulting']
        vendors = ['Vendor A', 'Vendor B', 'Vendor C', 'Vendor D', 'Vendor E', 
                  'Vendor F', 'Vendor G', 'Vendor H', 'Vendor I', 'Vendor J']
        payment_methods = ['Credit Card', 'Wire Transfer', 'Check', 'Cash', 'Electronic Payment']
        employees = [f'Employee {i}' for i in range(1, 31)]
        
        # Generate transaction data with some anomalies
        data = {
            'transaction_id': [f'TRX-{i:06d}' for i in range(1, num_records + 1)],
            'date': transaction_dates,
            'amount': np.random.exponential(scale=500, size=num_records),
            'department': [random.choice(departments) for _ in range(num_records)],
            'expense_type': [random.choice(expense_types) for _ in range(num_records)],
            'vendor': [random.choice(vendors) for _ in range(num_records)],
            'payment_method': [random.choice(payment_methods) for _ in range(num_records)],
            'employee': [random.choice(employees) for _ in range(num_records)],
            'description': [f'Payment for {random.choice(expense_types).lower()}' for _ in range(num_records)]
        }
        
        df = pd.DataFrame(data)
        
        # Inject some anomalies
        anomaly_indices = random.sample(range(num_records), int(num_records * 0.05))
        
        # Round amount anomalies
        for idx in anomaly_indices[:len(anomaly_indices)//5]:
            df.loc[idx, 'amount'] = float(random.randint(1, 10)) * 1000.0
        
        # Weekend transactions anomalies
        for idx in anomaly_indices[len(anomaly_indices)//5:2*len(anomaly_indices)//5]:
            weekend_date = start_date + timedelta(days=random.randint(0, date_range))
            while weekend_date.weekday() < 5:  # 5 = Saturday, 6 = Sunday
                weekend_date = start_date + timedelta(days=random.randint(0, date_range))
            df.loc[idx, 'date'] = weekend_date
        
        # Duplicate transactions anomalies
        for idx in anomaly_indices[2*len(anomaly_indices)//5:3*len(anomaly_indices)//5]:
            duplicate_idx = random.randint(0, num_records-1)
            while duplicate_idx in anomaly_indices:
                duplicate_idx = random.randint(0, num_records-1)
            
            for col in ['amount', 'department', 'expense_type', 'vendor', 'payment_method']:
                df.loc[idx, col] = df.loc[duplicate_idx, col]
        
        # Unusual vendor transactions
        for idx in anomaly_indices[3*len(anomaly_indices)//5:4*len(anomaly_indices)//5]:
            df.loc[idx, 'vendor'] = f'Unusual Vendor {random.randint(1, 20)}'
        
        # High amount transactions
        for idx in anomaly_indices[4*len(anomaly_indices)//5:]:
            df.loc[idx, 'amount'] = df.loc[idx, 'amount'] * 10 + 5000
            
        return df
    
