import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

class ComplaintProcessor:
    """
    A class to handle loading, EDA, filtering, and preprocessing of 
    CFPB complaint data.
    """
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        
        # The 5 strict categories required by your business objective
        self.target_categories = [
            "Credit card", 
            "Personal loan", 
            "Savings account", 
            "Money transfers",
            "Buy Now - Pay Later"
        ]

        # Mapping for the MAIN 'Product' column
        self.product_map = {
            # Raw Value in CSV  :  Your Target Category
            "Credit card": "Credit card",
            "Credit card or prepaid card": "Credit card",
            "Prepaid card": "Credit card",
            
            "Payday loan, title loan, or personal loan": "Personal loan",
            "Payday loan": "Personal loan",
            "Student loan": "Personal loan",
            "Consumer Loan": "Personal loan",
            
            "Checking or savings account": "Savings account",
            "Bank account or service": "Savings account",
            "Savings account": "Savings account",
            
            "Money transfer, virtual currency, or money service": "Money transfers",
            "Money transfers": "Money transfers"
        }

    def load_data(self):
        """
        Loads the dataset from the CSV file.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found at {self.file_path}")
        
        print(f"Loading data from {self.file_path}...")
        self.df = pd.read_csv(self.file_path)
        print(f"Data loaded successfully. Shape: {self.df.shape}")
        return self.df



    def get_basic_stats(self):
        """
        Returns basic statistics about the dataset structure.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        narrative_col = 'Consumer complaint narrative'
        total_rows = len(self.df)
        missing_narratives = self.df[narrative_col].isna().sum()
        present_narratives = total_rows - missing_narratives
        
        stats = {
            "Total Rows": total_rows,
            "Columns": list(self.df.columns),
            "Missing Narratives": missing_narratives,
            "Present Narratives": present_narratives
        }
        return stats

    def plot_product_distribution(self):
        """
        Visualizes the distribution of complaints across products.
        """
        if self.df is None:
            raise ValueError("Data not loaded.")
            
        plt.figure(figsize=(12, 6))
        product_counts = self.df['Product'].value_counts().head(10) # Top 10 for readability
        sns.barplot(x=product_counts.values, y=product_counts.index, palette='viridis')
        plt.title('Top 10 Product Categories by Complaint Volume')
        plt.xlabel('Number of Complaints')
        plt.ylabel('Product')
        plt.show()

    def normalize_product(self, row):
        """
        Custom logic to categorize a single row.
        Priority:
        1. Check 'Sub-product' for BNPL.
        2. Map 'Product' for everything else.
        """
        product = row.get('Product', '')
        sub_product = row.get('Sub-product', '')

        # 1. Check for Buy Now - Pay Later in Sub-product
        # Note: CFPB often lists BNPL as 'Buy Now, Pay Later' (with comma)
        if isinstance(sub_product, str) and 'Buy Now' in sub_product:
            return "Buy Now - Pay Later"

        # 2. Map the main Product column
        return self.product_map.get(product, None)

    def process_and_filter(self, chunk_size=100000):
        """
        Reads the large CSV in chunks and filters it to avoid MemoryError.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found at {self.file_path}")

        processed_chunks = []
        print(f"Starting chunked processing (Size: {chunk_size})...")

        # Use iterator to read file in parts
        for i, chunk in enumerate(pd.read_csv(self.file_path, chunksize=chunk_size, low_memory=False)):
            
            # 1. Immediate filtering: Only keep rows where 'Consumer complaint narrative' is NOT null
            chunk = chunk.dropna(subset=['Consumer complaint narrative']).copy()
            
            # 2. Apply mapping logic
            # We check Sub-product for BNPL first
            chunk['Normalized_Product'] = chunk.apply(
                lambda row: "Buy Now - Pay Later" if (isinstance(row.get('Sub-product'), str) and 'Buy Now' in row['Sub-product'])
                else self.product_map.get(row['Product'], None), axis=1
            )

            # 3. Keep only our target categories
            chunk = chunk[chunk['Normalized_Product'].isin(self.target_categories)]
            
            # 4. Clean up columns to save memory
            chunk['Product'] = chunk['Normalized_Product']
            # Keep only columns necessary for the RAG pipeline
            cols_to_keep = ['Date received', 'Product', 'Sub-product', 'Consumer complaint narrative', 'Complaint ID']
            chunk = chunk[cols_to_keep]

            processed_chunks.append(chunk)
            if (i + 1) % 5 == 0:
                print(f"Processed {(i + 1) * chunk_size} rows...")

        # Combine the small filtered chunks into one manageable DataFrame
        self.df = pd.concat(processed_chunks, ignore_index=True)
        print(f"Finished! Final dataset size: {len(self.df)} rows.")
        return self.df


    def clean_text(self, text):
        """
        Helper method to clean a single text string.
        """
        if not isinstance(text, str):
            return ""
        
        # 1. Lowercase
        text = text.lower()
        
        # 2. Remove standard boilerplate "I am writing to file a complaint..."
        # We can use regex to catch variations
        text = re.sub(r'i am writing to (file a|make a) complaint.*?', '', text)
        
        # 3. Remove "XX/XX/XXXX" style anonymized dates often found in CFPB data
        text = re.sub(r'x{2}/x{2}/x{4}', '', text)
        
        # 4. Remove purely special characters (keeping alphanumeric and basic punctuation)
        text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
        
        # 5. Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


    def preprocess_narratives(self):
        """
        Applies text cleaning to the narrative column.
        """
        if self.df is None:
            raise ValueError("Data not loaded.")
        
        print("Cleaning text narratives (this may take a moment)...")
        self.df['cleaned_narrative'] = self.df['Consumer complaint narrative'].apply(self.clean_text)
        print("Text cleaning complete.")

    def analyze_narrative_length(self):
        """
        Calculates and visualizes word counts of the cleaned narratives.
        """
        if 'cleaned_narrative' not in self.df.columns:
            raise ValueError("Narratives not processed. Call preprocess_narratives() first.")
            
        self.df['word_count'] = self.df['cleaned_narrative'].apply(lambda x: len(str(x).split()))
        
        plt.figure(figsize=(10, 5))
        sns.histplot(self.df['word_count'], bins=50, kde=True, color='teal')
        plt.title('Distribution of Complaint Narrative Lengths (Word Count)')
        plt.xlabel('Word Count')
        plt.show()
        
        print(self.df['word_count'].describe())

    def save_processed_data(self, output_path):
        """
        Saves the filtered and processed dataframe to CSV.
        """
        if self.df is None:
            raise ValueError("No data to save.")
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")