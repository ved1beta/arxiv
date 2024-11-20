    
import json
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm.notebook import tqdm

class ArxivDataPreparator:
    def __init__(self, max_papers=1000, categories=None):
        """
        Initialize the ArXiv data preparator
        
        Args:
            max_papers: Maximum number of papers to process
            categories: List of arXiv categories to include (e.g., ['cs.AI', 'cs.CL'])
        """
        self.max_papers = max_papers
        self.categories = categories
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    def load_jsonl(self, filepath):
        """Load data from a JSONL file (where each line is a separate JSON object)"""
        papers = []
        with open(filepath, 'r', encoding='utf-8') as file:
            # Use tqdm to show progress bar
            pbar = tqdm(desc="Loading papers")
            count = 0
            
            for line in file:
                try:
                    paper = json.loads(line.strip())
                    
                    # Filter by categories if specified
                    if self.categories:
                        paper_categories = paper.get('categories', '').split()
                        if not any(cat in paper_categories for cat in self.categories):
                            continue
                    
                    papers.append(paper)
                    count += 1
                    pbar.update(1)
                    
                    # Break if we've reached max_papers
                    if count >= self.max_papers:
                        break
                        
                except json.JSONDecodeError:
                    continue  # Skip invalid JSON lines
                except Exception as e:
                    print(f"Error processing line: {e}")
                    continue
            
            pbar.close()
        
        return self._process_json_data(papers)
    
    def _process_json_data(self, data):
        """Process JSON data into a pandas DataFrame"""
        print(f"Processing {len(data)} papers...")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Select and rename relevant columns
        columns_mapping = {
            'title': 'title',
            'abstract': 'abstract',
            'categories': 'categories',
            'authors': 'authors',
            'update_date': 'date'
        }
        
        df = df[list(columns_mapping.keys())].rename(columns=columns_mapping)
        
        # Combine title and abstract for text field
        df['text'] = df['title'] + ' [SEP] ' + df['abstract']
        
        # Create labels based on primary category
        df['label'] = df['categories'].apply(lambda x: x.split()[0])
        
        # Clean text
        df['text'] = df['text'].apply(self._clean_text)
        
        print(f"Processed data shape: {df.shape}")
        return df
    
    def _clean_text(self, text):
        """Clean text data"""
        if pd.isna(text):
            return ''
        
        # Basic cleaning
        text = text.replace('\n', ' ')
        text = ' '.join(text.split())
        return text
    
    def prepare_for_training(self, df, test_size=0.2):
        """Prepare data for model training"""
        print("Preparing data for training...")
        
        # Convert categories to numerical labels
        label_map = {label: idx for idx, label in enumerate(df['label'].unique())}
        df['label_id'] = df['label'].map(label_map)
        
        # Create HuggingFace dataset
        dataset = Dataset.from_pandas(df)
        
        # Split dataset
        train_test = dataset.train_test_split(test_size=test_size)
        
        # Tokenize
        tokenized_datasets = train_test.map(
            self._tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing datasets"
        )
        
        return tokenized_datasets, label_map
    
    def _tokenize_function(self, examples):
        """Tokenize text data"""
        return self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512
        )

def main():
    # Initialize preparator
    preparator = ArxivDataPreparator(
        max_papers=1000,
        categories=['cs.AI', 'cs.CL']  # Optional: specify categories to filter
    )
    
    # Load and process JSONL data
    df = preparator.load_jsonl('./data/arxiv/arxiv-metadata-oai-snapshot.json')
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Number of papers: {len(df)}")
    print("\nCategory distribution:")
    print(df['label'].value_counts().head())
    
    # Prepare for training
    tokenized_datasets, label_map = preparator.prepare_for_training(df)
    
    print("\nLabel mapping:")
    for category, idx in label_map.items():
        print(f"{category}: {idx}")
    
    return tokenized_datasets, label_map

if __name__ == "__main__":
    tokenized_datasets, label_map = main()