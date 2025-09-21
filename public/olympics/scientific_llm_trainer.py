#!/usr/bin/env python3
"""
Scientific LLM Trainer
A module to train domain-specific language models on sprint running research papers.
This creates specialized models that can understand biomechanics, physics, and sports science
concepts to generate new knowledge from scientific literature.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    get_scheduler
)
import re
import nltk
from nltk.tokenize import sent_tokenize
import PyPDF2
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import io
import base64
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Ensure nltk packages are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

@dataclass
class ScientificPaper:
    """Class to store structured data from a scientific paper"""
    filename: str
    title: str = ""
    authors: List[str] = None
    abstract: str = ""
    introduction: str = ""
    methods: str = ""
    results: str = ""
    discussion: str = ""
    conclusion: str = ""
    references: List[str] = None
    tables: List[Dict] = None
    equations: List[Dict] = None
    figures: List[Dict] = None
    full_text: str = ""
    
    def __post_init__(self):
        if self.authors is None:
            self.authors = []
        if self.references is None:
            self.references = []
        if self.tables is None:
            self.tables = []
        if self.equations is None:
            self.equations = []
        if self.figures is None:
            self.figures = []

    def to_training_examples(self) -> List[Dict[str, str]]:
        """Convert paper to training examples"""
        examples = []
        
        # Question-answer pairs for title
        if self.title:
            examples.append({
                "question": "What is the title of this research paper?",
                "answer": self.title
            })
            
            # Generate query-based examples from title
            title_keywords = re.findall(r'\b[A-Za-z]{3,}\b', self.title)
            if len(title_keywords) > 3:
                query = f"Tell me about {' '.join(title_keywords[:3])}"
                examples.append({
                    "question": query,
                    "answer": f"Based on the research paper '{self.title}', {self.abstract}"
                })
        
        # Abstract summary
        if self.abstract:
            examples.append({
                "question": "Summarize this research paper",
                "answer": self.abstract
            })
            
            # Generate more detailed response using abstract
            sentences = sent_tokenize(self.abstract)
            if len(sentences) > 2:
                key_findings = " ".join(sentences[len(sentences)//2:])
                examples.append({
                    "question": "What are the key findings of this research?",
                    "answer": key_findings
                })
        
        # Method questions
        if self.methods:
            examples.append({
                "question": "What methodology was used in this research?",
                "answer": self.methods
            })
            
            # Generate specific methodology questions
            if "participants" in self.methods.lower() or "subjects" in self.methods.lower():
                examples.append({
                    "question": "Who were the participants in this study?",
                    "answer": self._extract_participant_info(self.methods)
                })
        
        # Results questions
        if self.results:
            examples.append({
                "question": "What were the results of this study?",
                "answer": self.results
            })
            
            # Generate specific results questions
            if len(self.tables) > 0:
                table_desc = self._summarize_tables()
                examples.append({
                    "question": "What do the data tables show in this research?",
                    "answer": table_desc
                })
        
        # Implication questions
        if self.discussion:
            examples.append({
                "question": "What are the implications of this research?",
                "answer": self._extract_implications()
            })
        
        # Return all examples
        return examples
    
    def _extract_participant_info(self, methods_text: str) -> str:
        """Extract information about study participants"""
        sentences = sent_tokenize(methods_text)
        participant_info = []
        
        # Look for sentences containing participant-related keywords
        keywords = ["participant", "subject", "athlete", "runner", "volunteer", "male", "female"]
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                participant_info.append(sentence)
        
        if participant_info:
            return " ".join(participant_info)
        return "No specific participant information was provided in the study."
    
    def _summarize_tables(self) -> str:
        """Summarize the tables in the paper"""
        if not self.tables:
            return "No data tables were found in this study."
        
        summaries = []
        for i, table in enumerate(self.tables):
            if "caption" in table and "data" in table:
                summaries.append(f"Table {i+1}: {table['caption']}")
        
        if summaries:
            return "This research includes the following data tables: " + " ".join(summaries)
        return "The study includes data tables but their specific contents could not be determined."
    
    def _extract_implications(self) -> str:
        """Extract implications from discussion section"""
        if not self.discussion:
            return "No discussion of implications was found."
        
        sentences = sent_tokenize(self.discussion)
        implication_sentences = []
        
        # Keywords that often indicate implications
        keywords = ["suggest", "implication", "indicate", "conclude", "impact", "future", 
                    "important", "significant", "finding", "demonstrate"]
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                implication_sentences.append(sentence)
        
        if implication_sentences:
            return " ".join(implication_sentences)
            
        # If no clear implications found, return the last few sentences of discussion
        if len(sentences) > 3:
            return " ".join(sentences[-3:])
        
        return "The specific implications of this research were not clearly stated."


class ScientificPaperDataset(Dataset):
    """Dataset class for scientific papers"""
    
    def __init__(self, tokenizer, papers: List[ScientificPaper], max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Convert papers to training examples
        for paper in papers:
            self.examples.extend(paper.to_training_examples())
        
        logger.info(f"Created dataset with {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format as instruction with question and answer
        text = f"Below is a question about sprint running research:\n\nQuestion: {example['question']}\n\nAnswer: {example['answer']}"
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        
        # Set up labels for causal LM (same as input_ids)
        item["labels"] = item["input_ids"].clone()
        
        return item


class ScientificPaperProcessor:
    """Class to extract structured data from scientific papers"""
    
    def __init__(self):
        self.processed_papers = []
    
    def process_directory(self, pdf_dir: Union[str, Path]) -> List[ScientificPaper]:
        """Process all PDF files in a directory"""
        pdf_dir = Path(pdf_dir)
        logger.info(f"Processing PDFs in {pdf_dir}")
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                paper = self.process_pdf(pdf_file)
                if paper:
                    self.processed_papers.append(paper)
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {str(e)}")
        
        logger.info(f"Successfully processed {len(self.processed_papers)} papers")
        return self.processed_papers
    
    def process_pdf(self, pdf_path: Union[str, Path]) -> Optional[ScientificPaper]:
        """Process a single PDF file"""
        pdf_path = Path(pdf_path)
        
        # Initialize paper with filename
        paper = ScientificPaper(filename=pdf_path.name)
        
        try:
            # Extract text from PDF
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                full_text = ""
                
                # Extract text from all pages
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    full_text += page.extract_text() + "\n"
                
                paper.full_text = full_text
            
            # Extract structured sections
            self._extract_sections(paper)
            
            # Extract tables and equations (simplified)
            paper.tables = self._extract_tables(paper.full_text)
            paper.equations = self._extract_equations(paper.full_text)
            
            return paper
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {str(e)}")
            return None
    
    def _extract_sections(self, paper: ScientificPaper):
        """Extract standard sections from the paper"""
        # Simple rule-based section extraction
        full_text = paper.full_text
        
        # Extract title (typically at the beginning)
        title_match = re.search(r'^(.+?)\n', full_text)
        if title_match:
            paper.title = title_match.group(1).strip()
        
        # Extract common sections
        sections = {
            "abstract": r'(?i)abstract\s*\n(.*?)(?:\n\s*(?:introduction|keywords|background)|\Z)',
            "introduction": r'(?i)(?:introduction|background)\s*\n(.*?)(?:\n\s*(?:methods|methodology|materials)|\Z)',
            "methods": r'(?i)(?:methods|methodology|materials and methods|experimental setup)\s*\n(.*?)(?:\n\s*(?:results|findings)|\Z)',
            "results": r'(?i)(?:results|findings)\s*\n(.*?)(?:\n\s*(?:discussion|conclusion|implications)|\Z)',
            "discussion": r'(?i)discussion\s*\n(.*?)(?:\n\s*(?:conclusion|summary|references)|\Z)',
            "conclusion": r'(?i)(?:conclusion|summary)\s*\n(.*?)(?:\n\s*(?:references|bibliography|acknowledgments)|\Z)',
        }
        
        for section_name, pattern in sections.items():
            match = re.search(pattern, full_text, re.DOTALL)
            if match:
                section_content = match.group(1).strip()
                setattr(paper, section_name, section_content)
    
    def _extract_tables(self, text: str) -> List[Dict]:
        """Extract tables from text (simplified version)"""
        tables = []
        
        # Look for table markers
        table_markers = re.finditer(r'(?i)table\s+(\d+)[:\.]?\s*(.*?)(?:\n|$)', text)
        
        for marker in table_markers:
            table_num = marker.group(1)
            caption = marker.group(2).strip()
            
            # Extract table position (rough estimate)
            start_pos = marker.start()
            end_pos = start_pos + 2000  # Rough estimate of max table length
            if end_pos > len(text):
                end_pos = len(text)
            
            # Extract potential table content
            table_text = text[start_pos:end_pos]
            
            # Add to tables list
            tables.append({
                "number": table_num,
                "caption": caption,
                "text": table_text.strip()
            })
        
        return tables
    
    def _extract_equations(self, text: str) -> List[Dict]:
        """Extract mathematical equations from text (simplified version)"""
        equations = []
        
        # Look for equation markers and patterns
        # This is a simplified approach, actual equation extraction is more complex
        eq_patterns = [
            r'(?i)equation\s+(\d+)[:\.]?\s*(.*?)(?:\n|$)',  # Labeled equations
            r'([a-zA-Z])\s*=\s*([^=]+?)(?:\n|$)',  # Simple assignments
            r'([a-zA-Z])\s*=\s*([^=]+?)(?:\([\d\.]+\))',  # Equations with numbers
        ]
        
        for pattern in eq_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                # Extract equation info
                if "equation" in pattern.lower():
                    eq_num = match.group(1)
                    eq_text = match.group(2).strip()
                    eq_type = "labeled"
                else:
                    eq_num = None
                    eq_text = match.group(0).strip()
                    eq_type = "inline"
                
                # Add to equations list
                equations.append({
                    "number": eq_num,
                    "text": eq_text,
                    "type": eq_type
                })
        
        return equations


class ScientificLLMTrainer:
    """Class for training domain-specific LLMs on scientific papers"""
    
    def __init__(self, 
                 model_name: str = "gpt2", 
                 output_dir: Union[str, Path] = "./scientific_llm",
                 papers_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the trainer
        
        Args:
            model_name: Base model to fine-tune
            output_dir: Directory to save trained model
            papers_dir: Directory containing scientific papers in PDF format
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.papers_dir = Path(papers_dir) if papers_dir else None
        self.papers = []
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_papers(self, papers_dir: Optional[Union[str, Path]] = None):
        """Load papers from directory"""
        if papers_dir:
            self.papers_dir = Path(papers_dir)
        
        if not self.papers_dir:
            raise ValueError("No papers directory specified")
        
        # Process papers
        processor = ScientificPaperProcessor()
        self.papers = processor.process_directory(self.papers_dir)
        logger.info(f"Loaded {len(self.papers)} scientific papers")
        
        # Save papers metadata
        self._save_papers_metadata()
    
    def _save_papers_metadata(self):
        """Save metadata about processed papers"""
        metadata = []
        for paper in self.papers:
            metadata.append({
                "filename": paper.filename,
                "title": paper.title,
                "has_abstract": bool(paper.abstract),
                "has_methods": bool(paper.methods),
                "has_results": bool(paper.results),
                "has_discussion": bool(paper.discussion),
                "num_tables": len(paper.tables),
                "num_equations": len(paper.equations)
            })
        
        metadata_path = self.output_dir / "papers_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved papers metadata to {metadata_path}")
    
    def prepare_model(self):
        """Initialize and prepare the model for training"""
        logger.info(f"Initializing model: {self.model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train(self, 
              num_epochs: int = 3, 
              batch_size: int = 4, 
              learning_rate: float = 5e-5,
              warmup_steps: int = 500,
              max_length: int = 1024):
        """
        Train the model on scientific papers
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            max_length: Maximum sequence length
        """
        if not self.papers:
            raise ValueError("No papers loaded. Call load_papers() first.")
        
        if not self.model or not self.tokenizer:
            self.prepare_model()
        
        logger.info("Preparing training dataset")
        dataset = ScientificPaperDataset(self.tokenizer, self.papers, max_length=max_length)
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Not using masked language modeling for causal LM
        )
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            save_steps=1000,
            save_total_limit=2,
            prediction_loss_only=True,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=100,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )
        
        # Train model
        logger.info("Starting training")
        self.trainer.train()
        
        # Save final model and tokenizer
        logger.info(f"Saving model to {self.output_dir}")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info("Training complete")
    
    def evaluate(self, test_queries: List[str]) -> Dict[str, Any]:
        """
        Evaluate the trained model on test queries
        
        Args:
            test_queries: List of test queries to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not trained. Call train() first or load a saved model.")
        
        logger.info("Evaluating model on test queries")
        
        results = []
        for query in test_queries:
            # Generate response
            inputs = self.tokenizer(query, return_tensors="pt")
            
            # Move inputs to same device as model
            if torch.cuda.is_available():
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate output
            output_sequences = self.model.generate(
                **inputs,
                max_length=1024,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.92,
                temperature=0.7
            )
            
            # Decode generated output
            response = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            
            # Add to results
            results.append({
                "query": query,
                "response": response
            })
        
        # Save evaluation results
        eval_path = self.output_dir / "evaluation_results.json"
        with open(eval_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved evaluation results to {eval_path}")
        
        return {
            "num_queries": len(test_queries),
            "results": results
        }
    
    def save(self, model_path: Optional[Union[str, Path]] = None):
        """Save the trained model and tokenizer"""
        if not self.model or not self.tokenizer:
            raise ValueError("No model to save. Call train() first.")
        
        save_path = Path(model_path) if model_path else self.output_dir
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {save_path}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save model config and metadata
        config = {
            "model_name": self.model_name,
            "num_papers": len(self.papers),
            "created_at": pd.Timestamp.now().isoformat()
        }
        
        with open(save_path / "model_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    def load(self, model_path: Union[str, Path]):
        """Load a saved model and tokenizer"""
        model_path = Path(model_path)
        
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Load model config
        config_path = model_path / "model_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.model_name = config.get("model_name", "unknown")
            logger.info(f"Loaded model trained on {config.get('num_papers', 'unknown')} papers")
        
        logger.info("Model loaded successfully")


def main():
    """Main function for running the script directly"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a domain-specific LLM on scientific papers")
    parser.add_argument("--papers_dir", type=str, required=True, help="Directory containing PDF papers")
    parser.add_argument("--output_dir", type=str, default="./scientific_llm", help="Directory to save trained model")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ScientificLLMTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        papers_dir=args.papers_dir
    )
    
    # Load papers
    trainer.load_papers()
    
    # Train model
    trainer.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Save model
    trainer.save()


if __name__ == "__main__":
    main() 