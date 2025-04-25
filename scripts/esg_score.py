#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 11:27:20 2025

@author: ...
"""

import os
import re
import pandas as pd
from sec_edgar_downloader import Downloader

from bs4 import BeautifulSoup

import torch
import nltk
from nltk.tokenize import word_tokenize
from transformers import AutoModelForSequenceClassification, BertTokenizer
import numpy as np

#Download necessary NLTK data
nltk.download('punkt')

#Load FinBERT sentiment model and BERT tokenizer (manual BERT tokenization approach)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-esg")

import gc
gc.collect()

import spacy
nlp = spacy.load("en_core_web_sm")
max_length = 5_000_000 
nlp.max_length = max_length  # Adjust max_length if needed

#start by deleting the csv file to start fresh
csv_file = "all_filings_esg_text.csv"
# Check if the file exists and delete it
if os.path.exists(csv_file):
    os.remove(csv_file)
    print(f"{csv_file} deleted successfully.")
else:
    print(f"{csv_file} does not exist.")


# Function to download filings using sec-edgar-downloader
def download_filings(cik, filing_types, save_dir="filings"):
    """
    Downloads SEC filings (default is 10-K) for a specific company by CIK using sec-edgar-downloader.
    
    :param cik: Company CIK code (e.g., Apple is 320193)
    :param filing_type: Type of filing to download (default is '10-K')
    :param save_dir: Directory where the filings will be saved
    """
    dl = Downloader(email_address="...", company_name="Toronto Metropolitan University", download_folder=save_dir)
    for filing_type in filing_types:
        try:
            dl.get(filing_type, cik, after="2021-01-01")
            print(f"{filing_type} filings downloaded for CIK {cik}.")
        except Exception as e:
            print(f"Failed to download {filing_type} for CIK {cik}: {e}")

# Function to read and extract text from a downloaded filing
def read_filing(file_path):
    """Read the content of a downloaded filing."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to clean the filing text and remove unwanted sections (headers, footers, etc.)
def clean_filing_text(filing_text):
    """Clean the filing text to remove unwanted headers, footers, or irrelevant information."""
    filing_text = re.sub(r'\n.*(disclaimer|footnote|legal).*', '', filing_text, flags=re.IGNORECASE)
    # Remove URLs (http, https, www)
    filing_text = re.sub(r'https?://\S+|www\.\S+', '', filing_text)
    return filing_text.strip()

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'lxml')
    return soup.get_text()

E_keywords = {"environmental", "climate change", "carbon", "renewable", "sustainability", "greenhouse", "pollution", "energy", "emission", "reduce", "waste", "effluent", "spill", "biodiversity", "natural resources", "climate risk", "net zero", "environmental impact"}
S_keywords = {"diversity", "human rights", "labor standards", "workforce safety", "community engagement", "human capital", "growth", "skill", "initiative", "turnover", "talent", "experience", "recruitment", "privacy", "development", "train", "cybersecurity", "breach", "equity", "health", "fatality", "inclusion", "employee engagement", "supply chain", "fair wages", "working conditions"}
G_keywords = {"board diversity", "executive compensation", "ethics", "corporate governance", "shareholder rights", "leadership", "integrity", "incentive", "competitive", "corruption", "stakeholder", "transparency", "accountability", "pay", "audit", "whistleblower", "compliance", "risk management"}

def truncate_large_text(text, max_length=max_length): 
    return text[:max_length] if len(text) > max_length else text

def extract_esg_sections(text):
    """Extracts sections related to ESG topics based on NLP and predefined keywords."""
    doc = nlp(truncate_large_text(text))

    e_sections = [
    re.sub(r'\s+', ' ', sent.text).strip() for sent in doc.sents if any(token.text.lower() in E_keywords for token in sent)
    ]
    
    s_sections = [
    re.sub(r'\s+', ' ', sent.text).strip() for sent in doc.sents if any(token.text.lower() in S_keywords for token in sent)
    ]
    
    g_sections = [
     re.sub(r'\s+', ' ', sent.text).strip() for sent in doc.sents if any(token.text.lower() in G_keywords for token in sent)
    ]
    
    #important sentences only
    e_sections_imp = [text for text in e_sections if len(text) > 50]
    
    s_sections_imp = [text for text in s_sections if len(text) > 50]
    
    g_sections_imp = [text for text in g_sections if len(text) > 50]
    
    return [e_sections_imp,s_sections_imp, g_sections_imp]

def extract_year(text):
    """Extracts the filing year from the SEC-DOCUMENT header."""
    match = re.search(r'<ACCEPTANCE-DATETIME>(\d{4})\d{4}', text)
    if match:
        return match.group(1)[:4]  # Extract YYYY from YYYYMMDD format
    return "Unknown"

def analyze_esg_sentiment(text):
    tokens = word_tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_tensor = torch.tensor([token_ids])
    
    if input_tensor.shape[1] > 512:
        input_tensor = input_tensor[:, :512]  # Manually truncate to max 512 tokens
    
    with torch.no_grad():
        outputs = model(input_tensor)
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()
    labels = ["Negative", "Neutral", "Positive"]
    sentiment_score = {label: score for label, score in zip(labels, scores)}

    return sentiment_score["Positive"] - sentiment_score["Negative"]  # Compute a compound score

def safe_mean(sentiments):
    """Calculates the mean of g_sentiments safely."""
    if not sentiments:  # Check if list is empty
        print("Warning: sentiments is empty. Returning default value 0.")
        return 0  # Or return np.nan if that fits better

    sentiments = [val for val in sentiments if isinstance(val, (int, float)) and not np.isnan(val)]  # Remove invalid values

    if not sentiments:  # Check again after cleaning
        print("Warning: No valid numbers in sentiments. Returning default value 0.")
        return 0

    return np.mean(sentiments)
  

def process_filings(ciks, filing_types, save_dir="filings", csv_file=csv_file):
    """Download filings for multiple CIKs, clean text, and append data to a CSV file."""
    all_data = []

    for cik in ciks:
        print(f"\nProcessing filings for CIK: {cik}")
        
        # Step 1: Download filings for the current CIK
        download_filings(cik, filing_types, save_dir)
        
        for root, dirs, files in os.walk(save_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                if filename.endswith('.txt'):
                    print(f"Processing {filename}...")
                    
                    filing_text = read_filing(file_path)
                    year = extract_year(filing_text)
                    clean_text = clean_filing_text(filing_text)
                    raw_text = extract_text_from_html(clean_text)
                    esg_texts = extract_esg_sections(raw_text)
                    
                    #sentiment analysis
                    e_sentiments = [analyze_esg_sentiment(text) for text in esg_texts[0]]
                    s_sentiments = [analyze_esg_sentiment(text) for text in esg_texts[1]]
                    g_sentiments = [analyze_esg_sentiment(text) for text in esg_texts[2]]
                    
                    all_data.append({
                        "CIK": cik,
                        "Year": year,
                        "E_Text": esg_texts[0],
                        "S_Text": esg_texts[1],
                        "G_Text": esg_texts[2],
                        "e_score": safe_mean(e_sentiments),
                        "s_score": safe_mean(s_sentiments),
                        "g_score": safe_mean(g_sentiments)
                    })
                    
                    os.remove(file_path)

    if all_data:
        df = pd.DataFrame(all_data)
        
        # Check if the file exists to determine whether to write the header
        file_exists = os.path.isfile(csv_file)

        df.to_csv(csv_file, mode='a', header=not file_exists, index=False)
        print(f"Data appended to '{csv_file}'.")
    else:
        print("No new data to save.")

def main():
    # Read the CSV file
    data = pd.read_csv("sp500_firm_execu.csv", dtype={'cik':'Int64'})
    unique_firm = data[['cik']].dropna(subset=["cik"]).drop_duplicates()
    print("Count: ", unique_firm.shape[0])

    cik_list = unique_firm['cik'].tolist()

    print("Count: ", len(cik_list))
    ###################
    
    # Filings of possible ESG content
    filing_types = ["10-K"]
    
    # Process CIKs in batches
    batch_size = 5

    while cik_list:
        batch = cik_list[:batch_size]  # Get the first 2batch
        cik_list = cik_list[batch_size:]  # Remove processed CIKs from the list
        process_filings(batch, filing_types)  # Process the batch


if __name__ == "__main__":
    main()
