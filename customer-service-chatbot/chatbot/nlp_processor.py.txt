"""
NLP Processor for text preprocessing and tokenization.
Handles basic NLP tasks: tokenization, stemming, stopword removal.
"""

import re
import string
from typing import List, Set


class NLPProcessor:
    """Handles text preprocessing for the chatbot."""
    
    def __init__(self):
        # Common English stopwords
        self.stopwords: Set[str] = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
            'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
            'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
            'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
            'between', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
            'off', 'over', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
            't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll',
            'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn',
            'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
            'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn',
            'please', 'thanks', 'thank', 'hi', 'hello', 'hey'
        }
        
        # Simple suffix rules for stemming
        self.suffix_rules = [
            ('ational', 'ate'), ('tional', 'tion'), ('enci', 'ence'),
            ('anci', 'ance'), ('izer', 'ize'), ('isation', 'ize'),
            ('ization', 'ize'), ('ation', 'ate'), ('ator', 'ate'),
            ('alism', 'al'), ('iveness', 'ive'), ('fulness', 'ful'),
            ('ousness', 'ous'), ('aliti', 'al'), ('iviti', 'ive'),
            ('biliti', 'ble'), ('alli', 'al'), ('entli', 'ent'),
            ('eli', 'e'), ('ousli', 'ous'), ('ling', 'l'),
            ('ement', 'e'), ('ment', ''), ('ness', ''),
            ('ing', ''), ('ings', ''), ('ed', ''), ('es', ''),
            ('ly', ''), ('ies', 'y'), ('s', '')
        ]
    
    def preprocess(self, text: str) -> str:
        """Clean and normalize text."""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)       # Normalize whitespace
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Split text into tokens."""
        processed = self.preprocess(text)
        return processed.split()
    
    def simple_stem(self, word: str) -> str:
        """Apply simple suffix-stripping stemmer."""
        if len(word) <= 3:
            return word
        
        for suffix, replacement in self.suffix_rules:
            if word.endswith(suffix):
                stem = word[:-len(suffix)] + replacement
                if len(stem) >= 2:
                    return stem
        return word
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove common stopwords from token list."""
        return [t for t in tokens if t not in self.stopwords and len(t) > 1]
    
    def process(self, text: str) -> List[str]:
        """Full preprocessing pipeline."""
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = [self.simple_stem(t) for t in tokens]
        return tokens
    
    def extract_entities(self, text: str) -> dict:
        """Extract simple entities like order numbers and emails."""
        entities = {}
        
        # Order number patterns (e.g., ORD-12345, #12345)
        order_pattern = r'(?:order\s*#?\s*|ord-?)(\d{4,10})|#(\d{4,10})'
        order_match = re.search(order_pattern, text.lower())
        if order_match:
            entities['order_number'] = order_match.group(1) or order_match.group(2)
        
        # Email pattern
        email_pattern = r'[\w.+-]+@[\w-]+\.[\w.-]+'
        email_match = re.search(email_pattern, text)
        if email_match:
            entities['email'] = email_match.group()
        
        # Phone pattern
        phone_pattern = r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            entities['phone'] = phone_match.group()
        
        return entities
