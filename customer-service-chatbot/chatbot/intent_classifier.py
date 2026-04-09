"""
Intent Classifier using keyword matching and TF-IDF similarity.
Classifies user queries into predefined intent categories.
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import math

from .nlp_processor import NLPProcessor


class IntentClassifier:
    """Classifies user intents using keyword matching and TF-IDF."""
    
    def __init__(self, nlp_processor: NLPProcessor):
        self.nlp = nlp_processor
        self.intents: Dict[str, Dict] = {}
        self.idf_scores: Dict[str, float] = {}
        self.intent_vectors: Dict[str, Dict[str, float]] = {}
        
        self._load_default_intents()
        self._compute_idf()
    
    def _load_default_intents(self):
        """Load default intent patterns and keywords."""
        self.intents = {
            'greeting': {
                'keywords': ['hello', 'hi', 'hey', 'greetings', 'good morning',
                            'good afternoon', 'good evening', 'howdy'],
                'patterns': ['hi there', 'hello there', 'hey there'],
                'priority': 1
            },
            'goodbye': {
                'keywords': ['bye', 'goodbye', 'see you', 'later', 'quit',
                            'exit', 'close', 'end'],
                'patterns': ['see you later', 'talk to you later', 'have a good day'],
                'priority': 1
            },
            'order_status': {
                'keywords': ['order', 'status', 'track', 'tracking', 'shipment',
                            'delivery', 'shipping', 'where', 'package', 'parcel',
                            'arriving', 'shipped', 'dispatched'],
                'patterns': ['where is my order', 'track my order', 'order status',
                            'when will i receive', 'shipping status', 'delivery status'],
                'priority': 2
            },
            'return_refund': {
                'keywords': ['return', 'refund', 'money back', 'exchange',
                            'cancel', 'cancellation', 'replacement', 'damaged',
                            'broken', 'wrong', 'defective'],
                'patterns': ['return policy', 'get refund', 'return item',
                            'cancel order', 'exchange product', 'money back'],
                'priority': 2
            },
            'payment': {
                'keywords': ['payment', 'pay', 'card', 'credit', 'debit',
                            'transaction', 'charge', 'billing', 'invoice',
                            'receipt', 'paid', 'checkout', 'price', 'cost'],
                'patterns': ['payment method', 'payment options', 'payment failed',
                            'how to pay', 'card declined', 'billing issue'],
                'priority': 2
            },
            'product_info': {
                'keywords': ['product', 'item', 'price', 'stock', 'available',
                            'availability', 'size', 'color', 'specification',
                            'details', 'features', 'description', 'warranty'],
                'patterns': ['product details', 'is it available', 'in stock',
                            'product information', 'tell me about'],
                'priority': 2
            },
            'account': {
                'keywords': ['account', 'password', 'login', 'signin', 'signup',
                            'register', 'profile', 'username', 'email', 'reset',
                            'forgot', 'change', 'update'],
                'patterns': ['forgot password', 'reset password', 'create account',
                            'login problem', 'update profile', 'change email'],
                'priority': 2
            },
            'contact_human': {
                'keywords': ['human', 'agent', 'representative', 'person',
                            'speak', 'talk', 'call', 'phone', 'support',
                            'help', 'escalate', 'manager', 'supervisor'],
                'patterns': ['talk to human', 'speak to agent', 'contact support',
                            'real person', 'customer service', 'call me'],
                'priority': 3
            },
            'hours_location': {
                'keywords': ['hours', 'open', 'close', 'location', 'address',
                            'store', 'shop', 'branch', 'timing', 'schedule',
                            'directions', 'near', 'closest'],
                'patterns': ['store hours', 'opening hours', 'store location',
                            'where are you located', 'business hours'],
                'priority': 2
            },
            'complaint': {
                'keywords': ['complaint', 'unhappy', 'dissatisfied', 'terrible',
                            'awful', 'worst', 'bad', 'poor', 'horrible',
                            'disappointed', 'frustrated', 'angry', 'unacceptable'],
                'patterns': ['file complaint', 'make complaint', 'not happy',
                            'very disappointed', 'bad experience', 'poor service'],
                'priority': 3
            },
            'thanks': {
                'keywords': ['thank', 'thanks', 'appreciate', 'helpful',
                            'great', 'awesome', 'excellent', 'wonderful'],
                'patterns': ['thank you', 'thanks a lot', 'much appreciated'],
                'priority': 1
            }
        }
    
    def _compute_idf(self):
        """Compute IDF scores for all keywords."""
        doc_freq = defaultdict(int)
        total_docs = len(self.intents)
        
        for intent_data in self.intents.values():
            seen_terms = set()
            all_terms = intent_data['keywords'] + intent_data.get('patterns', [])
            
            for term in all_terms:
                processed = self.nlp.process(term)
                for token in processed:
                    if token not in seen_terms:
                        doc_freq[token] += 1
                        seen_terms.add(token)
        
        for term, freq in doc_freq.items():
            self.idf_scores[term] = math.log(total_docs / (1 + freq)) + 1
        
        # Precompute intent vectors
        for intent_name, intent_data in self.intents.items():
            vector = defaultdict(float)
            all_terms = intent_data['keywords'] + intent_data.get('patterns', [])
            
            for term in all_terms:
                processed = self.nlp.process(term)
                for token in processed:
                    tf = 1 + math.log(1 + processed.count(token))
                    vector[token] = tf * self.idf_scores.get(token, 1)
            
            self.intent_vectors[intent_name] = dict(vector)
    
    def _cosine_similarity(self, vec1: Dict[str, float], 
                          vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two vectors."""
        common_keys = set(vec1.keys()) & set(vec2.keys())
        
        if not common_keys:
            return 0.0
        
        dot_product = sum(vec1[k] * vec2[k] for k in common_keys)
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def _keyword_match_score(self, tokens: List[str], 
                            intent_name: str) -> float:
        """Calculate keyword match score."""
        intent_data = self.intents[intent_name]
        keywords = set()
        
        for kw in intent_data['keywords']:
            keywords.update(self.nlp.process(kw))
        
        if not keywords:
            return 0.0
        
        matches = sum(1 for t in tokens if t in keywords)
        return matches / len(keywords)
    
    def classify(self, text: str) -> Tuple[str, float]:
        """
        Classify user input into an intent.
        Returns (intent_name, confidence_score).
        """
        tokens = self.nlp.process(text)
        
        if not tokens:
            return ('unknown', 0.0)
        
        # Build query vector
        query_vector = defaultdict(float)
        for token in tokens:
            tf = 1 + math.log(1 + tokens.count(token))
            query_vector[token] = tf * self.idf_scores.get(token, 1)
        
        best_intent = 'unknown'
        best_score = 0.0
        
        for intent_name in self.intents:
            # TF-IDF similarity
            tfidf_score = self._cosine_similarity(
                dict(query_vector), 
                self.intent_vectors[intent_name]
            )
            
            # Keyword match score
            keyword_score = self._keyword_match_score(tokens, intent_name)
            
            # Combined score (weighted average)
            combined_score = 0.4 * tfidf_score + 0.6 * keyword_score
            
            # Boost by priority for tie-breaking
            priority = self.intents[intent_name].get('priority', 2)
            combined_score *= (1 + 0.1 * (3 - priority))
            
            if combined_score > best_score:
                best_score = combined_score
                best_intent = intent_name
        
        # Confidence threshold
        confidence = min(best_score * 2, 1.0)  # Scale to 0-1
        
        if confidence < 0.15:
            return ('unknown', confidence)
        
        return (best_intent, confidence)
    
    def get_top_intents(self, text: str, n: int = 3) -> List[Tuple[str, float]]:
        """Get top N matching intents with scores."""
        tokens = self.nlp.process(text)
        
        if not tokens:
            return [('unknown', 0.0)]
        
        query_vector = defaultdict(float)
        for token in tokens:
            tf = 1 + math.log(1 + tokens.count(token))
            query_vector[token] = tf * self.idf_scores.get(token, 1)
        
        scores = []
        for intent_name in self.intents:
            tfidf_score = self._cosine_similarity(
                dict(query_vector), 
                self.intent_vectors[intent_name]
            )
            keyword_score = self._keyword_match_score(tokens, intent_name)
            combined_score = 0.4 * tfidf_score + 0.6 * keyword_score
            confidence = min(combined_score * 2, 1.0)
            scores.append((intent_name, confidence))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]
