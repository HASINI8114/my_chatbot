"""
Main Chatbot class that orchestrates NLP, intent classification, and responses.
"""

from typing import Dict, Tuple, List, Optional
from datetime import datetime

from .nlp_processor import NLPProcessor
from .intent_classifier import IntentClassifier
from .response_generator import ResponseGenerator


class CustomerServiceChatbot:
    """Main chatbot class for customer service interactions."""
    
    def __init__(self, debug: bool = False):
        self.nlp = NLPProcessor()
        self.classifier = IntentClassifier(self.nlp)
        self.response_gen = ResponseGenerator()
        self.debug = debug
        
        self.conversation_history: List[Dict] = []
        self.session_start = datetime.now()
    
    def process_message(self, user_input: str) -> Dict:
        """
        Process user message and generate response.
        Returns dict with response, intent, confidence, and entities.
        """
        # Preprocess and extract entities
        entities = self.nlp.extract_entities(user_input)
        
        # Classify intent
        intent, confidence = self.classifier.classify(user_input)
        
        # Check conversation context for better response
        self._update_context(intent, entities)
        
        # Generate response
        response = self.response_gen.generate(intent, confidence, entities)
        
        # Log conversation
        turn = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'intent': intent,
            'confidence': round(confidence, 3),
            'entities': entities,
            'response': response
        }
        self.conversation_history.append(turn)
        
        result = {
            'response': response,
            'intent': intent,
            'confidence': confidence,
            'entities': entities
        }
        
        if self.debug:
            result['debug'] = {
                'tokens': self.nlp.process(user_input),
                'top_intents': self.classifier.get_top_intents(user_input)
            }
        
        return result
    
    def _update_context(self, intent: str, entities: Dict):
        """Update conversation context based on current turn."""
        if entities.get('order_number'):
            self.response_gen.set_context('last_order', entities['order_number'])
        
        if entities.get('email'):
            self.response_gen.set_context('user_email', entities['email'])
        
        self.response_gen.set_context('last_intent', intent)
    
    def get_response(self, user_input: str) -> str:
        """Simple interface - just get the response string."""
        result = self.process_message(user_input)
        return result['response']
    
    def reset_conversation(self):
        """Reset conversation state."""
        self.conversation_history = []
        self.response_gen.clear_context()
        self.session_start = datetime.now()
    
    def get_conversation_summary(self) -> Dict:
        """Get summary of current conversation."""
        if not self.conversation_history:
            return {'turns': 0, 'duration': '0:00:00'}
        
        intents_used = {}
        for turn in self.conversation_history:
            intent = turn['intent']
            intents_used[intent] = intents_used.get(intent, 0) + 1
        
        duration = datetime.now() - self.session_start
        
        return {
            'turns': len(self.conversation_history),
            'duration': str(duration).split('.')[0],
            'intents_breakdown': intents_used,
            'entities_captured': {
                k: v for turn in self.conversation_history 
                for k, v in turn.get('entities', {}).items()
            }
        }
    
    def export_conversation(self) -> List[Dict]:
        """Export full conversation history."""
        return self.conversation_history.copy()
