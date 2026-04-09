"""
Unit tests for the Customer Service Chatbot.
Run: python -m pytest tests/ -v
"""

import pytest
from chatbot import (
    CustomerServiceChatbot, 
    NLPProcessor, 
    IntentClassifier, 
    ResponseGenerator
)


class TestNLPProcessor:
    """Tests for NLP preprocessing."""
    
    def setup_method(self):
        self.nlp = NLPProcessor()
    
    def test_preprocess_lowercase(self):
        assert self.nlp.preprocess("HELLO World") == "hello world"
    
    def test_preprocess_punctuation(self):
        result = self.nlp.preprocess("Hello! How are you?")
        assert "!" not in result
        assert "?" not in result
    
    def test_tokenize(self):
        tokens = self.nlp.tokenize("hello world test")
        assert tokens == ["hello", "world", "test"]
    
    def test_remove_stopwords(self):
        tokens = ["the", "quick", "brown", "fox"]
        filtered = self.nlp.remove_stopwords(tokens)
        assert "the" not in filtered
        assert "quick" in filtered
    
    def test_extract_order_number(self):
        entities = self.nlp.extract_entities("My order ORD-12345 is late")
        assert entities.get('order_number') == '12345'
    
    def test_extract_email(self):
        entities = self.nlp.extract_entities("Contact me at test@example.com")
        assert entities.get('email') == 'test@example.com'
    
    def test_extract_phone(self):
        entities = self.nlp.extract_entities("Call me at 555-123-4567")
        assert 'phone' in entities


class TestIntentClassifier:
    """Tests for intent classification."""
    
    def setup_method(self):
        self.nlp = NLPProcessor()
        self.classifier = IntentClassifier(self.nlp)
    
    def test_greeting_intent(self):
        intent, conf = self.classifier.classify("Hello there!")
        assert intent == "greeting"
        assert conf > 0.3
    
    def test_goodbye_intent(self):
        intent, conf = self.classifier.classify("Goodbye, see you later")
        assert intent == "goodbye"
    
    def test_order_status_intent(self):
        intent, conf = self.classifier.classify("Where is my order?")
        assert intent == "order_status"
    
    def test_return_refund_intent(self):
        intent, conf = self.classifier.classify("I want to return this item")
        assert intent == "return_refund"
    
    def test_payment_intent(self):
        intent, conf = self.classifier.classify("What payment methods do you accept?")
        assert intent == "payment"
    
    def test_contact_human_intent(self):
        intent, conf = self.classifier.classify("I want to talk to a real person")
        assert intent == "contact_human"
    
    def test_unknown_intent(self):
        intent, conf = self.classifier.classify("asdfghjkl random gibberish")
        assert conf < 0.3  # Low confidence for nonsense
    
    def test_get_top_intents(self):
        top = self.classifier.get_top_intents("track my order delivery", n=3)
        assert len(top) == 3
        assert top[0][0] == "order_status"


class TestResponseGenerator:
    """Tests for response generation."""
    
    def setup_method(self):
        self.generator = ResponseGenerator()
    
    def test_greeting_response(self):
        response = self.generator.generate("greeting", 0.9)
        assert len(response) > 0
        assert any(word in response.lower() for word in ["hello", "hi", "welcome"])
    
    def test_response_with_entities(self):
        entities = {'order_number': '12345'}
        response = self.generator.generate("order_status_with_number", 0.9, entities)
        assert "12345" in response
    
    def test_unknown_response(self):
        response = self.generator.generate("unknown", 0.1)
        assert len(response) > 0
    
    def test_low_confidence_response(self):
        response = self.generator.generate("order_status", 0.25)
        # Should ask for clarification
        assert "?" in response
    
    def test_context_management(self):
        self.generator.set_context("user_name", "John")
        assert self.generator.get_context("user_name") == "John"
        self.generator.clear_context()
        assert self.generator.get_context("user_name") is None


class TestChatbot:
    """Integration tests for the full chatbot."""
    
    def setup_method(self):
        self.bot = CustomerServiceChatbot(debug=True)
    
    def test_full_conversation_flow(self):
        # Test greeting
        result = self.bot.process_message("Hi there!")
        assert result['intent'] == 'greeting'
        assert 'response' in result
        
        # Test order inquiry
        result = self.bot.process_message("I want to check my order #12345")
        assert result['intent'] == 'order_status'
        assert '12345' in result['entities'].get('order_number', '')
    
    def test_conversation_history(self):
        self.bot.process_message("Hello")
        self.bot.process_message("Track my order")
        
        history = self.bot.export_conversation()
        assert len(history) == 2
    
    def test_conversation_summary(self):
        self.bot.process_message("Hello")
        self.bot.process_message("What are your store hours?")
        
        summary = self.bot.get_conversation_summary()
        assert summary['turns'] == 2
        assert 'greeting' in summary['intents_breakdown']
    
    def test_reset_conversation(self):
        self.bot.process_message("Hello")
        self.bot.reset_conversation()
        
        assert len(self.bot.conversation_history) == 0
    
    def test_get_response_simple(self):
        response = self.bot.get_response("Hello")
        assert isinstance(response, str)
        assert len(response) > 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def setup_method(self):
        self.bot = CustomerServiceChatbot()
    
    def test_empty_input(self):
        result = self.bot.process_message("")
        assert result['intent'] == 'unknown'
    
    def test_very_long_input(self):
        long_input = "order " * 100
        result = self.bot.process_message(long_input)
        assert 'response' in result
    
    def test_special_characters(self):
        result = self.bot.process_message("Hello!!! @#$% ???")
        assert 'response' in result
    
    def test_mixed_case_intent(self):
        result = self.bot.process_message("WHERE IS MY ORDER???")
        assert result['intent'] == 'order_status'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
