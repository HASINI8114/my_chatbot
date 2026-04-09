"""
Response Generator for creating contextual responses.
Uses templates and entity substitution for dynamic responses.
"""

import random
from typing import Dict, List, Optional
from datetime import datetime


class ResponseGenerator:
    """Generates responses based on intent and context."""
    
    def __init__(self):
        self.responses = self._load_default_responses()
        self.context: Dict = {}
    
    def _load_default_responses(self) -> Dict[str, List[str]]:
        """Load default response templates."""
        return {
            'greeting': [
                "Hello! Welcome to our customer service. How can I help you today?",
                "Hi there! I'm here to assist you. What can I do for you?",
                "Good day! How may I assist you today?",
                "Hello! Thanks for reaching out. What brings you here today?"
            ],
            'goodbye': [
                "Goodbye! Thanks for contacting us. Have a great day!",
                "Take care! Feel free to reach out if you have more questions.",
                "Bye! It was nice helping you. Have a wonderful day!",
                "Thanks for chatting with us. Goodbye!"
            ],
            'order_status': [
                "I can help you track your order! Please provide your order number (e.g., ORD-12345).",
                "Sure, I'll help you check your order status. What's your order number?",
                "To track your order, I'll need your order number. It should be in your confirmation email."
            ],
            'order_status_with_number': [
                "I found order #{order_number}. It's currently {status}. Expected delivery: {delivery_date}.",
                "Order #{order_number} is {status}. You should receive it by {delivery_date}.",
                "Your order #{order_number} status: {status}. Estimated arrival: {delivery_date}."
            ],
            'return_refund': [
                "I understand you'd like to return an item or request a refund. Our return policy allows returns within 30 days of purchase. Would you like me to start a return for you?",
                "I can help with returns and refunds! Items can be returned within 30 days in original condition. Do you have your order number handy?",
                "For returns and refunds, items must be in original packaging and returned within 30 days. Please provide your order number to proceed."
            ],
            'payment': [
                "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay. Is there a specific payment question I can help with?",
                "I can help with payment questions! We support all major credit cards, PayPal, and digital wallets. What would you like to know?",
                "For payment options, we accept credit/debit cards, PayPal, and Apple Pay. Are you having trouble with a payment?"
            ],
            'payment_issue': [
                "I'm sorry to hear about the payment issue. Common solutions: 1) Check card details, 2) Ensure sufficient funds, 3) Try a different payment method. Would you like me to connect you with our billing team?",
                "Payment issues can be frustrating. Please verify your card information and try again. If the problem persists, our billing team can help."
            ],
            'product_info': [
                "I'd be happy to help with product information! Could you tell me which product you're interested in?",
                "Sure, I can provide product details. What product would you like to know about?",
                "I can help with product questions. Which item are you looking at?"
            ],
            'account': [
                "I can help with account-related questions. Are you trying to: create an account, reset password, or update your information?",
                "For account issues, I'm here to help! Do you need to reset your password or update your profile?",
                "Account management is easy! What would you like to do - reset password, update details, or something else?"
            ],
            'account_password': [
                "To reset your password: 1) Go to the login page, 2) Click 'Forgot Password', 3) Enter your email, 4) Check your inbox for reset link. The link expires in 24 hours.",
                "Password reset is simple! Click 'Forgot Password' on the login page, enter your email, and follow the link we send you."
            ],
            'contact_human': [
                "I understand you'd like to speak with a human agent. Our customer service team is available:\n• Phone: 1-800-123-4567 (Mon-Fri, 9AM-6PM EST)\n• Email: support@example.com\n• Live chat: Available on our website",
                "Of course! You can reach our support team at 1-800-123-4567 or email support@example.com. Would you like me to have someone call you back?",
                "I'll connect you with a human agent. You can call us at 1-800-123-4567 or I can arrange a callback. Which works better for you?"
            ],
            'hours_location': [
                "Our customer service hours are Monday-Friday, 9AM-6PM EST. Our main office is at 123 Business Ave, New York, NY 10001.",
                "We're available Mon-Fri, 9AM-6PM EST. You can also visit our store locator on our website to find the nearest location.",
                "Business hours: Mon-Fri 9AM-6PM EST. For store locations, please check our website's store finder."
            ],
            'complaint': [
                "I'm truly sorry to hear about your negative experience. Your feedback is important to us. Could you please describe what happened so I can help resolve this?",
                "I apologize for any inconvenience you've experienced. Please share the details and I'll do my best to make it right.",
                "I'm sorry you're not satisfied. We take all complaints seriously. Please tell me more about the issue so we can address it properly."
            ],
            'thanks': [
                "You're welcome! Is there anything else I can help you with?",
                "Happy to help! Let me know if you need anything else.",
                "My pleasure! Don't hesitate to ask if you have more questions.",
                "Glad I could assist! Anything else on your mind?"
            ],
            'unknown': [
                "I'm not quite sure I understand. Could you rephrase that or choose from these topics:\n• Order status & tracking\n• Returns & refunds\n• Payment questions\n• Product information\n• Account help",
                "I didn't catch that. I can help with orders, returns, payments, products, or account issues. Which would you like?",
                "Sorry, I'm not sure what you need. Try asking about: order tracking, returns, payments, or account help.",
                "I couldn't understand that request. Could you try rephrasing? I'm best at helping with orders, returns, and account questions."
            ],
            'fallback_low_confidence': [
                "I think you might be asking about {intent}. Is that correct? If not, please try rephrasing your question.",
                "Did you mean to ask about {intent}? Let me know and I'll do my best to help."
            ]
        }
    
    def set_context(self, key: str, value: any):
        """Set context variable for response personalization."""
        self.context[key] = value
    
    def get_context(self, key: str) -> Optional[any]:
        """Get context variable."""
        return self.context.get(key)
    
    def clear_context(self):
        """Clear all context."""
        self.context = {}
    
    def generate(self, intent: str, confidence: float, 
                entities: Dict = None) -> str:
        """Generate response based on intent and entities."""
        entities = entities or {}
        
        # Handle low confidence with clarification
        if 0.15 <= confidence < 0.4 and intent != 'unknown':
            templates = self.responses.get('fallback_low_confidence', [])
            if templates:
                response = random.choice(templates)
                readable_intent = intent.replace('_', ' ')
                return response.format(intent=readable_intent)
        
        # Select appropriate response category
        response_key = intent
        
        # Handle special cases with entities
        if intent == 'order_status' and entities.get('order_number'):
            response_key = 'order_status_with_number'
            # Simulate order lookup
            entities['status'] = random.choice([
                'being prepared for shipping',
                'shipped and in transit',
                'out for delivery'
            ])
            entities['delivery_date'] = 'within 3-5 business days'
        
        if intent == 'account' and any(kw in str(entities) for kw in ['password', 'reset', 'forgot']):
            response_key = 'account_password'
        
        if intent == 'payment' and any(kw in str(entities) for kw in ['fail', 'issue', 'problem', 'decline']):
            response_key = 'payment_issue'
        
        # Get response templates
        templates = self.responses.get(response_key, self.responses['unknown'])
        response = random.choice(templates)
        
        # Substitute entities
        try:
            response = response.format(**entities)
        except KeyError:
            pass  # Keep original if substitution fails
        
        return response
    
    def add_response(self, intent: str, responses: List[str]):
        """Add custom responses for an intent."""
        if intent in self.responses:
            self.responses[intent].extend(responses)
        else:
            self.responses[intent] = responses
