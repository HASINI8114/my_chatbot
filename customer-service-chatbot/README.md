# Customer Service Chatbot

A rule-based chatbot with NLP capabilities for handling customer service queries.

## Features

- **Intent Classification** - Identifies user intents using TF-IDF and keyword matching
- **Entity Extraction** - Extracts order numbers, emails, and phone numbers
- **Context Management** - Maintains conversation context across turns
- **Multiple Response Templates** - Varied, natural-sounding responses
- **Debug Mode** - Inspect classification decisions

## Supported Intents

| Intent | Example Queries |
|--------|----------------|
| Greeting | "Hello", "Hi there" |
| Order Status | "Where is my order?", "Track order #12345" |
| Returns/Refunds | "I want to return this", "Refund policy" |
| Payment | "Payment methods", "Card declined" |
| Product Info | "Tell me about this product" |
| Account | "Reset password", "Update email" |
| Contact Human | "Talk to a person", "Call support" |
| Hours/Location | "Store hours", "Where are you located" |
| Complaint | "I'm unhappy with service" |
| Thanks | "Thank you", "That helped" |
| Goodbye | "Bye", "See you later" |

## Installation

```bash
git clone [github.com](https://github.com/yourusername/customer-service-chatbot.git)
cd customer-service-chatbot
pip install -r requirements.txt
