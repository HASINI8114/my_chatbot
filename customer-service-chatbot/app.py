#!/usr/bin/env python3
"""
Customer Service Chatbot - Command Line Interface
Run: python app.py
"""

from chatbot import CustomerServiceChatbot


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 60)
    print("       🤖 Customer Service Chatbot")
    print("=" * 60)
    print("Type your message and press Enter.")
    print("Commands: 'quit' to exit, 'reset' to start over,")
    print("          'debug' to toggle debug mode, 'summary' for stats")
    print("=" * 60 + "\n")


def format_response(result: dict, debug_mode: bool) -> str:
    """Format chatbot response for display."""
    output = f"\n🤖 Bot: {result['response']}\n"
    
    if debug_mode and 'debug' in result:
        output += f"\n   [Debug] Intent: {result['intent']} "
        output += f"(confidence: {result['confidence']:.2%})"
        if result['entities']:
            output += f"\n   [Debug] Entities: {result['entities']}"
        output += f"\n   [Debug] Top intents: {result['debug']['top_intents'][:3]}"
    
    return output


def main():
    """Main chat loop."""
    print_banner()
    
    bot = CustomerServiceChatbot(debug=False)
    debug_mode = False
    
    # Initial greeting
    result = bot.process_message("hello")
    print(f"🤖 Bot: {result['response']}\n")
    
    while True:
        try:
            user_input = input("👤 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == 'quit':
                print("\n🤖 Bot: Thank you for using our service. Goodbye!\n")
                break
            
            if user_input.lower() == 'reset':
                bot.reset_conversation()
                print("\n🤖 Bot: Conversation reset. How can I help you?\n")
                continue
            
            if user_input.lower() == 'debug':
                debug_mode = not debug_mode
                bot.debug = debug_mode
                status = "enabled" if debug_mode else "disabled"
                print(f"\n🤖 Bot: Debug mode {status}.\n")
                continue
            
            if user_input.lower() == 'summary':
                summary = bot.get_conversation_summary()
                print(f"\n📊 Conversation Summary:")
                print(f"   Turns: {summary['turns']}")
                print(f"   Duration: {summary['duration']}")
                print(f"   Intents: {summary['intents_breakdown']}")
                if summary['entities_captured']:
                    print(f"   Entities: {summary['entities_captured']}")
                print()
                continue
            
            # Process message
            result = bot.process_message(user_input)
            print(format_response(result, debug_mode))
            
        except KeyboardInterrupt:
            print("\n\n🤖 Bot: Session ended. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
