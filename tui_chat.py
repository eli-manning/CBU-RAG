import requests

url = "http://localhost:7860/chat"

if __name__ == "__main__":
    print("Hello, I am Lancer, how can I help you today? (Type 'exit' to quit)")
    conversation_history = []
    while True:
        query = input("\nYou: ")
        # escape key to exit
        if query == "exit":
            print("Goodbye!")
            break
        conversation_history.append({"role": "user", "content": query})
        try:
            response = requests.post(
                url, json={"query": query, "conversation_history": conversation_history})
            print(f"\nLancer: {response.json()["answer"]}")
            conversation_history.append(
                {"role": "assistant", "content": response.json()["answer"]})
        except Exception as e:
            print(f"Error: {e}")
