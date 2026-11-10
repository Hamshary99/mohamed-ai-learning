"""
test_client.py
Interactive CLI Chat Client with Server-Sent Events (SSE) Streaming.
Maintains multi-turn conversational memory via thread_id.

Usage:
    python test_client.py                     # Starts interactive multi-turn chat
    python test_client.py "What is an F-1?"    # Single-shot query
"""

import sys
import uuid
import json
import requests

URL = "http://localhost:8000/api/v1/chat"


def stream_message(message: str, thread_id: str):
    payload = {"message": message, "thread_id": thread_id}
    sources_buffer = []
    try:
        with requests.post(URL, json=payload, stream=True, timeout=60) as response:
            if response.status_code != 200:
                print(f"\n[HTTP Error {response.status_code}]: {response.text}")
                return

            print("Advisor: ", end="", flush=True)
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue

                raw_json = line[len("data:"):].strip()
                if raw_json == "[DONE]":
                    print("\n")
                    if sources_buffer:
                        print("  [Retrieved Sources & Cross-Encoder Scores (Terminal)]: ")
                        for idx, s in enumerate(sources_buffer, 1):
                            print(f"    {idx}. {s['source']} (Page {s['page']}) | Cross-Encoder Score: {s['score']:+.3f}")
                        print()
                    break

                try:
                    event = json.loads(raw_json)
                    if "token" in event:
                        print(event["token"], end="", flush=True)
                    elif "intent" in event:
                        print(f"[{event['intent']}] ", end="", flush=True)
                    elif "sources" in event:
                        sources_buffer = event["sources"]
                    elif "error" in event:
                        print(f"\n[Stream Error]: {event['error']}")
                except json.JSONDecodeError:
                    pass

    except requests.exceptions.ConnectionError:
        print(f"\n[Connection Error] Could not connect to {URL}.")
        print("Please ensure the backend is running with: python src/main.py")
        sys.exit(1)


def main():
    # Single-shot query mode
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        stream_message(query, thread_id=f"cli_{uuid.uuid4().hex[:8]}")
        return

    # Interactive multi-turn chat mode
    thread_id = f"session_{uuid.uuid4().hex[:8]}"
    print("=" * 65)
    print("StudentGator AI Advisory - Interactive Chat Session")
    print(f"Session Thread ID: {thread_id} (maintains conversation memory)")
    print("Commands: 'exit' to quit | 'reset' to clear context | 'examples' for queries")
    print("=" * 65)

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("Session ended. Goodbye!")
            break

        if user_input.lower() in ("reset", "clear"):
            thread_id = f"session_{uuid.uuid4().hex[:8]}"
            print(f"\n[State Cleared] Started fresh session thread: {thread_id}\n")
            continue

        if user_input.lower() in ("examples", "help"):
            print("\n" + "=" * 65)
            print("Suggested Example Queries Across Knowledge Base:")
            print("• Germany:")
            print("  - Can international students work in Germany while studying?")
            print("  - What is the 140-day work rule under Section 16b AufenthG?")
            print("• United States:")
            print("  - What are the requirements for an F-1 STEM OPT extension?")
            print("  - How do I report my job's relationship to my major in SEVP?")
            print("  - What are the rules for an F-1 visa travel signature?")
            print("• France:")
            print("  - What is the deadline and fee for validating a French VLS-TS visa?")
            print("  - How many hours per year can international students work in France?")
            print("• Spain:")
            print("  - How do international students obtain a TIE card in Spain?")
            print("• StudentGator Platform:")
            print("  - What if my university application is rejected on StudentGator?")
            print("• Out-of-Scope Test Queries:")
            print("  - how to make pizza (tests 0ms conversational filter)")
            print("  - Can international students work in Canada? (tests out-of-scope country guidance)")
            print("=" * 65 + "\n")
            continue

        stream_message(user_input, thread_id=thread_id)


if __name__ == "__main__":
    main()

