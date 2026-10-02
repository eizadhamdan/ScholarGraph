# Application entry point / CLI interface
import sys
from .agent import ScholarGraphAgent


def main():
    print("======================================================")
    print("   ScholarGraph: Agentic GraphRAG Research Engine     ")
    print("======================================================\n")

    agent = ScholarGraphAgent()

    while True:
        try:
            query = input("\nEnter your query (or 'exit' to quit): ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting ScholarGraph. Goodbye!")
                sys.exit(0)

            response = agent.run(query)
            print("\n---------------- RESPONSE ----------------")
            print(response)
            print("------------------------------------------")
        except KeyboardInterrupt:
            print("\nExiting ScholarGraph. Goodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()
