from textwrap import dedent
from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.tools.serpapi import SerpApiTools
import re
import os
import argparse
from agno.models.ollama import Ollama
from datetime import datetime, timedelta





def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="AI Travel Planner using Llama-3.2")
    parser.add_argument("prompt", type=str, help="Travel prompt (e.g., 'Paris for 5 days')")
    parser.add_argument("--serp-api-key", type=str, help="SerpAPI key for search functionality")
    args = parser.parse_args()
    
    ollama_model = Ollama(id="llama3.2:latest")
    
    print("=" * 60)
    print("AI Travel Planner using Llama-3.2")
    print("=" * 60)
    
    # Get SerpAPI key from arguments, environment, or prompt
    serp_api_key = args.serp_api_key or os.getenv("SERP_API_KEY")
    if not serp_api_key:
        serp_api_key = input("Enter Serp API Key for Search functionality: ").strip()
    
    if not serp_api_key:
        print("Error: SerpAPI key is required!")
        return
    
    prompt = args.prompt.strip()
    
    if not prompt:
        print("Error: Travel prompt is required!")
        return
    
    # Create agents
    researcher = Agent(
        name="Researcher",
        role="Searches for travel destinations, activities, and accommodations based on user preferences",
        model=ollama_model,
        description=dedent(
            """\
        You are a world-class travel researcher. Given a travel destination and the number of days the user wants to travel for,
        generate a list of search terms for finding relevant travel activities and accommodations.
        Then search the web for each term, analyze the results, and return the 10 most relevant results.
        """
        ),
        instructions=[
            "Given a travel destination and the number of days the user wants to travel for, first generate a list of 3 search terms related to that destination and the number of days.",
            "For each search term, `search_google` and analyze the results."
            "From the results of all searches, return the 10 most relevant results to the user's preferences.",
            "Remember: the quality of the results is important.",
        ],
        tools=[SerpApiTools(api_key=serp_api_key)],
        add_datetime_to_context=True,
    )
    
    planner = Agent(
        name="Planner",
        role="Generates a draft itinerary based on user preferences and research results",
        model=ollama_model,
        description=dedent(
            """\
        You are a senior travel planner. Given a travel destination, the number of days the user wants to travel for, and a list of research results,
        your goal is to generate a draft itinerary that meets the user's needs and preferences.
        """
        ),
        instructions=[
            "Given a travel destination, the number of days the user wants to travel for, and a list of research results, generate a draft itinerary that includes suggested activities and accommodations.",
            "Ensure the itinerary is well-structured, informative, and engaging.",
            "Ensure you provide a nuanced and balanced itinerary, quoting facts where possible.",
            "Remember: the quality of the itinerary is important.",
            "Focus on clarity, coherence, and overall quality.",
            "Never make up facts or plagiarize. Always provide proper attribution.",
        ],
        add_datetime_to_context=True,
    )
    
    # Generate itinerary
    print("\nGenerating your travel itinerary...\n")
    researcher_response: RunOutput = researcher.run(prompt, stream=False)
    response: RunOutput = planner.run(f"{prompt}. Search result by researcher {researcher_response.content}", stream=False)
    itinerary = response.content
    
    # Display the itinerary
    print("\n" + "=" * 60)
    print("YOUR TRAVEL ITINERARY")
    print("=" * 60 + "\n")
    print(itinerary)
    print("\n" + "=" * 60)


def call_api(prompt: str, options: dict, context: dict) -> dict:
    """
    Calls the AI travel planner agent with the provided prompt.
    Wraps the function call for Promptfoo.
    """
    try:
        ollama_model = Ollama(id="llama3.2:latest")

        # Get credentials from environment
        serp_api_key = os.environ.get("SERP_API_KEY")

        if not serp_api_key:
            return {"error": "Missing required environment variable: SERP_API_KEY"}
        
        researcher = Agent(
            name="Researcher",
            role="Searches for travel destinations, activities, and accommodations based on user preferences",
            model=ollama_model,
            description=dedent(
                """\
            You are a world-class travel researcher. Given a travel destination and the number of days the user wants to travel for,
            generate a list of search terms for finding relevant travel activities and accommodations.
            Then search the web for each term, analyze the results, and return the 10 most relevant results.
            """
            ),
            instructions=[
                "Given a travel destination and the number of days the user wants to travel for, first generate a list of 3 search terms related to that destination and the number of days.",
                "For each search term, `search_google` and analyze the results."
                "From the results of all searches, return the 10 most relevant results to the user's preferences.",
                "Remember: the quality of the results is important.",
            ],
            tools=[SerpApiTools(api_key=serp_api_key)],
            add_datetime_to_context=True,
        )
        
        planner = Agent(
            name="Planner",
            role="Generates a draft itinerary based on user preferences and research results",
            model=ollama_model,
            description=dedent(
                """\
            You are a senior travel planner. Given a travel destination, the number of days the user wants to travel for, and a list of research results,
            your goal is to generate a draft itinerary that meets the user's needs and preferences.
            """
            ),
            instructions=[
                "Given a travel destination, the number of days the user wants to travel for, and a list of research results, generate a draft itinerary that includes suggested activities and accommodations.",
                "Ensure the itinerary is well-structured, informative, and engaging.",
                "Ensure you provide a nuanced and balanced itinerary, quoting facts where possible.",
                "Remember: the quality of the itinerary is important.",
                "Focus on clarity, coherence, and overall quality.",
                "Never make up facts or plagiarize. Always provide proper attribution.",
            ],
            add_datetime_to_context=True,
        )

        researcher_response: RunOutput = researcher.run(prompt, stream=False)
        response: RunOutput = planner.run(f"{prompt}. Search result by researcher {researcher_response.content}", stream=False)
        result = response.content

        return {"output": result}

    except Exception as e:
        return {"error": f"An error occurred in call_api: {str(e)}"}


if __name__ == "__main__":
    main()