from dotenv import load_dotenv, find_dotenv
import os

# Load the environment variables from .env file
load_dotenv(find_dotenv())

from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from langchain.chat_models import init_chat_model
from langchain.chains import LLMChain


# create a model instance
model = init_chat_model(
    "command-r-plus",
    model_provider="cohere",
    model_kwargs={"api_key": os.getenv("COHERE_API_KEY")}
)
model.invoke("Hello, world!")

def generate_res_name_and_items(cuisine):

    #Chain 1 -> name
    prompt_template_name=PromptTemplate(
        input_variables=['cuisine'],
        template="I want to open a restaurant for {cuisine} food. Sugges a fency name for it(only one), just give the name in response."
    )

    # add a ouptut key here for the next chain to get it 
    name_chain=LLMChain(llm=model, prompt=prompt_template_name, output_key='restaurant_name')

    # Chain 2 -> items
    prompt_template_items=PromptTemplate(
        input_variables=['restaurant_name'],
        template=" Suggest some food menu items(names only) for restaraunt {restaurant_name} food. Return in a list format, just give the list items nothing else."
    )

    food_items_chain=LLMChain(llm=model, prompt=prompt_template_items, output_key='menu_items')

    # --------------------------------------------------------------------------------
    # Creating a sequential chain for above two chains

    seq_chain=SequentialChain(
        chains=[name_chain, food_items_chain],
        input_variables=['cuisine'],
        output_variables=['restaurant_name', 'menu_items'],
        verbose=True
    )

    # no run here to run a sequential chain
    response=seq_chain({'cuisine': cuisine})    
    return response

if __name__=="__main__":
    print(generate_res_name_and_items("Italian"))