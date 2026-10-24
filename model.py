from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List , Optional
from langchain_core.output_parsers import PydanticOutputParser


load_dotenv()

model = ChatGroq(
    model = "openai/gpt-oss-20b",
    temperature=0.3
)

class PersonDetails(BaseModel):
    name : str
    Age : Optional[str]
    Education : Optional[str]
    skill : List[str]
    college : Optional[str]


parser= PydanticOutputParser(pydantic_object=PersonDetails)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     """ 
    You are an information extraction assistant.
    {format_instruction}

        Tasks:
        1. Extract only information explicitly present in the text.
        2. Never invent missing information.
        3. Preserve the original meaning.
        4. Follow the required output schema.
        5. Use null for missing single values and empty lists
           for missing collections.

        Extract information about the person, skills,
        experience, company, projects, and career goals.
        """
     )
     ,
     (
         "human",
         """ 
    Extract information from this paragraph:\n{text}
    """
     )
])

para = input(" Enter paragraph : ")

final_promt = prompt.invoke(
    {"text":para,
    'format_instruction':parser.get_format_instructions()
      }
    )

response = model.invoke(final_promt)
personal_data = parser.parse(response.content)

print(personal_data)