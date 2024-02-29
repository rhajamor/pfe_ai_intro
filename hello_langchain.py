from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

application_prompt ="""Make sure that all reponses are in json format, 
here is the template file 
 json goes her .... 
    DESCRIPTION:
    {user_input}
"""
user_input =  """temperature= ..; ph=....; ... """

llm = ChatOpenAI(
    #base_url="http://localhost:1234/v1",
    temperature=0.7,
    max_tokens=500,
    model='gpt-3.5-turbo'
)
prompt = PromptTemplate(  
    input_variables=["user_input"],
    template=application_prompt
)

#using LCEL lanchain expression language
chain = prompt | llm | StrOutputParser()

result = chain.invoke({"user_input": user_input})

print(result)

# for streaing use
#results = chain.stream({"user_input": user_input})
#for chunk in results:
#    print(chunk, end='')
