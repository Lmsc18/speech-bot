from dotenv import load_dotenv
load_dotenv()
import os
index_name=os.getenv("INDEX")
from langchain_community.embeddings import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.output_parsers import PydanticOutputParser,OutputFixingParser
from langchain.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap,RunnableLambda, RunnablePassthrough,RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from gtts import gTTS 
import uvicorn

language = 'en'

llm=ChatGoogleGenerativeAI(model='gemini-pro',temperature=0)

embed = CohereEmbeddings(model="embed-english-v3.0")

vs = PineconeVectorStore(index_name=index_name, embedding=embed)
retr=vs.as_retriever(search_kwargs={"k": 1})

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-tiny.en"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)


class PO(BaseModel):
        answered: bool = Field(description="""True if the question is answered. False if the question is not answered""")


parser = PydanticOutputParser(pydantic_object=PO)
fixing_parser = OutputFixingParser.from_llm(
    parser=parser,
    llm=llm 
)

temp="""
You are an AI Answer validator.
You are given a question and a response.
Your task is to carefully look at the question and the answer and conclude that the question is answered using the answer or not.

Question: {{qns}}
Answer: {% for item in ans %}
    content : {{item.page_content}}
    {% endfor %}

Output Format:
{{format_instructions}}
    """

prompt = PromptTemplate(template=temp,
                        input_variables=['qns','ans'],
                        partial_variables={"format_instructions": fixing_parser.get_format_instructions()},
                        template_format="jinja2"
                        )

chain = RunnableParallel({
    "ans": retr,
    "qns":RunnablePassthrough()
}) | prompt | llm | fixing_parser


def generate(query):
    o=retr.get_relevant_documents(query)
    resp=o[0].page_content
    status=chain.invoke(query).dict()['answered']
    if status==True:
          fa=resp
    else:
          fa='I am sorry. I dont have the answer to your question'
    myobj = gTTS(text=fa, lang=language, slow=False,tld='us') 
    myobj.save("response/resp.mp3") 

def empty_dir(folder_path):
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        os.remove(file_path)
    print(f'Removed files from {folder_path}')

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
app = FastAPI()

@app.post("/inference")
async def upload_audio(file: UploadFile):
    input_path = 'input'
    empty_dir(input_path)
    output_path="response"
    empty_dir(output_path)
    file_contents = await file.read()
    with open("input/input.mp3", "wb") as f:
        f.write(file_contents)
    print('saved audio')
    result=pipe("input/input.mp3")['text']
    generate(result)
    
    return FileResponse("response/resp.mp3", media_type="audio/mpeg")


if __name__ == "__main__":
    # Run the FastAPI application using uvicorn
    uvicorn.run(app,host="0.0.0.0", port=8000)
    
