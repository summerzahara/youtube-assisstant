from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from icecream import ic

load_dotenv()

embeddings = OpenAIEmbeddings()

# video_url = "https://www.youtube.com/watch?v=JMUxmLyrhSk&t=3890s"
# input = "What is reinforcement learning?"


def create_yt_vector_db(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db


yt_assistant_prompt = """
You are a helpful YouTube assistant that can answer questions about videos based on the video's transcript.

Answer the following question: {question}
By searching the following video transcript: {docs}

Only user the factual information from the transcript to answer the question.

If you feel like you don't have enough information to answer the question, say 'I don't know'.

Your answers should be detailed.
"""


def get_query_response(db, query, k=4):
    # gpt-3.5-turbo-instruct can handle 4096 tokens
    docs = db.similarity_search(
        query,
        k=k,  # 4*1000 = 4000 (compare to token max)
    )

    doc_page_content = " ".join(
        [d.page_content for d in docs]
    )

    llm = OpenAI(model='gpt-3.5-turbo-instruct')

    prompt = PromptTemplate(
        input_variables=['question', 'docs'],
        template=yt_assistant_prompt,
    )

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
    )

    response = chain.run(
        question=query,
        docs=doc_page_content,
    ).replace("\n", "")

    return response


# my_db = create_yt_vector_db(video_url)
# result = get_query_response(my_db, input)
#
# ic(result)
