import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.llms.openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor


def main():
    load_dotenv()
    # os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']

    st.set_page_config(page_title='Ask your PDF')
    st.header("Ask your PDF")

    file = st.file_uploader('upload file' ,type='pdf')

    if file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        # memory = ConversationBufferMemory(memory_key="chat_history")

        splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)

        chunks = splitter.split_text(text)

        embeddings = OpenAIEmbeddings()

        knowledge = FAISS.from_texts(chunks, embeddings)

        question = st.text_input("Question here:")

        if question:
            docs = knowledge.similarity_search(question)
            chain = load_qa_chain(OpenAI(), chain_type='stuff')
            # agent_chain = AgentExecutor.from_agent_and_tools(agent=chain, verbose=True, memory=memory)
            answer = chain.run(input_documents=docs, question=question)

            st.write(answer)


if __name__ == "__main__":
    main()