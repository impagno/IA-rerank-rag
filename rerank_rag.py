#!/usr/bin/env python
# coding: utf-8

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

import dotenv
dotenv.load_dotenv() 

# LISTANDO AS PERGUNTAS A SEREM RESPONDIDAS
questions = [
    "Qual é a visão de Euclides da Cunha sobre o ambiente natural do sertão nordestino e como ele influencia a vida dos habitantes?",
    "Quais são as principais características da população sertaneja descritas por Euclides da Cunha? Como ele relaciona essas características com o ambiente em que vivem?",
    "Qual foi o contexto histórico e político que levou à Guerra de Canudos, segundo Euclides da Cunha?",
    "Como Euclides da Cunha descreve a figura de Antônio Conselheiro e seu papel na Guerra de Canudos?",
    "Quais são os principais aspectos da crítica social e política presentes em \"Os Sertões\"? Como esses aspectos refletem a visão do autor sobre o Brasil da época?",
]

# Carregando o modelo de embeddings da OpenAI
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # Modelo de embedding para gerar vetores

# Inicializando o modelo de linguagem ChatGPT (gpt-3.5-turbo)
# Este modelo será usado para gerar respostas para as perguntas com base no contexto
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # Nome do modelo de linguagem
    max_tokens=500,              # Número máximo de tokens na resposta gerada
)

# Carregando o PDF
# O carregador (loader) será responsável por carregar o PDF e dividir o conteúdo em páginas
pdf_link = "os-sertoes.pdf"  # Caminho para o arquivo PDF

# Inicializando o carregador de PDF
# O parâmetro 'extract_images=False' garante que imagens não sejam extraídas do PDF
loader = PyPDFLoader(pdf_link, extract_images=False)

# Carregando o conteúdo do PDF e dividindo-o em páginas
# A função 'load_and_split' divide o PDF em uma lista de páginas para processamento posterior
pages = loader.load_and_split()

# Separando o conteúdo em chunks (pedaços menores)
# O objetivo é dividir o conteúdo do PDF em pedaços menores para processamento posterior
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,          # Tamanho máximo de cada pedaço (chunk)
    chunk_overlap=20,         # Sobreposição entre os pedaços (chunks)
    length_function=len,      # Função para calcular o comprimento dos pedaços
    add_start_index=True      # Adiciona o índice de início a cada pedaço
)

# Dividindo o conteúdo do PDF (páginas) em pedaços menores (chunks)
chunks = text_splitter.split_documents(pages)

# Salvando os chunks no VectorDB
# O banco de dados vetorial (VectorDB) armazena os pedaços (chunks) em formato vetorial para recuperação posterior
vectordb = Chroma(embedding_function=embeddings)  # Criando o banco de dados vetorial utilizando o modelo de embeddings

# Carregando os documentos (chunks) no VectorDB
# Aqui, estamos armazenando os pedaços (chunks) no banco de dados vetorial para que possam ser recuperados mais tarde
vectordb.add_documents(chunks)

# Carregando o DB e configurando o recuperador (retriever)
# O recuperador irá buscar os 10 documentos mais relevantes baseados na consulta
naive_retriever = vectordb.as_retriever(search_kwargs={"k": 10})

# Configurando o Reranker para melhorar a qualidade da recuperação
# O reranker é responsável por reclassificar os documentos retornados pelo recuperador, mantendo os mais relevantes
rerank = CohereRerank(model="rerank-v3.5", top_n=3)  # Rerank com o modelo "rerank-v3.5", retornando os 3 melhores resultados

# Criando o ContextualCompressionRetriever
# O compressor contextual é responsável por combinar o reranker com o recuperador base para melhorar a precisão das respostas
compressor_retriever = ContextualCompressionRetriever(
    base_compressor=rerank,       # Usando o reranker para melhorar a recuperação
    base_retriever=naive_retriever,  # Usando o recuperador base (naive_retriever)
)

# Definindo o template para o prompt de Chat
# O template será usado para estruturar a pergunta e o contexto que serão passados para o modelo de linguagem
TEMPLATE = """
Você é um especialista em literatura brasileira. Responda a pergunta abaixo utilizando o contexto informado

Contexto: {context}
    
Pergunta: {question}
"""

# Criando o prompt do chat a partir do template definido
# 'ChatPromptTemplate.from_template' cria um prompt com base no template fornecido
rag_prompt = ChatPromptTemplate.from_template(TEMPLATE)

# Configurando o processo de recuperação paralelo
# A recuperação paralelo permite passar a pergunta e o contexto para o recuperador de forma independente
setup_retrieval = RunnableParallel({"question": RunnablePassthrough(), "context": compressor_retriever})  # Processamento paralelo para pergunta e contexto

# Inicializando o parser para a saída
# O parser converte a resposta gerada pelo modelo de linguagem para um formato desejado
output_parser = StrOutputParser()

# Criando a cadeia de recuperação com compressão
# O fluxo de execução inclui o setup de recuperação, o prompt, o modelo de linguagem e o parser de saída
compressor_retrieval_chain = setup_retrieval | rag_prompt | llm | output_parser

# Função para responder a uma pergunta utilizando o fluxo de recuperação com compressão
# A função usa a cadeia de recuperação configurada para processar a pergunta e gerar uma resposta
def answer_question(question: str):
    # Chamando a cadeia de recuperação para processar a pergunta e retornar a resposta
    return compressor_retrieval_chain.invoke(question)

# Iterando sobre as perguntas e obtendo as respostas
# 'enumerate' permite obter tanto o índice quanto a pergunta de cada item na lista
for index, question in enumerate(questions):
    # Obtendo a resposta para a pergunta atual utilizando a função 'answer_question'
    resposta = answer_question(question)
    # Exibindo o número da pergunta, a pergunta e a resposta gerada
    print({"numero": index, "pergunta": question, "resposta": resposta})

