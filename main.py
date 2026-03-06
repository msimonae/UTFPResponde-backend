# main.py (Backend FastAPI)
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPI
from langchain_community.graphs import Neo4jGraph
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

# Configuração de Observabilidade (LangSmith) para monitorar Drift
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "UTFPResponde-Prod"

app = FastAPI(title="API UTFPResponde", description="Microserviço do Agente PPGI")
logging.basicConfig(level=logging.INFO)

# Modelos de Entrada/Saída da API
class QueryRequest(BaseModel):
    query: str
    session_id: str = "default_session"

class QueryResponse(BaseModel):
    answer: str

# Variáveis globais para instâncias de banco
vector_db = None
graph_db = None
agente_ppgi = None

@app.on_event("startup")
def startup_event():
    """Carrega os bancos de dados e o agente na inicialização do servidor."""
    global vector_db, graph_db, agente_ppgi
    try:
        logging.info("Carregando FAISS e Neo4j...")
        embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=os.environ.get("HF_TOKEN"), model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Atenção: O diretório 'faiss_index_ppgi' deve ser enviado para o servidor junto com o código
        vector_db = FAISS.load_local("faiss_index_ppgi", embeddings, allow_dangerous_deserialization=True)
        graph_db = Neo4jGraph(
            url=os.environ.get('NEO4J_URI'),
            username=os.environ.get('NEO4J_USERNAME'),
            password=os.environ.get('NEO4J_PASSWORD')
        )
        
        # Tools (Simulando o MCP Server)
        @tool
        def search_vector_db(query: str) -> str:
            """Busca regras sobre Prazos, Regulamentos, Estágio e Créditos."""
            docs = vector_db.similarity_search(query, k=3)
            return "\n".join([d.page_content for d in docs])

        @tool
        def query_graph_db(entidade: str) -> str:
            """Verifica pré-requisitos entre disciplinas no Grafo."""
            cypher = f"MATCH (d:Disciplina)-[r:TEM_PRE_REQUISITO]->(pre) WHERE d.nome CONTAINS '{entidade}' RETURN d.nome, pre.nome"
            res = graph_db.query(cypher)
            return str(res) if res else "Nenhuma relação encontrada."

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        system_prompt = "Você é o Agente Acadêmico Autônomo Oficial do PPGI-UTFPR. Responda baseando-se nas ferramentas."
        agente_ppgi = create_react_agent(llm, [search_vector_db, query_graph_db], state_modifier=system_prompt)
        logging.info("Agente inicializado com sucesso!")
    except Exception as e:
        logging.error(f"Erro na inicialização: {e}")

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """Endpoint que recebe a pergunta e retorna a resposta do Agente."""
    if not agente_ppgi:
        raise HTTPException(status_code=500, detail="Agente não está pronto.")
    try:
        # A requisição invoca o LangGraph. O LangSmith registra automaticamente o rastro (trace).
        response = agente_ppgi.invoke(
            {"messages": [("user", request.query)]},
            config={"configurable": {"thread_id": request.session_id}}
        )
        answer = response['messages'][-1].content
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))