import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

# Configuração de Observabilidade (LangSmith)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "UTFPResponde-Prod"

app = FastAPI(title="API UTFPResponde")
logging.basicConfig(level=logging.INFO)

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default_session"

class QueryResponse(BaseModel):
    answer: str

# Variáveis globais
vector_db = None
graph_db = None
agente_ppgi = None

@app.on_event("startup")
def startup_event():
    """Carrega os bancos de dados de forma resiliente, inspirado no pdm_v0."""
    global vector_db, graph_db, agente_ppgi
    
    logging.info("⏳ Iniciando a carga do cérebro digital...")

    # 1. Carregar Embeddings
    try:
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=os.environ.get("HF_TOKEN"),
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception as e:
        logging.error(f"Erro nos Embeddings: {e}")
        embeddings = None

    # 2. Carregar FAISS (Try/Except isolado)
    if os.path.exists("faiss_index_ppgi") and embeddings:
        try:
            vector_db = FAISS.load_local("faiss_index_ppgi", embeddings, allow_dangerous_deserialization=True)
            logging.info("✅ FAISS carregado com sucesso!")
        except Exception as e:
            logging.error(f"Erro ao carregar FAISS: {e}")
    else:
        logging.warning("⚠️ Pasta 'faiss_index_ppgi' ausente. RAG vetorial offline.")

    # 3. Carregar Neo4j (Try/Except isolado)
    try:
        graph_db = Neo4jGraph(
            url=os.environ.get('NEO4J_URI'),
            username=os.environ.get('NEO4J_USERNAME'),
            password=os.environ.get('NEO4J_PASSWORD')
        )
        graph_db.query("RETURN 1") # Testa a conexão
        logging.info("✅ Neo4j carregado com sucesso!")
    except Exception as e:
        logging.error(f"Erro ao conectar no Neo4j: {e}")
        graph_db = None

    # 4. Ferramentas Resilientes (Exatamente como no pdm_v0)
    @tool
    def search_vector_db(query: str) -> str:
        """Busca regras sobre Prazos, Regulamentos, Estágio e Créditos nas normativas."""
        if not vector_db: return "Banco de dados vetorial offline no momento."
        try:
            docs = vector_db.similarity_search(query, k=3)
            return "\n---\n".join([d.page_content for d in docs])
        except Exception as e:
            return f"Erro na busca vetorial: {e}"

    @tool
    def query_graph_db(entidade: str) -> str:
        """Verifica pré-requisitos entre disciplinas no Grafo."""
        if not graph_db: return "Grafo de conhecimento offline no momento."
        try:
            cypher = f"MATCH (d:Disciplina)-[r:TEM_PRE_REQUISITO]->(pre) WHERE d.nome CONTAINS '{entidade}' OR d.sigla CONTAINS '{entidade}' RETURN d.nome, pre.nome"
            res = graph_db.query(cypher)
            return str(res) if res else "Nenhuma relação encontrada."
        except Exception as e:
            return f"Erro na busca relacional do grafo: {e}"

    # 5. Criar o Agente (sem o state_modifier problemático)
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        agente_ppgi = create_react_agent(llm, [search_vector_db, query_graph_db])
        logging.info("✅ Agente Autônomo inicializado e pronto!")
    except Exception as e:
        logging.error(f"Erro crítico ao instanciar o agente: {e}")

@app.get("/")
async def root_endpoint():
    return {"status": "online", "message": "API do UTFPResponde rodando!"}

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    if not agente_ppgi:
        raise HTTPException(status_code=500, detail="Ocorreu um erro crítico na inicialização. Verifique os logs do Render.")
    
    try:
        # Injetando o Prompt de Sistema como Mensagem (Corrigido conforme o pdm_v0)
        system_msg = SystemMessage(content="Você é o Agente Acadêmico Autônomo Oficial do PPGI-UTFPR. Responda baseando-se sempre nas ferramentas de banco de dados. Cite a origem.")
        user_msg = HumanMessage(content=request.query)
        
        response = agente_ppgi.invoke(
            {"messages": [system_msg, user_msg]},
            config={"configurable": {"thread_id": request.session_id}}
        )
        answer = response['messages'][-1].content
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))