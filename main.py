import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

# Correção 1: Nova importação do Neo4j
from langchain_neo4j import Neo4jGraph 

from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "UTFPResponde-Prod"

app = FastAPI(title="API UTFPResponde", description="Assistente Agentivo do PPGI-UTFPR")
logging.basicConfig(level=logging.INFO)

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default_session"

class QueryResponse(BaseModel):
    answer: str

vector_db = None
graph_db = None
agente_ppgi = None
memoria_agente = MemorySaver()

@app.on_event("startup")
def startup_event():
    global vector_db, graph_db, agente_ppgi
    
    logging.info("⏳ Iniciando a carga do cérebro digital...")

    try:
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=os.environ.get("HF_TOKEN"),
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception as e:
        logging.error(f"Erro ao carregar Embeddings: {e}")
        embeddings = None

    if os.path.exists("faiss_index_ppgi") and embeddings:
        try:
            vector_db = FAISS.load_local("faiss_index_ppgi", embeddings, allow_dangerous_deserialization=True)
            logging.info("✅ Banco Vetorial FAISS carregado com sucesso!")
        except Exception as e:
            logging.error(f"Erro ao carregar o índice FAISS: {e}")
    else:
        logging.warning("⚠️ Pasta 'faiss_index_ppgi' não encontrada.")

    try:
        graph_db = Neo4jGraph(
            url=os.environ.get('NEO4J_URI'),
            username=os.environ.get('NEO4J_USERNAME'),
            password=os.environ.get('NEO4J_PASSWORD')
        )
        graph_db.query("RETURN 1")
        logging.info("✅ Conexão com o Grafo Neo4j estabelecida!")
    except Exception as e:
        logging.error(f"Erro ao conectar no banco Neo4j: {e}")
        graph_db = None

    @tool
    def search_vector_db(query: str) -> str:
        """Use esta ferramenta para buscar e ler Ementas de disciplinas, Bibliografias, Cargas Horárias, além de regras sobre Prazos, Regulamentos, Estágio e Créditos nas normativas."""
        if not vector_db: 
            return "O banco de dados vetorial está offline no momento."
        try:
            docs = vector_db.similarity_search(query, k=5)
            return "\n---\n".join([d.page_content for d in docs])
        except Exception as e:
            return f"Erro na busca vetorial: {e}"

    @tool
    def query_graph_db(entidade: str) -> str:
        """Use esta ferramenta APENAS para verificar conexões de PRÉ-REQUISITOS entre disciplinas no Grafo."""
        if not graph_db: 
            return "O banco de grafos está offline no momento."
        try:
            cypher = f"MATCH (d:Disciplina)-[r:TEM_PRE_REQUISITO]->(pre) WHERE d.nome CONTAINS '{entidade}' OR d.sigla CONTAINS '{entidade}' RETURN d.nome AS Disciplina, pre.nome AS Pre_Requisito"
            res = graph_db.query(cypher)
            return str(res) if res else "Nenhuma relação hierárquica/pré-requisito encontrada no Grafo."
        except Exception as e:
            return f"Erro na busca relacional: {e}"

    system_prompt = (
        "Você é o Agente Acadêmico Autônomo Oficial do PPGI-UTFPR. "
        "Sua missão é ajudar discentes de mestrado.\n"
        "REGRAS OBRIGATÓRIAS:\n"
        "1. SEMPRE utilize a ferramenta 'search_vector_db' para ler os textos de ementas, bibliografias e resoluções.\n"
        "2. Se a pergunta for ESPECIFICAMENTE sobre 'pré-requisitos' de disciplinas, use a ferramenta 'query_graph_db'.\n"
        "3. Ao responder, seja claro e cite os documentos oficiais e resoluções encontrados."
    )

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        # O state_modifier agora funcionará pois travamos o langgraph na versão >=0.2.0 no requirements
        agente_ppgi = create_react_agent(llm, [search_vector_db, query_graph_db], state_modifier=system_prompt, checkpointer=memoria_agente)
        logging.info("✅ Agente ReAct Autônomo inicializado com sucesso!")
    except Exception as e:
        logging.error(f"Erro crítico ao compilar o Agente LangGraph: {e}")

# Correção 2: Adicionada rota HEAD para o Health Check automático do Render
@app.head("/")
@app.get("/")
async def root_endpoint():
    return {"status": "online", "message": "API do UTFPResponde rodando!"}

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    if not agente_ppgi:
        raise HTTPException(status_code=500, detail="Erro 500: O Agente não inicializou corretamente. Verifique os logs.")
    
    try:
        config = {"configurable": {"thread_id": request.session_id}}
        response = agente_ppgi.invoke(
            {"messages": [HumanMessage(content=request.query)]},
            config=config
        )
        answer = response['messages'][-1].content
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))