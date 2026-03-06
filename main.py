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
from langgraph.checkpoint.memory import MemorySaver # <-- Importação da Memória

# Observabilidade
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "UTFPResponde-Prod"

app = FastAPI(title="API UTFPResponde")
logging.basicConfig(level=logging.INFO)

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default_session"

class QueryResponse(BaseModel):
    answer: str

vector_db = None
graph_db = None
agente_ppgi = None
memoria_agente = MemorySaver() # <-- Instancia a memória do Agente

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
        logging.error(f"Erro nos Embeddings: {e}")
        embeddings = None

    if os.path.exists("faiss_index_ppgi") and embeddings:
        try:
            vector_db = FAISS.load_local("faiss_index_ppgi", embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            logging.error(f"Erro ao carregar FAISS: {e}")

    try:
        graph_db = Neo4jGraph(
            url=os.environ.get('NEO4J_URI'),
            username=os.environ.get('NEO4J_USERNAME'),
            password=os.environ.get('NEO4J_PASSWORD')
        )
        graph_db.query("RETURN 1")
    except Exception as e:
        logging.error(f"Erro ao conectar no Neo4j: {e}")
        graph_db = None

    @tool
    def search_vector_db(query: str) -> str:
        """Use esta ferramenta para buscar regras sobre Prazos, Regulamentos, Estágio e Créditos nas INs e Resoluções."""
        if not vector_db: return "Banco de dados vetorial offline."
        try:
            docs = vector_db.similarity_search(query, k=3)
            return "\n---\n".join([d.page_content for d in docs])
        except Exception as e:
            return f"Erro na busca vetorial: {e}"

    @tool
    def query_graph_db(entidade: str) -> str:
        """Use esta ferramenta APENAS para verificar pré-requisitos entre disciplinas das ementas ou ligações lógicas estruturadas no Grafo."""
        if not graph_db: return "Grafo offline."
        try:
            # Query Cypher exata do seu Colab original
            cypher = f"MATCH (d:Disciplina)-[r:TEM_PRE_REQUISITO]->(pre) WHERE d.nome CONTAINS '{entidade}' OR d.sigla CONTAINS '{entidade}' RETURN d.nome AS Disciplina, pre.nome AS Pre_Requisito"
            res = graph_db.query(cypher)
            return str(res) if res else "Nenhuma relação hierárquica encontrada no Grafo para esta disciplina."
        except Exception as e:
            return f"Erro na busca relacional: {e}"

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        # Agente agora possui memória (checkpointer) para reter contexto!
        agente_ppgi = create_react_agent(llm, [search_vector_db, query_graph_db], checkpointer=memoria_agente)
        logging.info("✅ Agente Autônomo inicializado!")
    except Exception as e:
        logging.error(f"Erro crítico no agente: {e}")

@app.get("/")
async def root_endpoint():
    return {"status": "online"}

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    if not agente_ppgi:
        raise HTTPException(status_code=500, detail="Agente não está pronto.")
    
    try:
        # Prompt EXATO do Colab, restaurando a capacidade autônoma de decisão
        system_prompt = (
            "Você é o Agente Acadêmico Autônomo Oficial do PPGI-UTFPR. "
            "Sua missão é ajudar discentes de mestrado. "
            "SEMPRE utilize as ferramentas de busca (Vetor e Grafo) para basear sua resposta em documentos oficiais. "
            "Se a pergunta envolver regras e também ementas, invoque ambas as ferramentas sequencialmente."
        )
        
        # Thread_id garante que o agente lembre da conversa com este usuário específico
        config = {"configurable": {"thread_id": request.session_id}}
        
        response = agente_ppgi.invoke(
            {"messages": [SystemMessage(content=system_prompt), HumanMessage(content=request.query)]},
            config=config
        )
        answer = response['messages'][-1].content
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))