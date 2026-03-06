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
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# ---------------------------------------------------------
# Configuração de Observabilidade (LangSmith) para MLOps
# ---------------------------------------------------------
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "UTFPResponde-Prod"

app = FastAPI(title="API UTFPResponde", description="Assistente Agentivo do PPGI-UTFPR")
logging.basicConfig(level=logging.INFO)

# Modelos Pydantic para a API
class QueryRequest(BaseModel):
    query: str
    session_id: str = "default_session"

class QueryResponse(BaseModel):
    answer: str

# Variáveis globais de infraestrutura
vector_db = None
graph_db = None
agente_ppgi = None
memoria_agente = MemorySaver() # Mantém o histórico da conversa em RAM

@app.on_event("startup")
def startup_event():
    """Inicialização resiliente dos módulos do cérebro (Vetor, Grafo e Agente)."""
    global vector_db, graph_db, agente_ppgi
    
    logging.info("⏳ Iniciando a carga do cérebro digital...")

    # 1. Carregamento dos Embeddings via API (Economiza RAM no Render)
    try:
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=os.environ.get("HF_TOKEN"),
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception as e:
        logging.error(f"Erro ao carregar Embeddings da HuggingFace: {e}")
        embeddings = None

    # 2. Carregamento do FAISS (RAG Vetorial)
    if os.path.exists("faiss_index_ppgi") and embeddings:
        try:
            vector_db = FAISS.load_local("faiss_index_ppgi", embeddings, allow_dangerous_deserialization=True)
            logging.info("✅ Banco Vetorial FAISS carregado com sucesso!")
        except Exception as e:
            logging.error(f"Erro ao carregar o índice FAISS: {e}")
    else:
        logging.warning("⚠️ Pasta 'faiss_index_ppgi' não encontrada no servidor.")

    # 3. Conexão com Neo4j (GraphRAG)
    try:
        graph_db = Neo4jGraph(
            url=os.environ.get('NEO4J_URI'),
            username=os.environ.get('NEO4J_USERNAME'),
            password=os.environ.get('NEO4J_PASSWORD')
        )
        graph_db.query("RETURN 1") # Ping para testar a conexão
        logging.info("✅ Conexão com o Grafo Neo4j estabelecida!")
    except Exception as e:
        logging.error(f"Erro ao conectar no banco Neo4j: {e}")
        graph_db = None

    # ---------------------------------------------------------
    # Ferramentas (Tools) do Agente (Simulando Endpoints MCP)
    # ---------------------------------------------------------
    @tool
    def search_vector_db(query: str) -> str:
        """Use esta ferramenta para buscar e ler Ementas de disciplinas, Bibliografias, Cargas Horárias, além de regras sobre Prazos, Regulamentos, Estágio e Créditos nas normativas."""
        if not vector_db: 
            return "O banco de dados vetorial está offline no momento."
        try:
            # Aumentado para k=5 para não cortar ementas longas
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
            return str(res) if res else "Nenhuma relação hierárquica/pré-requisito encontrada no Grafo para esta disciplina."
        except Exception as e:
            return f"Erro na busca relacional: {e}"

    # ---------------------------------------------------------
    # Prompt de Sistema e Criação do Agente
    # ---------------------------------------------------------
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
        # O state_modifier injeta o prompt de sistema, e o checkpointer cuida da memória da conversa
        agente_ppgi = create_react_agent(llm, [search_vector_db, query_graph_db], state_modifier=system_prompt, checkpointer=memoria_agente)
        logging.info("✅ Agente ReAct Autônomo inicializado com sucesso!")
    except Exception as e:
        logging.error(f"Erro crítico ao compilar o Agente LangGraph: {e}")

# ---------------------------------------------------------
# Endpoints da API
# ---------------------------------------------------------
@app.get("/")
async def root_endpoint():
    """Health check endpoint."""
    return {"status": "online", "message": "API do UTFPResponde rodando perfeitamente!"}

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """Endpoint principal de interação do Agente."""
    if not agente_ppgi:
        raise HTTPException(status_code=500, detail="Erro 500: O Agente não inicializou corretamente. Verifique os logs.")
    
    try:
        # A thread_id permite que o agente reconheça o usuário e o histórico da sessão
        config = {"configurable": {"thread_id": request.session_id}}
        
        # Invocando o Agente (o LangGraph lidará com a memória automaticamente)
        response = agente_ppgi.invoke(
            {"messages": [HumanMessage(content=request.query)]},
            config=config
        )
        
        # A última mensagem da lista é a resposta final gerada pelo LLM
        answer = response['messages'][-1].content
        return QueryResponse(answer=answer)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro durante o processamento da resposta: {str(e)}")