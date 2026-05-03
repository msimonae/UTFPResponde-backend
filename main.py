import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.graphs import Neo4jGraph
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Configuração de Observabilidade
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "UTFPResponde-Prod-V13"

app = FastAPI(title="API UTFPResponde", description="Assistente Agentivo do PPGI-UTFPR (Arquitetura V13)")
logging.basicConfig(level=logging.INFO)

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default_session"

class QueryResponse(BaseModel):
    answer: str

# Variáveis globais do sistema
vector_db = None
graph_db = None
agente_ppgi = None
memoria_agente = MemorySaver()

@app.on_event("startup")
def startup_event():
    global vector_db, graph_db, agente_ppgi
    
    logging.info("⏳ Iniciando a carga do cérebro digital (SKLearn Vector-to-Graph)...")

    try:
        # 1. Configuração de Embeddings 
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", 
            openai_api_base=os.environ.get("OPENROUTER_BASE_URL"), 
            openai_api_key=os.environ.get("OPENROUTER_API_KEY")
        )
        
        # 2. Carregar SKLearnVectorStore (Arquivo .parquet local)
        # O arquivo sklearn_index_ppgi_v11.parquet DEVE estar na raiz do projeto no GitHub
        caminho_parquet = "./sklearn_index_ppgi_v11.parquet"
        
        if not os.path.exists(caminho_parquet):
            logging.error(f"❌ ARQUIVO NÃO ENCONTRADO: {caminho_parquet}. Suba o arquivo para o GitHub!")
            return

        vector_db = SKLearnVectorStore(embedding=embeddings, persist_path=caminho_parquet)
        logging.info("✅ Banco Vetorial SKLearn carregado com sucesso.")
        
        # 3. Conectar ao Neo4j
        graph_db = Neo4jGraph(
            url=os.environ.get("NEO4J_URI"), 
            username=os.environ.get("NEO4J_USERNAME"), 
            password=os.environ.get("NEO4J_PASSWORD")
        )

        # 4. Definição da Ferramenta Híbrida V13 (O Acoplamento MD5)
        @tool
        def hybrid_normative_search(pergunta: str) -> str:
            """Ferramenta OBRIGATÓRIA para buscar regras do PPGI-UTFPR. 
            Consulta o banco vetorial e expande a busca no grafo Neo4j via chunk_id."""
            docs = vector_db.similarity_search(pergunta, k=3)
            contexto_agregado = []

            for d in docs:
                cid = d.metadata.get("chunk_id")
                fonte = d.metadata.get("source", "Documento PPGI")
                texto_base = d.page_content

                try:
                    # Travessia no Grafo Neo4j
                    relacoes = graph_db.query("""
                        MATCH (c:Chunk {id: $cid})-[:MENTIONS]->(e)-[r]-(related)
                        RETURN e.id AS Entidade, type(r) AS Relacao, related.id AS Destino LIMIT 10
                    """, params={"cid": cid})
                    rels_str = ", ".join([f"({x['Entidade']} -> {x['Relacao']} -> {x['Destino']})" for x in relacoes]) if relacoes else "Nenhuma relação extra."
                except Exception as e:
                    logging.error(f"Erro no Neo4j: {e}")
                    rels_str = "Erro na consulta ao grafo."

                contexto_agregado.append(f"📄 FONTE: {fonte}\n📝 TEXTO: {texto_base}\n🔗 REGRAS CONECTADAS: {rels_str}")

            return "\n\n".join(contexto_agregado)

        ferramentas = [hybrid_normative_search]

        # 5. Configuração do LLM Orquestrador
        llm_agente = ChatOpenAI(
            model="openai/gpt-4o-mini",
            temperature=0,
            max_retries=3,
            max_tokens=1000,
            openai_api_base=os.environ.get("OPENROUTER_BASE_URL"),
            openai_api_key=os.environ.get("OPENROUTER_API_KEY")
        )

        # 6. Instruções de Sistema
        system_message = (
            "Você é o Agente Acadêmico do PPGI-UTFPR. "
            "SEMPRE use a ferramenta 'hybrid_normative_search' para consultar os dados. "
            "Se a resposta não estiver nos dados recuperados, RECUSE-SE a responder dizendo 'Informação não consta na normativa'. "
            "Sempre cite explicitamente a Resolução ou Portaria de onde a regra foi extraída."
        )

        # 7. Inicialização do Agente LangGraph
        agente_ppgi = create_react_agent(
            llm_agente, 
            ferramentas, 
            checkpointer=memoria_agente,
            messages_modifier=system_message
        )

        logging.info("✅ Agente V13 (SKLearn + Neo4j) pronto para inferência.")

    except Exception as e:
        logging.error(f"❌ Erro ao inicializar o sistema: {e}")

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    if not agente_ppgi:
        raise HTTPException(status_code=500, detail="Erro 500: Agente não inicializou. Verifique os logs.")
    
    try:
        config = {"configurable": {"thread_id": request.session_id}}
        
        response = agente_ppgi.invoke(
            {"messages": [("user", request.query)]},
            config=config
        )
        
        resposta_final = response["messages"][-1].content
        return QueryResponse(answer=resposta_final)
        
    except Exception as e:
        logging.error(f"Erro na geração: {e}")
        raise HTTPException(status_code=500, detail=str(e))
