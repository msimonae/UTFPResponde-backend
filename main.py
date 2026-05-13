# Instalação das bibliotecas
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.graphs import Neo4jGraph
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

# Configuração de Observabilidade
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "UTFPResponde-Compliance-V19"

app = FastAPI(
    title="API UTFPResponde", 
    description="Assistente Agentivo com Grounding Estrito e Segurança IAM (PPGI-UTFPR)"
)
logging.basicConfig(level=logging.INFO)

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default_session"

class QueryResponse(BaseModel):
    answer: str

# Variáveis globais de estado
vector_db = None
graph_db = None
agente_ppgi = None
memoria_agente = MemorySaver()

@app.on_event("startup")
def startup_event():
    global vector_db, graph_db, agente_ppgi
    
    logging.info("⏳ Iniciando o provisionamento da infraestrutura de IA do PPGI...")

    try:
        # --- 1. DOWNLOAD SEGURO DO GOOGLE CLOUD STORAGE ---
        BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "utfpresponde-vectordb-data")
        BLOB_NAME = "sklearn_index_ppgi.parquet"
        LOCAL_PATH = "/tmp/sklearn_index_ppgi.parquet"

        logging.info(f"📥 Acessando bucket: {BUCKET_NAME}")
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(BLOB_NAME)
        blob.download_to_filename(LOCAL_PATH)
        logging.info("✅ Base vetorial carregada com sucesso do GCS para o ambiente local.")

        # --- 2. CONFIGURAÇÃO DE EMBEDDINGS E VECTOR STORE ---
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_base=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            openai_api_key=os.environ.get("OPENROUTER_API_KEY")
        )
        
        vector_db = SKLearnVectorStore(
            embedding=embeddings, 
            persist_path=LOCAL_PATH
        )

        # --- 3. CONEXÃO COM O KNOWLEDGE GRAPH (NEO4J) ---
        graph_db = Neo4jGraph(
            url=os.environ.get("NEO4J_URI"),
            username=os.environ.get("NEO4J_USERNAME"),
            password=os.environ.get("NEO4J_PASSWORD")
        )

        # --- 4. DEFINIÇÃO DA FERRAMENTA DE BUSCA HÍBRIDA ---
        @tool
        def hybrid_normative_search(pergunta: str) -> str:
            """Consulta regras e ementas acadêmicas no PPGI-UTFPR."""
            query_expandida = f"{pergunta} PPGI UTFPR"
            
            docs = vector_db.similarity_search(query_expandida, k=3)
            contexto_agregado = []

            for d in docs:
                cid = d.metadata.get("chunk_id")
                texto_base = d.page_content
                fonte = d.metadata.get("source", "Documento PPGI")

                try:
                    relacoes = graph_db.query("""
                        MATCH (c:Chunk {id: $cid})-[:MENTIONS]->(e)
                        OPTIONAL MATCH (e)-[r]-(related)
                        RETURN e.id AS Entidade, type(r) AS Tipo, related.id AS Destino LIMIT 5
                    """, params={"cid": cid})
                    rels_str = "; ".join([f"({x['Entidade']} {x['Tipo']} {x['Destino']})" for x in relacoes]) if relacoes else "Sem conexões extras."
                except:
                    rels_str = "Erro na consulta ao grafo."

                contexto_agregado.append(f"FONTE: {fonte}\nCONTEÚDO: {texto_base}\nRELAÇÕES: {rels_str}")

            return "\n\n---\n\n".join(contexto_agregado)

        # --- 5. CONFIGURAÇÃO DO MODELO E PROMPT DE CONFORMIDADE ---
        llm_agente = ChatOpenAI(
            model="openai/gpt-4o-mini",
            temperature=0, 
            openai_api_base=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            openai_api_key=os.environ.get("OPENROUTER_API_KEY")
        )

        system_message = (
            "Você é o Assistente Virtual UTFPResponde. Especialista em normas do PPGI-UTFPR.\n\n"
            "DIRETRIZES DE AUDITORIA:\n"
            "1. RESPONDA APENAS com base nos dados recuperados pela ferramenta 'hybrid_normative_search'.\n"
            "2. Se a informação não for encontrada nos dados, diga: 'Não localizei esta informação específica nas normas do PPGI'.\n"
            "3. EXIBA SEMPRE a fonte da informação (Resolução, Portaria ou Documento).\n"
            "4. Nunca utilize seu conhecimento geral para inventar regras ou prazos acadêmicos."
        )

        # --- 6. CRIAÇÃO DO AGENTE (LANGGRAPH) ---
        # CORREÇÃO: Utilizando messages_modifier no lugar de state_modifier
        agente_ppgi = create_react_agent(
            llm_agente, 
            [hybrid_normative_search], 
            checkpointer=memoria_agente,
            messages_modifier=system_message
        )

        logging.info("🚀 Agente UTFPResponde V19 (GCS-Private) pronto para uso.")

    except Exception as e:
        logging.error(f"❌ ERRO CRÍTICO NA INICIALIZAÇÃO: {str(e)}")

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    if not agente_ppgi:
        raise HTTPException(status_code=500, detail="O motor de IA não pôde ser inicializado. Verifique os logs do Cloud Run.")
    
    try:
        config = {"configurable": {"thread_id": request.session_id}}
        prompt_expandido = f"{request.query} (Focar em: Programa do PPGI UTFPR)"
        
        response = agente_ppgi.invoke(
            {"messages": [HumanMessage(content=prompt_expandido)]},
            config=config
        )
        
        resposta_final = response["messages"][-1].content
        return QueryResponse(answer=resposta_final)

    except Exception as e:
        logging.error(f"Erro no processamento da consulta: {e}")
        raise HTTPException(status_code=500, detail="Erro interno no assistente.")
