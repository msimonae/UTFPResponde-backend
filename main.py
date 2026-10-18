import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_neo4j import Neo4jGraph
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage

# Configuração de Observabilidade
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "UTFPResponde-Compliance-V23"

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

# Diretrizes alinhadas com o Notebook V23 (Abstenção Segura e Rastreabilidade)
SYSTEM_PROMPT = (
    "Você é o Assistente Virtual UTFPResponde, especializado EXCLUSIVAMENTE em normas acadêmicas e tutoriais operacionais do PPGI-UTFPR.\n\n"
    "DIRETRIZES DE AUDITORIA E RIGOR INSTITUCIONAL:\n"
    "1. ASSUNÇÃO DE CONTEXTO: Assuma que TODAS as perguntas são referentes ao PPGI da UTFPR.\n"
    "2. USO OBRIGATÓRIO DA FERRAMENTA: Utilize SEMPRE a ferramenta 'hybrid_normative_search' para consultar a base híbrida antes de responder.\n"
    "3. RESPOSTA BASEADA EM EVIDÊNCIAS E ABSTENÇÃO SEGURA: Use APENAS as informações recuperadas no contexto. "
    "Se a pergunta tratar de assunto fora do escopo acadêmico ou se a informação não existir nos documentos recuperados, "
    "inicie OBRIGATORIAMENTE sua resposta com a frase exata: 'Não encontrei informações específicas sobre essa questão nas normas vigentes do PPGI-UTFPR.' "
    "e oriente o usuário a consultar diretamente a coordenação do programa.\n"
    "4. TRACEABILITY (RASTREABILIDADE): Sempre que a resposta constar nos fragmentos recuperados, cite explicitamente o documento ou artigo de origem."
)

@app.on_event("startup")
def startup_event():
    global vector_db, graph_db, agente_ppgi
    
    logging.info("⏳ Iniciando o provisionamento da infraestrutura de IA do PPGI (Versão 23)...")

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

        # --- 4. DEFINIÇÃO DA FERRAMENTA DE BUSCA HÍBRIDA COM EXPANSÃO SEMÂNTICA ---
        @tool
        def hybrid_normative_search(pergunta: str) -> str:
            """Ferramenta OBRIGATÓRIA para buscar regras, resoluções e tutoriais do PPGI-UTFPR."""
            query_expandida = pergunta
            p_low = pergunta.lower()
            
            # Expansão inteligente idêntica à Célula 4.5 do Notebook V23
            if "cra" in p_low or "coeficiente de rendimento" in p_low:
                query_expandida += " Coeficiente de Rendimento CR acumulado inferior a 7,0 sete segundo período letivo desligamento Art. 28 Art. 49"
            if "inpi" in p_low or "agint" in p_low or "relatório técnico" in p_low or "consultoria" in p_low or "créditos especiais" in p_low:
                query_expandida += " Resolução 008/2021 créditos especiais Art. 3º seis créditos Art. 4º dois créditos AGINT INPI setor produtivo"
            if "membro interno" in p_low or ("insere" in p_low and "banca" in p_low) or "incluir servidor" in p_low:
                query_expandida += " Portal do Aluno Solicitação Marcação da Defesa Qualificação Incluir Servidor da UTFPR Membros da Banca pertencentes à Instituição"

            # Aumentado de k=3 para k=6 para garantir recall em questões complexas
            docs = vector_db.similarity_search(query_expandida, k=6)
            contexto_agregado = []

            for d in docs:
                cid = d.metadata.get("chunk_id")
                texto_base = d.page_content
                fonte = d.metadata.get("source", "Documento PPGI")

                try:
                    relacoes = graph_db.query("""
                        MATCH (c:Chunk {id: $cid})-[:MENTIONS]->(e)
                        OPTIONAL MATCH (e)-[r]-(related)
                        RETURN e.id AS Entidade, type(r) AS Tipo, related.id AS Destino LIMIT 10
                    """, params={"cid": cid})
                    rels_str = "; ".join([f"({x['Entidade']} {x['Tipo']} {x['Destino']})" for x in relacoes]) if relacoes else "Sem conexões extras."
                except:
                    rels_str = "Erro na consulta ao grafo."

                contexto_agregado.append(f"FONTE: {fonte}\nCONTEÚDO: {texto_base}\nRELAÇÕES: {rels_str}")

            return "\n\n---\n\n".join(contexto_agregado)

        # --- 5. CONFIGURAÇÃO DO MODELO ---
        llm_agente = ChatOpenAI(
            model="openai/gpt-4o-mini",
            temperature=0, 
            openai_api_base=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            openai_api_key=os.environ.get("OPENROUTER_API_KEY")
        )

        # --- 6. CRIAÇÃO DO AGENTE (LANGGRAPH) ---
        agente_ppgi = create_react_agent(
            model=llm_agente, 
            tools=[hybrid_normative_search], 
            checkpointer=memoria_agente
        )

        logging.info("🚀 Agente UTFPResponde V23 (Híbrido + Expansão Semântica) pronto para uso no Cloud Run.")

    except Exception as e:
        logging.error(f"❌ ERRO CRÍTICO NA INICIALIZAÇÃO: {str(e)}")

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    if not agente_ppgi:
        raise HTTPException(status_code=500, detail="O motor de IA não pôde ser inicializado. Verifique os logs do Cloud Run.")
    
    try:
        config = {"configurable": {"thread_id": request.session_id}}
        
        # Injetando as diretrizes de sistema dinamicamente na invocação junto com a query
        response = agente_ppgi.invoke(
            {"messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=request.query)
            ]},
            config=config
        )
        
        resposta_final = response["messages"][-1].content
        return QueryResponse(answer=resposta_final)

    except Exception as e:
        logging.error(f"Erro no processamento da consulta: {e}")
        raise HTTPException(status_code=500, detail="Erro interno no assistente.")
