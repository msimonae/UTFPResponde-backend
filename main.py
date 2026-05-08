import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.graphs import Neo4jGraph
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Configuração de Observabilidade (LangSmith para Auditoria)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "UTFPResponde-Prod-V19"

app = FastAPI(title="API UTFPResponde", description="Assistente Agentivo do PPGI-UTFPR (Arquitetura V19 Blindada)")
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
            openai_api_base=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            openai_api_key=os.environ.get("OPENROUTER_API_KEY")
        )

        # 2. Vector DB - Carregamento explícito via Parquet
        caminho_vector_db = os.environ.get("SKLEARN_PATH", "sklearn_index_ppgi.parquet")
        if not os.path.exists(caminho_vector_db):
            raise FileNotFoundError(f"Arquivo vetorial {caminho_vector_db} não encontrado.")
        
        vector_db = SKLearnVectorStore(
            embedding=embeddings, 
            persist_path=caminho_vector_db,
            serializer="parquet"
        )

        # 3. Graph DB Neo4j (Conexão AuraDB)
        graph_db = Neo4jGraph(
            url=os.environ.get("NEO4J_URI"),
            username=os.environ.get("NEO4J_USERNAME"),
            password=os.environ.get("NEO4J_PASSWORD")
        )

        # 4. Tool de Busca Híbrida Blindada
        @tool
        def hybrid_normative_search(pergunta: str) -> str:
            """Ferramenta OBRIGATÓRIA para buscar regras, ementas e resoluções do PPGI-UTFPR. 
            Consulta o banco vetorial e expande a busca no grafo Neo4j via chunk_id."""
            try:
                docs = vector_db.similarity_search(pergunta, k=3)
            except Exception as e:
                return f"⚠️ Erro ao acessar o VectorStore: {str(e)}."

            contexto_agregado = []
            for d in docs:
                cid = d.metadata.get("chunk_id")
                fonte = d.metadata.get("source", "Documento PPGI")
                texto_base = d.page_content

                try:
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

        # 5. LLM Orquestrador
        llm_agente = ChatOpenAI(
            model="openai/gpt-4o-mini",
            temperature=0,
            openai_api_base=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            openai_api_key=os.environ.get("OPENROUTER_API_KEY")
        )

        # 6. Prompt do Sistema - Blindagem de Compliance
        system_message = (
            "Você é o Assistente Virtual UTFPResponde, especializado EXCLUSIVAMENTE em normas acadêmicas do PPGI-UTFPR.\n\n"
            "DIRETRIZES DE RIGOR INSTITUCIONAL E CONTEXTO IMPLÍCITO:\n"
            "1. ASSUNÇÃO DE CONTEXTO: Todas as perguntas referem-se ao Programa de Pós-Graduação em Informática (PPGI) da UTFPR.\n"
            "2. RESPOSTA BASEADA EM EVIDÊNCIAS: Use APENAS informações recuperadas pela ferramenta 'hybrid_normative_search'.\n"
            "3. POLÍTICA DE ABSTENÇÃO: Se a informação não for encontrada na busca, responda: 'Não localizei esta informação nas normas vigentes do PPGI'.\n"
            "4. TRACEABILITY: Sempre cite o documento fonte (ex: Resolução 01/2024 ou Ementa de Disciplinas).\n"
            "5. ZERO ALUCINAÇÃO: Nunca utilize conhecimento externo sobre outras universidades."
        )

        # 7. Inicialização do Agente com state_modifier
        agente_ppgi = create_react_agent(
            llm_agente, 
            ferramentas, 
            checkpointer=memoria_agente,
            state_modifier=system_message
        )

        logging.info("✅ Agente UTFPResponde pronto para produção.")

    except Exception as e:
        logging.error(f"❌ Erro crítico na inicialização: {e}")

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    if not agente_ppgi:
        raise HTTPException(status_code=500, detail="O sistema não foi inicializado corretamente.")
    
    try:
        config = {"configurable": {"thread_id": request.session_id}}
        
        # INJEÇÃO DE CONTEXTO HARDCODED (Blindagem contra abstenção em ementas)
        prompt_expandido = f"{request.query} (Contexto obrigatório para a busca: Programa do PPGI UTFPR)"
        
        response = agente_ppgi.invoke(
            {"messages": [("user", prompt_expandido)]},
            config=config
        )
        
        resposta_final = response["messages"][-1].content
        return QueryResponse(answer=resposta_final)

    except Exception as e:
        logging.error(f"Erro no endpoint de chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # A porta padrão do Cloud Run é 8080
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
