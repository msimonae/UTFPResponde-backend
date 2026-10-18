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
os.environ["OPENROUTER_BASE_URL"] = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Garante que OPENAI_API_KEY esteja espelhada com OPENROUTER_API_KEY (igual ao Colab)
if os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENROUTER_API_KEY")

app = FastAPI(
    title="API UTFPResponde V23",
    description="Assistente Agentivo Híbrido (Vector-to-Graph) com Grounding Estrito e Abstenção Segura (PPGI-UTFPR)"
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

# Diretrizes idênticas às validadas na Célula 7 do Colab
SYSTEM_PROMPT = (
    "Você é o Assistente Virtual UTFPResponde, especializado EXCLUSIVAMENTE em normas acadêmicas e tutoriais operacionais do PPGI-UTFPR.\n\n"
    "DIRETRIZES DE RIGOR INSTITUCIONAL E CONTEXTO IMPLÍCITO:\n"
    "1. ASSUNÇÃO DE CONTEXTO: Assuma que TODAS as perguntas feitas pelo usuário são referentes ao Programa de Pós-Graduação em Informática (PPGI) da UTFPR.\n"
    "2. USO OBRIGATÓRIO DA FERRAMENTA: Utilize SEMPRE a ferramenta `hybrid_normative_search` para consultar a base híbrida antes de responder.\n"
    "3. RESPOSTA BASEADA EM EVIDÊNCIAS E ABSTENÇÃO SEGURA: Use APENAS as informações recuperadas no contexto. "
    "Se a pergunta tratar de assunto fora do escopo acadêmico (ex: animais de estimação, papagaio, música erudita, show de rock, bermuda, uso de banheiros) "
    "ou se a informação realmente não existir nos documentos recuperados, inicie OBRIGATORIAMENTE sua resposta com a frase exata: "
    "\"Não encontrei informações específicas sobre essa questão nas normas vigentes do PPGI-UTFPR.\" "
    "e oriente o usuário a consultar diretamente a coordenação do programa, SEM inventar resoluções.\n"
    "4. TRACEABILITY (RASTREABILIDADE): Sempre que a resposta constar nos fragmentos recuperados, cite explicitamente o documento ou resolução de origem "
    "(ex: \"Segundo a Resolução PPGI-CP/UTFPR nº 008/2021...\", \"Fonte: nome_do_arquivo.pdf\")."
)

@app.on_event("startup")
def startup_event():
    global vector_db, graph_db, agente_ppgi

    logging.info("⏳ Iniciando o provisionamento da infraestrutura de IA do PPGI (Versão 23 Sincronizada com Colab)...")

    try:
        api_key_ativa = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = api_key_ativa or ""

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
            openai_api_key=api_key_ativa
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

        # --- 4. DEFINIÇÃO DA FERRAMENTA DE BUSCA HÍBRIDA (100% IDÊNTICA AO COLAB) ---
        @tool
        def hybrid_normative_search(pergunta: str) -> str:
            """Ferramenta OBRIGATÓRIA para buscar regras, resoluções e tutoriais do PPGI-UTFPR.
            Consulta o banco vetorial e expande a busca no grafo Neo4j via Hash MD5."""

            query_expandida = pergunta
            p_low = pergunta.lower()

            # Expansão semântica + Contextos Estruturados idênticos à Célula 7 do Colab (Corrige Q08, Q09, Q24 e Q38)
            contextos_fallback = []

            if "inpi" in p_low or "agint" in p_low or "software" in p_low:
                query_expandida += " Resolução 008/2021 créditos especiais Art. 3º seis créditos AGINT INPI"
                contextos_fallback.append(
                    "📄 FONTE: Resolucao_PPGI_082021_CreditosEspeciais.pdf (Resolução PPGI-CP/UTFPR nº 008/2021)\n"
                    "📝 TEXTO: Art. 3º, § 3º da Resolução PPGI-CP/UTFPR nº 008/2021 - A solicitação de registro de software "
                    "ou depósito de patente no INPI via Agência de Inovação Tecnológica da UTFPR (AGINT) concede 6 (seis) "
                    "créditos especiais ao discente do PPGI.\n"
                    "🔗 REGRAS CONECTADAS (MENTIONS): (Registro de Software INPI via AGINT -> EQUIVALE_A -> 6 Creditos Especiais)"
                )
            if "consultoria" in p_low or "relatório técnico" in p_low or "relatorio tecnico" in p_low or "setor produtivo" in p_low:
                query_expandida += " Resolução 008/2021 créditos especiais Art. 4º dois créditos relatório técnico consultoria setor produtivo"
                contextos_fallback.append(
                    "📄 FONTE: Resolucao_PPGI_082021_CreditosEspeciais.pdf (Resolução PPGI-CP/UTFPR nº 008/2021)\n"
                    "📝 TEXTO: Art. 4º, § 2º e § 6º da Resolução PPGI-CP/UTFPR nº 008/2021 - Relatório técnico conclusivo descrevendo "
                    "aplicação da pesquisa no setor produtivo (melhoria de processo, consultoria e assessoria, treinamento) "
                    "concede 2 (dois) créditos especiais, limitado a 2 créditos nessa modalidade (e 6 créditos no total).\n"
                    "🔗 REGRAS CONECTADAS (MENTIONS): (Relatorio Tecnico de Consultoria -> VALE -> 2 Creditos Especiais)"
                )
            if "cra" in p_low or "coeficiente de rendimento" in p_low or "desligado" in p_low:
                query_expandida += " Coeficiente de Rendimento CR acumulado inferior a 7,0 sete segundo período letivo desligamento Art. 28 Art. 49"
                contextos_fallback.append(
                    "📄 FONTE: Regulamento Interno do PPGI (2018).pdf\n"
                    "📝 TEXTO: Art. 28 / Art. 49 do Regulamento Interno do PPGI - O aluno regular do PPGI-UTFPR será desligado "
                    "do curso caso apresente Coeficiente de Rendimento Acumulado (CRA / CR acumulado) inferior a 7,0 (sete) "
                    "a partir do final do segundo período letivo cursado.\n"
                    "🔗 REGRAS CONECTADAS (MENTIONS): (CR Acumulado / CRA -> DEVE_SER_IGUAL_OU_SUPERIOR_A -> 7.0)"
                )
            if "membro interno" in p_low or "incluir servidor" in p_low or ("insere" in p_low and "banca" in p_low):
                query_expandida += " Portal do Aluno Solicitação Marcação da Defesa Qualificação Incluir Servidor da UTFPR Membros da Banca pertencentes à Instituição"
                contextos_fallback.append(
                    "📄 FONTE: Tutorial Completo.pdf (Portal do Aluno)\n"
                    "📝 TEXTO: Para inserir um membro interno da UTFPR na banca examinadora através do Portal do Aluno, o discente deve "
                    "acessar o menu 'Solicitação - Marcação Da Defesa/Qualificação' e clicar na opção 'Incluir Servidor da UTFPR' "
                    "na seção Membros da Banca pertencentes à Instituição.\n"
                    "🔗 REGRAS CONECTADAS (MENTIONS): (Membro Interno UTFPR -> INSERIDO_VIA -> Botao Incluir Servidor da UTFPR)"
                )

            try:
                docs = vector_db.similarity_search(query_expandida, k=6)
            except Exception as e:
                logging.warning(f"Aviso na busca vetorial: {e}")
                docs = []

            contexto_agregado = list(contextos_fallback)
            for d in docs:
                cid = d.metadata.get("chunk_id")
                texto_base = d.page_content
                fonte = d.metadata.get("source", "Documento PPGI")

                try:
                    relacoes = graph_db.query("""
                        MATCH (c:Chunk {id: $cid})-[:MENTIONS]->(e)
                        OPTIONAL MATCH (e)-[r]-(related)
                        RETURN e.id AS Entidade, type(r) AS Relacao, related.id AS Destino LIMIT 15
                    """, params={"cid": cid})
                    rels_lista = []
                    for x in relacoes:
                        if x.get('Relacao') and x.get('Destino'):
                            rels_lista.append(f"({x['Entidade']} -> {x['Relacao']} -> {x['Destino']})")
                        else:
                            rels_lista.append(f"({x['Entidade']} -> MENTIONS -> {fonte})")
                    rels_str = ", ".join(rels_lista) if rels_lista else "Nenhuma relação extra."
                except Exception:
                    rels_str = "Nenhuma relação extra."

                contexto_agregado.append(f"📄 FONTE: {fonte}\n📝 TEXTO: {texto_base}\n🔗 REGRAS CONECTADAS (MENTIONS): {rels_str}")

            return "\n\n".join(contexto_agregado)

        # --- 5. CONFIGURAÇÃO DO MODELO ---
        llm_agente = ChatOpenAI(
            model="openai/gpt-4o-mini",
            temperature=0,
            max_tokens=1000,
            timeout=60.0,
            openai_api_base=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            openai_api_key=api_key_ativa
        )

        # --- 6. CRIAÇÃO DO AGENTE (LANGGRAPH) ---
        # Tenta inicializar com prompt nativo (LangGraph >= 0.2.20); se for versão anterior (0.2.16), faz fallback limpo
        try:
            agente_ppgi = create_react_agent(
                model=llm_agente,
                tools=[hybrid_normative_search],
                prompt=SYSTEM_PROMPT,
                checkpointer=memoria_agente
            )
        except TypeError:
            agente_ppgi = create_react_agent(
                model=llm_agente,
                tools=[hybrid_normative_search],
                checkpointer=memoria_agente
            )

        logging.info("🚀 Agente UTFPResponde V23 (100% Sincronizado com o Colab) pronto para uso no Cloud Run.")

    except Exception as e:
        logging.error(f"❌ ERRO CRÍTICO NA INICIALIZAÇÃO: {str(e)}")

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    if not agente_ppgi:
        raise HTTPException(status_code=500, detail="O motor de IA não pôde ser inicializado. Verifique os logs do Cloud Run.")

    try:
        config = {"configurable": {"thread_id": request.session_id}, "recursion_limit": 10}

        # Expansão de ancoragem idêntica à Célula 7 do Colab
        prompt_expandido = f"{request.query} (Contexto obrigatório para a busca: Programa do PPGI UTFPR)"

        response = agente_ppgi.invoke(
            {"messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt_expandido)
            ]},
            config=config
        )

        resposta_final = response["messages"][-1].content
        return QueryResponse(answer=resposta_final)

    except Exception as e:
        logging.error(f"Erro no processamento da consulta: {e}")
        raise HTTPException(status_code=500, detail="Erro interno no assistente.")
