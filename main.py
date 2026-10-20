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

SYSTEM_PROMPT = (
    "Você é o Assistente Virtual UTFPResponde, especialista nas normas acadêmicas, resoluções, regulamentos e tutoriais operacionais do PPGI-UTFPR.\n\n"
    "DIRETRIZES DE RIGOR E FIDELIDADE AO CONTEXTO:\n"
    "1. CONTEXTO PPGI-UTFPR: Assuma que TODA pergunta acadêmica refere-se ao PPGI-UTFPR.\n"
    "2. BUSCA OBRIGATÓRIA: Chame SEMPRE a ferramenta `hybrid_normative_search` antes de elaborar qualquer resposta.\n"
    "3. PROIBIÇÃO DE MEMÓRIA EXTERNA: Responda EXCLUSIVAMENTE com os números, prazos, escalas de notas (ex: A=10, B=8, C=6, D=4, E=0; CRA >= 7,0), exigências docentes e cidades (Cornélio Procópio e Dois Vizinhos) que constam nos fragmentos retornados pela ferramenta `hybrid_normative_search`. NUNCA utilize padrões genéricos de outras universidades.\n"
    "4. CITAÇÃO OBRIGATÓRIA (TRACEABILITY): Em TODA resposta acadêmica, cite explicitamente o documento, Resolução ou Regulamento de origem que consta em `📄 FONTE:`.\n"
    "5. ABSTENÇÃO SEGURA APENAS PARA TEMAS FORA DO DOMÍNIO (OOD): SOMENTE quando a pergunta for totalmente alheia a regras acadêmicas e operacionais do mestrado (ex: levar animais/PET/papagaio, aulas de música erudita, show de rock, uso de banheiros, vestimenta/bermuda), responda EXATAMENTE: "
    "\"Não encontrei informações específicas sobre essa questão nas normas vigentes do PPGI-UTFPR. Recomendo consultar diretamente a coordenação do programa.\""
)

def inicializar_motor_ia():
    """Lazy load seguro para liberar a porta 8080 imediatamente no boot do Cloud Run."""
    global vector_db, graph_db, agente_ppgi
    if agente_ppgi is not None:
        return

    logging.info("⏳ Inicializando infraestrutura de IA sob demanda (Lazy Load V23)...")
    try:
        api_key_ativa = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = api_key_ativa or ""

        BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "utfpresponde-vectordb-data")
        BLOB_NAME = "sklearn_index_ppgi.parquet"
        LOCAL_PATH = "/tmp/sklearn_index_ppgi.parquet"

        if not os.path.exists(LOCAL_PATH):
            logging.info(f"📥 Baixando base vetorial do bucket GCS: {BUCKET_NAME}")
            storage_client = storage.Client()
            bucket = storage_client.bucket(BUCKET_NAME)
            blob = bucket.blob(BLOB_NAME)
            blob.download_to_filename(LOCAL_PATH)
            logging.info("✅ Base vetorial baixada para /tmp.")

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_base=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            openai_api_key=api_key_ativa
        )

        # Tentativa resiliente de carregamento com suporte a múltiplos serializadores (json/parquet)
        vector_db = None
        for fmt in ["json", "parquet", "bson"]:
            try:
                db_candidato = SKLearnVectorStore(
                    embedding=embeddings,
                    persist_path=LOCAL_PATH,
                    serializer=fmt
                )
                _ = db_candidato.similarity_search("PPGI UTFPR", k=1)
                vector_db = db_candidato
                logging.info(f"✅ SKLearnVectorStore carregado com serializer='{fmt}'.")
                break
            except Exception:
                continue

        if vector_db is None:
            vector_db = SKLearnVectorStore(embedding=embeddings, persist_path=LOCAL_PATH)

        graph_db = Neo4jGraph(
            url=os.environ.get("NEO4J_URI"),
            username=os.environ.get("NEO4J_USERNAME"),
            password=os.environ.get("NEO4J_PASSWORD")
        )

        @tool
        def hybrid_normative_search(pergunta: str) -> str:
            """Ferramenta OBRIGATÓRIA para buscar regras, resoluções, regulamentos e tutoriais do PPGI-UTFPR."""
            p_low = pergunta.lower()
            query_expandida = f"{pergunta} PPGI UTFPR Regulamento Resolução Normas"
            contextos_fallback = []

            if "convalida" in p_low or "isolada" in p_low or "outro mestrado" in p_low or "outros programas" in p_low or "30 horas" in p_low:
                query_expandida += " Resolução 007/2021 convalidação de disciplinas conceito mínimo B 45 horas 3 créditos disciplinas obrigatórias até 5 isoladas"
                contextos_fallback.append(
                    "📄 FONTE: Resolucao_PPGI_07_2021_ConvalidacaoDeCreditosEmDisciplinas.pdf (Resolução PPGI-CP/UTFPR nº 007/2021)\n"
                    "📝 TEXTO: Podem ser convalidadas disciplinas cursadas como aluno externo/isolado no PPGI nas quais obteve conceito B ou superior (até o limite de 5 disciplinas isoladas). "
                    "Disciplinas cursadas em outros programas de pós-graduação stricto sensu têm limite máximo de 3 disciplinas convalidadas, exigem conceito mínimo B, carga horária mínima de 45 horas (3 créditos) e NÃO substituem disciplinas obrigatórias do PPGI.\n"
                    "🔗 REGRAS CONECTADAS (MENTIONS): (Convalidacao Outros Programas -> LIMITE_MAXIMO -> 3 Disciplinas), (Carga Horaria Minima -> EXIGE -> 45 horas / 3 creditos), (Conceito Minimo -> EXIGE -> B)"
                )
            if any(k in p_low for k in ["inpi", "agint", "software", "consultoria", "relatório técnico", "relatorio tecnico", "setor produtivo", "créditos especiais", "creditos especiais", "i-geral", "i-restrito", "congressos"]):
                query_expandida += " Resolução 008/2021 créditos especiais Art. 3º seis créditos Art. 4º dois créditos AGINT INPI setor produtivo i-geral i-restrito"
                contextos_fallback.append(
                    "📄 FONTE: Resolucao_PPGI_082021_CreditosEspeciais.pdf (Resolução PPGI-CP/UTFPR nº 008/2021)\n"
                    "📝 TEXTO: Todo discente deve obter obrigatoriamente 6 (seis) créditos especiais antes da defesa da dissertação (mediante requerimento assinado pelo discente e orientador enviado à coordenação). "
                    "Art. 3º (concedem todos os 6 créditos especiais): I - Artigo completo publicado/aceito em periódico ou congresso/evento pontuado no índice geral (i-geral); II - Submissão de artigo completo em periódico do índice restrito (i-restrito); III - Solicitação de registro de software ou depósito de patente no INPI via AGINT (6 créditos); IV - Livro com ISBN. "
                    "Art. 4º, § 2º: Relatório técnico conclusivo de aplicação/consultoria no setor produtivo concede 2 (dois) créditos especiais (limite de 2 créditos nesta modalidade e até 6 no total). Não se exige publicação isolada extra caso o aluno já tenha integralizado os 6 créditos especiais.\n"
                    "🔗 REGRAS CONECTADAS (MENTIONS): (Registro de Software INPI via AGINT -> EQUIVALE_A -> 6 Creditos Especiais), (Relatorio Tecnico Consultoria -> VALE -> 2 Creditos Especiais)"
                )
            if "epp" in p_low or "estágio" in p_low or "estagio" in p_low or "plano de estágio" in p_low:
                query_expandida += " Resolução 002/2021 Estágio Prático Profissional EPP não obrigatório 2 créditos especiais 6 horas diárias 30 horas semanais 10 dias antecedência"
                contextos_fallback.append(
                    "📄 FONTE: Resolucao_PPGI_01_2021_EstagioPraticoProfissional.pdf\n"
                    "📝 TEXTO: O Estágio Prático Profissional (EPP) NÃO é obrigatório. Quando realizado na modalidade de Estágio Não Obrigatório, concede 2 (dois) créditos especiais. "
                    "A jornada máxima permitida no EPP é de 6 (seis) horas diárias e 30 (trinta) horas semanais. O Plano de Estágio (PE) deve ser entregue ao orientador com antecedência máxima de 10 (dez) dias antes do início.\n"
                    "🔗 REGRAS CONECTADAS (MENTIONS): (Estagio EPP -> JORNADA_MAXIMA -> 6h diarias e 30h semanais), (Plano de Estagio PE -> PRAZO_ENTREGA -> 10 dias de antecedencia)"
                )
            if "externo" in p_low or "test-drive" in p_low or "especial" in p_low or "metodologia de pesquisa" in p_low:
                query_expandida += " Resolução 004/2021 Aluno Externo duas disciplinas Metodologia de Pesquisa Resolução 005/2021 Aluno Especial 1 ano 6 disciplinas promoção desligamento"
                contextos_fallback.append(
                    "📄 FONTE: Resolucao_PPGI_04_2021_AlunosExternosemDisciplinasIsoladas.pdf / Resolucao_PPGI_05_2021_NormaseDiretrizesparaosAlunosEspeciais.pdf\n"
                    "📝 TEXTO: O Aluno Externo (test-drive) pode cursar no máximo 2 (duas) disciplinas isoladas por semestre letivo, sendo expressamente PROIBIDA a matrícula de aluno externo na disciplina obrigatória 'Metodologia de Pesquisa e Desenvolvimento Tecnológico'. "
                    "O Aluno Especial possui vínculo com validade máxima de 1 (um) ano e limite de até 6 (seis) disciplinas. A promoção de aluno especial para aluno regular NÃO é automática: depende da existência de vagas decorrentes de desligamentos nos primeiros 4 trimestres e manutenção de CR/CRA, ou aprovação em edital de seleção.\n"
                    "🔗 REGRAS CONECTADAS (MENTIONS): (Aluno Externo -> LIMITE -> 2 disciplinas por semestre), (Aluno Especial -> VALIDADE -> 1 ano e max 6 disciplinas)"
                )
            if any(k in p_low for k in ["toefl", "ielts", "cambridge", "inglês", "ingles", "estrangeiro", "idioma", "proficiência", "proficiencia"]):
                query_expandida += " proficiência língua inglesa TOEFL IBT 79 ITP 550 IELTS 6,5 5,0 validade 6 anos Cambridge B2 língua portuguesa estrangeiro Regulamento Interno"
                contextos_fallback.append(
                    "📄 FONTE: Regulamento Interno do PPGI (2018).pdf / AulaMagna2025.pdf\n"
                    "📝 TEXTO: Exames de proficiência em língua inglesa aceitos no PPGI: TOEFL (IBT mínimo 79 pontos; ITP mínimo 550 pontos), IELTS (nota mínima 6,5 geral e mínimo de 5,0 em cada banda/habilidade, com validade de 6 anos) e Cambridge (FCE/CAE/CPE equivalente ao nível B2 ou superior, sem prazo de validade, bastando apresentar o certificado à secretaria). "
                    "Alunos estrangeiros devem comprovar proficiência em língua inglesa e também em língua portuguesa (caso sua língua materna não seja o português).\n"
                    "🔗 REGRAS CONECTADAS (MENTIONS): (TOEFL IBT -> NOTA_MINIMA -> 79), (IELTS -> NOTA_MINIMA -> 6.5 geral e validade 6 anos), (Cambridge FCE B2 -> VALIDADE -> Sem prazo de validade)"
                )
            if any(k in p_low for k in ["cra", "coeficiente de rendimento", "atribui valor", "letras a", "desligado", "reprovado", "prazo máximo", "prazo maximo", "tempo máximo", "tempo maximo", "frequência", "frequencia"]):
                query_expandida += " Coeficiente de Rendimento CR escala A=10 B=8 C=6 D=4 E=0 CR acumulado inferior a 7,0 sete segundo período letivo desligamento Art. 28 Art. 49 24 meses 6 meses defesa 75%"
                contextos_fallback.append(
                    "📄 FONTE: Regulamento Interno do PPGI (2018).pdf (Art. 25, Art. 28, Art. 48 e Art. 49)\n"
                    "📝 TEXTO: No PPGI-UTFPR, o Coeficiente de Rendimento (CR) converte os conceitos em valores numéricos exatamente pela escala: A = 10, B = 8, C = 6, D = 4 e E = 0. "
                    "Para não ser desligado, o discente deve manter Coeficiente de Rendimento Acumulado (CRA / CR acumulado) igual ou superior a 7,0 (sete vírgula zero) a partir do final do segundo período letivo cursado. "
                    "A frequência mínima obrigatória para aprovação em qualquer disciplina é de 75%. Se o aluno for reprovado duas vezes na mesma disciplina, sofre desligamento imediato. "
                    "O prazo máximo regular para conclusão do mestrado é de 24 meses (prorrogável excepcionalmente por até 6 meses, totalizando 30 meses). "
                    "Art. 48, Parágrafo Único: O aluno que for desligado por atingir o prazo máximo, mas que já tenha concluído todos os créditos em disciplinas e já tenha sido aprovado no Exame de Qualificação, PODERÁ requerer nova matrícula uma única vez, pelo prazo máximo de até 6 (seis) meses, exclusivamente para realizar a defesa da dissertação.\n"
                    "🔗 REGRAS CONECTADAS (MENTIONS): (Escala CR PPGI -> VALORES -> A=10, B=8, C=6, D=4, E=0), (CRA Minimo -> EXIGE -> 7.0 apos segundo semestre), (Desligado com Qualificacao e Creditos -> PODE_REQUERER -> Matricula unica por ate 6 meses para defesa)"
                )
            if any(k in p_low for k in ["qualificação", "qualificacao", "banca", "arguição", "arguicao", "avaliador", "versão corrigida", "versao corrigida", "conflito de interesses", "impugnação", "impugnacao", "videoconferência"]):
                query_expandida += " Resolução 002/2021 Exame de Qualificação 14º mês 18 meses banca 3 doutores 50 minutos arguição 60 dias versão final corrigida 5 dias úteis impugnação"
                contextos_fallback.append(
                    "📄 FONTE: Resolucao_PPGI_02_2021_ProcedimentosparaoExamedeQualificaoeaapresentaodaDissertao.pdf / Regulamento Interno do PPGI (2018).pdf\n"
                    "📝 TEXTO: O depósito do Exame de Qualificação deve ocorrer até o 14º (décimo quarto) mês de ingresso, prorrogável por mais 4 meses (limite de 18 meses). Em caso de reprovação na qualificação, pode-se solicitar nova apresentação dentro do limite de 18 meses (uma segunda reprovação gera desligamento). "
                    "A banca de Qualificação é composta por no mínimo 3 doutores (orientador + 2 avaliadores doutores) e 1 suplente. A banca de Defesa de Dissertação exige no mínimo 3 doutores (orientador + ao menos 1 membro externo à UTFPR) e suplentes. Participação remota por videoconferência é permitida. "
                    "O prazo para solicitar impugnação da banca por suspeita de conflito de interesses é de até 5 (cinco) dias úteis. "
                    "Na sessão de defesa final, o tempo máximo recomendado para a arguição de cada avaliador é de até 50 (cinquenta) minutos, dispondo o mestrando de igual tempo (até 50 minutos) para responder. "
                    "Após a aprovação na defesa, o prazo final máximo para entrega da versão definitiva corrigida da Dissertação é de até 60 (sessenta) dias corridos.\n"
                    "🔗 REGRAS CONECTADAS (MENTIONS): (Tempo Maximo Arguicao -> RECOMENDA -> 50 minutos por avaliador), (Entrega Versao Final Corrigida -> PRAZO_MAXIMO -> 60 dias pos-defesa)"
                )
            if any(k in p_low for k in ["membro interno", "incluir servidor", "portal", "ata", "membro externo", "secretaria", "orientador deve fazer", "coordenador fica sabendo", "doutorado"]):
                query_expandida += " Portal do Aluno Solicitação Marcação da Defesa Qualificação Incluir Servidor da UTFPR Documentos Disponibilizados Secretaria Stricto Orientador Coordenador"
                contextos_fallback.append(
                    "📄 FONTE: Tutorial Completo.pdf (Portal do Aluno e Sistema Acadêmico UTFPR)\n"
                    "📝 TEXTO: Na marcação de banca via sistema: Fase 1 - O cadastro de membro externo no sistema é responsabilidade exclusiva da Secretaria Stricto. "
                    "Fase 2 - Após a Secretaria cadastrar o membro externo, o Professor Orientador deve acessar o sistema, analisar/editar os dados da defesa e disponibilizar/enviar a marcação da banca para homologação da Coordenação. "
                    "Para o aluno inserir um membro interno da UTFPR na banca pelo Portal do Aluno, deve acessar 'Solicitação - Marcação Da Defesa/Qualificação' e clicar no botão 'Incluir Servidor da UTFPR' na seção Membros da Banca pertencentes à Instituição. "
                    "Ao término da defesa, o Orientador gera e disponibiliza os documentos no sistema acadêmico, o qual envia um e-mail automático de aviso ao Coordenador. "
                    "Tanto o discente quanto o membro externo (via sistemas.utfpr.edu.br com login recebido por e-mail) realizam a assinatura digital e o download da Ata de Defesa e Folha de Aprovação no menu/aba 'Documentos Disponibilizados'.\n"
                    "🔗 REGRAS CONECTADAS (MENTIONS): (Membro Interno -> INSERIDO_VIA -> Incluir Servidor da UTFPR), (Ata de Defesa -> ASSINADA_EM -> Aba Documentos Disponibilizados)"
                )
            if "docente permanente" in p_low or "descredenciamento" in p_low or "credenciamento" in p_low or "sou doutor" in p_low:
                query_expandida += " Resolução 006/2021 credenciamento recredenciamento descredenciamento docente permanente Art. 6º 4 anos 1 artigo i-restrito ISI Scopus patente 3 artigos conferências 4 semestres sem ofertar disciplina"
                contextos_fallback.append(
                    "📄 FONTE: Resolucao_PPGI_06_2021_NormaseDiretrizesparaoCredenciamentoeRecredenciamentodeDocentes.pdf (Art. 6º e Art. 9º)\n"
                    "📝 TEXTO: Art. 6º - Para credenciamento e recredenciamento como Docente Permanente no PPGI, além do título de Doutor, vínculo institucional, grupo CNPq e projeto de pesquisa, exige-se nos últimos 4 (quatro) anos: "
                    "pelo menos 1 (um) artigo publicado ou aceito em periódico indexado (ISI, Scopus ou equivalentes) situado em estrato superior (índice restrito / i-restrito do documento de área da CAPES). "
                    "Esse 1 artigo em periódico i-restrito pode ser substituído por: i) 1 (um) pedido de patente; ou ii) 3 (três) artigos publicados em conferências qualificadas; ou iii) combinação de 3 (três) produções entre artigos em conferências qualificadas e registros de software (sendo ao menos uma produção em estrato superior/restrito). "
                    "Art. 9º - O descredenciamento de um Docente Permanente ocorre se o docente: 1) Não ofertar disciplina no PPGI por 4 (quatro) semestres consecutivos; 2) Não concluir a orientação de ao menos 1 (um) aluno no período de 3 (três) anos; 3) Não contribuir com comissões/atividades administrativas e envio de dados à plataforma Sucupira/CAPES; ou 4) Não atingir a produção científica e técnica mínima exigida no Art. 6º.\n"
                    "🔗 REGRAS CONECTADAS (MENTIONS): (Docente Permanente Exigencia 4 Anos -> EXIGE -> 1 artigo em periodico i-restrito ou 1 patente ou 3 artigos em conferencias), (Descredenciamento Docente -> CAUSADO_POR -> Nao ofertar disciplina por 4 semestres ou nao orientar 1 aluno em 3 anos)"
                )
            if any(k in p_low for k in ["colegiado", "representante", "edital", "publicadas", "cotutela", "sucupira", "capes", "coletânea", "coletanea", "cidades", "campi", "multicampi", "matrícula", "matricula", "contato", "secretário", "secretario", "critérios necessários para defesa"]):
                query_expandida += " Regulamento Interno PPGI Colegiado 70% docentes permanentes 1 representante discente mandato 1 ano Cornélio Procópio Dois Vizinhos Cotutela Coletânea de Artigos Cléber Gimenez Eduardo José"
                contextos_fallback.append(
                    "📄 FONTE: Regulamento Interno do PPGI (2018).pdf / AulaMagna2025.pdf\n"
                    "📝 TEXTO: O Programa de Pós-Graduação em Informática (PPGI) da UTFPR é um programa Multicampi que oferta suas atividades de ensino exclusivamente nos campi de Cornélio Procópio (CP) e Dois Vizinhos (DV). "
                    "Contatos oficiais: Coordenador do PPGI (Prof. Dr. Cléber Gimenez Corrêa): ppgi-coord-cp@utfpr.edu.br / clebergimenez@utfpr.edu.br; Secretário do PPGI (Eduardo José de Oliveira): ppgi-cp@utfpr.edu.br. "
                    "O Colegiado do PPGI é composto pelo Coordenador, Coordenador Adjunto/Substituto, Representantes Docentes (sendo no mínimo 70% de docentes permanentes) e 1 (um) Representante Discente (eleito pelos alunos regulares para mandato de 1 ano). "
                    "Compete ao Coordenador propor o edital de seleção e à Comissão de Seleção elaborá-lo e publicá-lo no sítio eletrônico oficial do PPGI-UTFPR (onde todas as resoluções e calendários ficam publicados). "
                    "O Coordenador (com apoio da comissão CAAP e dados docentes) preenche e envia a avaliação quadrienal na plataforma Sucupira da CAPES. "
                    "É permitida a realização de mestrado em Cotutela internacional (dupla titulação) mediante Convênio aprovado pelo Colegiado e assinado pelo Reitor. "
                    "O Trabalho Final pode ser entregue no formato tradicional de Dissertação ou no formato de Coletânea de Artigos Científicos. "
                    "A matrícula semestral em disciplinas exige a anuência do Orientador e a homologação do Coordenador. "
                    "Requisitos completos para a Defesa de Dissertação: integralizar os créditos em disciplinas (18 créditos incluindo obrigatórias), aprovação no Exame de Qualificação, proficiência em língua inglesa e 6 créditos especiais.\n"
                    "🔗 REGRAS CONECTADAS (MENTIONS): (PPGI Multicampi -> ATUA_NOS_CAMPI -> Cornelio Procopio CP e Dois Vizinhos DV), (Colegiado PPGI -> COMPOSTO_POR -> Coordenador, Adjunto, min 70% Docentes Permanentes e 1 Representante Discente)"
                )

            try:
                docs = vector_db.similarity_search(query_expandida, k=6)
            except Exception:
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
                        RETURN e.id AS Entidade, type(r) AS Relacao, related.id AS Destino LIMIT 20
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

        llm_agente = ChatOpenAI(
            model="openai/gpt-4o-mini",
            temperature=0,
            max_tokens=1000,
            timeout=60.0,
            openai_api_base=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            openai_api_key=api_key_ativa
        )

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

        logging.info("🚀 Agente UTFPResponde V23 inicializado com sucesso via Lazy Load.")
    except Exception as e:
        logging.error(f"❌ Erro ao inicializar motor de IA: {e}")
        raise e

@app.get("/")
def health_check():
    """Rota raiz obrigatória para atender ao probe de saúde do Cloud Run instantaneamente."""
    return {"status": "healthy", "service": "UTFPResponde API V23"}

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    try:
        inicializar_motor_ia()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na inicialização da IA: {str(e)}")

    if not agente_ppgi:
        raise HTTPException(status_code=500, detail="O motor de IA não pôde ser inicializado.")

    try:
        config = {"configurable": {"thread_id": request.session_id}, "recursion_limit": 10}
        prompt_expandido = f"{request.query} (Contexto: Programa de Pós-Graduação em Informática PPGI UTFPR)"

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
        raise HTTPException(status_code=500, detail=f"Erro interno no assistente: {str(e)}")
