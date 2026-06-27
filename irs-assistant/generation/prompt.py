"""
Prompt builder for the IRS Assistant generation layer.

Formats retrieved chunks into numbered excerpts and fills the system prompt
template. The system prompt enforces Portuguese-only answers, source citation,
and a mandatory professional-advice disclaimer.
"""

from retrieval.vector_search import SearchResult

SECTION_FALLBACK: str = "Secção geral"

SYSTEM_PROMPT: str = """És um assistente especializado em legislação fiscal portuguesa, especificamente sobre IRS.

REGRAS OBRIGATÓRIAS:
1. Responde SEMPRE em português europeu formal.
2. Usa APENAS a informação dos excertos fornecidos. Nunca uses conhecimento externo sobre leis fiscais.
3. Quando vários excertos abordam o mesmo tema, usa o mais específico e direto para a pergunta. Intervalos de datas (ex: "de abril a junho") são respostas válidas para perguntas sobre prazos.
4. Se a informação necessária não estiver nos excertos, diz explicitamente: "Não encontrei informação suficiente nos documentos disponíveis para responder a esta questão com segurança."
5. Cita SEMPRE a fonte de cada afirmação, indicando o documento e o artigo/secção entre parênteses. Exemplo: (Código do IRS, Artigo 3.º).
6. Nunca calcules valores específicos de impostos para situações pessoais.
7. Termina SEMPRE as tuas respostas com: " Esta informação é de carácter geral e não substitui aconselhamento fiscal profissional. Para a tua situação específica, consulta um contabilista certificado ou o serviço de apoio ao contribuinte da AT (www.portaldasfinancas.gov.pt)."

EXCERTOS DOS DOCUMENTOS OFICIAIS:
{chunks}

PERGUNTA DO UTILIZADOR:
{question}"""


def format_chunks(chunks: list[SearchResult]) -> str:
    """Format a list of retrieved chunks as numbered excerpts for the prompt.

    Each chunk is formatted as:
        [{i}] {source_doc} — {article or "Secção geral"}
        {content}
        ---

    Args:
        chunks: Retrieved and ranked SearchResult objects.

    Returns:
        Multi-line string of numbered excerpts joined by newlines.
    """
    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        label = chunk.article if chunk.article is not None else SECTION_FALLBACK
        lines.append(f"[{i}] {chunk.source_doc} — {label}\n{chunk.content}\n---")
    return "\n".join(lines)


def build_prompt(question: str, chunks: list[SearchResult]) -> str:
    """Build the full LLM prompt from a user question and retrieved chunks.

    Args:
        question: The user's raw question string.
        chunks: Retrieved SearchResult objects to use as grounding context.

    Returns:
        Filled SYSTEM_PROMPT string ready to send to OllamaClient.generate().
    """
    return SYSTEM_PROMPT.format(chunks=format_chunks(chunks), question=question)
