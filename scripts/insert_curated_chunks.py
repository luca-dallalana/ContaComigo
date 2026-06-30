"""
Insert hand-curated chunks for articles that parsed poorly from the source PDFs.

These cover:
  - CIRS Artigo 68: tax bracket table (split across pages, middle rows dropped by pypdf)
  - CIRS Artigo 69: quociente familiar (vector search misses it due to semantic overlap with Artigo 16)
  - CIRS Artigo 70: mínimo de existência (BM25 finds it but vector drowns it out)
  - CIRS Artigo 99: retenção na fonte mechanism (tables live in a separate Portaria not ingested)
  - EBF Artigo 21: PPR / planos de poupança-reforma (acronym "PPR" missing from parsed chunk)

Run this after ingestion. Idempotent — deletes any previous curated entries before inserting.

Usage:
    python scripts/insert_curated_chunks.py
"""

import os
import re
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.vector_search import embed_query

_TOKEN_SPLIT_RE: re.Pattern = re.compile(r"\W+")

load_dotenv()

CURATED_MARKER_START = 9900

CURATED_CHUNKS = [
    {
        "content": (
            "Artigo 68.º — Taxas gerais do IRS\n"
            "As taxas do IRS aplicáveis ao rendimento coletável são as seguintes (redação da Lei n.º 73-A/2025, de 30 de dezembro):\n\n"
            "Rendimento coletável (euros) | Taxa Normal (A) | Taxa Média (B)\n"
            "Até 8.342 | 12,50% | 12,500%\n"
            "De mais de 8.342 até 12.587 | 15,70% | 13,579%\n"
            "De mais de 12.587 até 17.838 | 21,20% | 15,823%\n"
            "De mais de 17.838 até 23.089 | 24,10% | 17,705%\n"
            "De mais de 23.089 até 29.397 | 31,10% | 20,579%\n"
            "De mais de 29.397 até 43.090 | 34,90% | 25,130%\n"
            "De mais de 43.090 até 46.566 | 43,10% | 26,472%\n"
            "De mais de 46.566 até 86.634 | 44,60% | 34,856%\n"
            "Superior a 86.634 | 48,00% | —\n\n"
            "O primeiro escalão tem taxa normal de 12,50% e aplica-se a rendimentos coletáveis até 8.342 €. "
            "Quando o rendimento coletável é superior a 8.342 €, divide-se em duas partes: "
            "uma igual ao limite do maior escalão que nele couber (à qual se aplica a taxa média B), "
            "outra igual ao excedente (à qual se aplica a taxa normal A do escalão imediatamente superior)."
        ),
        "source_doc": "Código do IRS (CIRS)",
        "article": "Artigo 68.",
        "fiscal_year": 2025,
        "chunk_index": CURATED_MARKER_START,
    },
    {
        "content": (
            "Artigo 69.º — Quociente familiar\n"
            "O quociente familiar é o mecanismo do IRS que reduz o imposto para casais com tributação conjunta e para famílias com filhos.\n\n"
            "1. Tributação conjunta de casados/unidos de facto: as taxas aplicáveis são as correspondentes ao rendimento coletável dividido por dois. "
            "As taxas do artigo 68.º aplicam-se ao quociente do rendimento coletável; o resultado obtido é depois multiplicado por dois para apurar a coleta final do IRS.\n\n"
            "Exemplo: casal com rendimento coletável conjunto de 40.000 €. "
            "Aplica-se a tabela de escalões a 20.000 € (metade), depois multiplica-se a coleta por 2.\n\n"
            "Este mecanismo de divisão por dois — o quociente familiar — evita que o rendimento conjunto salte para escalões mais elevados e reduz o imposto total pago pela família."
        ),
        "source_doc": "Código do IRS (CIRS)",
        "article": "Artigo 69.",
        "fiscal_year": 2025,
        "chunk_index": CURATED_MARKER_START + 1,
    },
    {
        "content": (
            "Artigo 70.º — Mínimo de existência\n"
            "O mínimo de existência garante que os contribuintes com rendimentos baixos ficam protegidos de uma tributação excessiva.\n\n"
            "O valor de referência do mínimo de existência é igual ao maior valor entre 12.880 € e 1,5 × 14 × IAS "
            "(Indexante dos Apoios Sociais) — redação da Lei n.º 73-A/2025, de 30 de dezembro.\n\n"
            "Para titulares de rendimentos predominantemente de trabalho dependente, atividades da tabela do Artigo 31.º ou pensões, "
            "é abatido ao rendimento coletável um montante pelo mínimo de existência, calculado segundo a fórmula:\n"
            "L = valor de referência − (Limite despesas gerais / Taxa 1.º escalão × 3,60) + (Limite 1.º escalão / 3,60)\n\n"
            "O abatimento não se aplica quando a soma dos rendimentos brutos de todos os titulares supera 2,2 × 14 × IAS "
            "multiplicado pelo número de sujeitos passivos."
        ),
        "source_doc": "Código do IRS (CIRS)",
        "article": "Artigo 70.",
        "fiscal_year": 2025,
        "chunk_index": CURATED_MARKER_START + 2,
    },
    {
        "content": (
            "Artigo 99.º — Retenção na fonte para trabalhadores por conta de outrem (Categoria A)\n"
            "A retenção na fonte é o mecanismo pelo qual a entidade empregadora desconta mensalmente ao trabalhador "
            "uma parte do IRS estimado, entregando-a diretamente à Autoridade Tributária.\n\n"
            "Como é calculada:\n"
            "As tabelas de retenção na fonte para rendimentos das categorias A (trabalho dependente) e H (pensões) "
            "são aprovadas por despacho do membro do Governo responsável pela área das finanças (Artigo 99.º-F do CIRS). "
            "As tabelas variam em função do:\n"
            "- Montante do salário bruto mensal\n"
            "- Estado civil (solteiro, casado único titular, casado dois titulares)\n"
            "- Número de dependentes a cargo\n"
            "- Deficiência do titular (taxa reduzida a 50%)\n\n"
            "A retenção aplica-se ao montante total do salário, incluindo subsídios, prémios e outras remunerações. "
            "No final do ano, a retenção é comparada com o imposto apurado na declaração de IRS: "
            "se a retenção for superior, há reembolso; se for inferior, há imposto a pagar.\n\n"
            "As tabelas atualizadas encontram-se publicadas no Portal das Finanças (portaldasfinancas.gov.pt)."
        ),
        "source_doc": "Código do IRS (CIRS)",
        "article": "Artigo 99.",
        "fiscal_year": 2025,
        "chunk_index": CURATED_MARKER_START + 3,
    },
    {
        "content": (
            "Artigo 21.º EBF — PPR (Planos de Poupança-Reforma) — Dedução à coleta do IRS\n"
            "Os PPR (planos de poupança-reforma), também designados como produtos individuais de reforma, "
            "são dedutíveis à coleta do IRS nas seguintes condições (Estatuto dos Benefícios Fiscais, Artigo 21.º):\n\n"
            "Dedução: 20% dos valores aplicados no ano em PPR por sujeito passivo não casado, "
            "ou por cada um dos cônjuges não separados judicialmente de pessoas e bens.\n\n"
            "Limites máximos da dedução:\n"
            "- Até 35 anos: 400 €\n"
            "- Entre 35 e 50 anos: 350 €\n"
            "- A partir dos 50 anos: 300 €\n\n"
            "A fruição do benefício fica sem efeito (com agravamento de 10% por cada ano de fruição) "
            "se os valores forem reembolsados antes dos prazos legais, exceto em casos de desemprego prolongado, "
            "incapacidade permanente ou doença grave."
        ),
        "source_doc": "Estatuto dos Benefícios Fiscais (EBF)",
        "article": "Artigo 21.",
        "fiscal_year": 2025,
        "chunk_index": CURATED_MARKER_START + 4,
    },
]


def main() -> None:
    conn = psycopg2.connect(os.environ["POSTGRES_URL"])
    cur = conn.cursor()

    cur.execute(
        "DELETE FROM bm25_corpus WHERE chunk_id IN (SELECT id FROM chunks WHERE chunk_index >= %s)",
        (CURATED_MARKER_START,),
    )
    cur.execute("DELETE FROM chunks WHERE chunk_index >= %s", (CURATED_MARKER_START,))
    deleted = cur.rowcount
    if deleted:
        print(f"Removed {deleted} previous curated chunk(s).")

    for chunk in CURATED_CHUNKS:
        embedding = embed_query(chunk["content"])
        cur.execute(
            """
            INSERT INTO chunks (content, embedding, source_doc, article, section, page_number, fiscal_year, chunk_index)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                chunk["content"],
                embedding,
                chunk["source_doc"],
                chunk["article"],
                chunk.get("section"),
                chunk.get("page_number"),
                chunk["fiscal_year"],
                chunk["chunk_index"],
            ),
        )
        chunk_id = cur.fetchone()[0]
        tokens = " ".join(t for t in _TOKEN_SPLIT_RE.split(chunk["content"].lower()) if t)
        cur.execute(
            "INSERT INTO bm25_corpus (chunk_id, tokens) VALUES (%s, %s)",
            (chunk_id, tokens),
        )
        print(f"Inserted: {chunk['source_doc']} — {chunk['article']}")

    conn.commit()
    conn.close()
    print(f"\nDone. {len(CURATED_CHUNKS)} curated chunk(s) inserted.")


if __name__ == "__main__":
    main()
