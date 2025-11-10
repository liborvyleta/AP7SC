Projekt: PageRank crawler

Co tento skript dělá
- Stáhne odkazy z jedné startovací stránky (BFS crawling do zvolené hloubky, default 2).
- Vytvoří dataset ve tvaru páry (zdroj -> cíl).
- Spočítá PageRank podle přesné Google formulace:
  r(0) = 1/N
  r(t+1) = A . r(t)
  A = beta*M + (1-beta)*(1/N)*E
  kde M je sloupcově-normalizovaná matice přechodů (pravděpodobnosti ze stránky j na i).
- Uloží dataset do `dataset.csv` a výsledky PageRank do `pagerank.csv`.

Použitý vzorec (přesně podle zadání)

r(0) = vektor délky N s hodnotami 1/N (N = celkový počet stránek)

r(t+1) = A . r(t)

A = beta*M + (1-beta)*(1/N)*E

beta = 0.85 - pravděpodobnost sledování existujících linků
N = celkový počet stránek
E = matice jedniček velikosti NxN
M = adjacenční matice (ve sloupcích jsou pravděpodobnosti přechodů ze stránky j na stránku i, tedy součet každého sloupce = 1)
r = výsledný ranking

Požadavky
- Python 3.8+
- Instalace závislostí:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Jak spustit
- Spustit s defaultními parametry (start = https://ailab.fai.utb.cz/):

```bash
python main.py
```

- Příklad se změnou startovací URL a menším crawl:

```bash
python main.py --start https://example.com --depth 1 --max_pages 50 --max_links_per_page 20 --iterations 50
```

Co vygeneruje
- `dataset.csv` — sloupce: source,target — dataset párů odkazů
- `pagerank.csv` — sloupce: url,pagerank — seřazené podle hodnocení
- Na stdout: průběžné informace, ukázka datasetu a top výsledků PageRanku

Důležité poznámky
- Skript respektuje robots.txt (pokud existuje) a ignoruje odkazy mimo doménu (volitelně lze povolit subdomény).
- Canonicalizuje URL (odstraňuje fragmenty a podle nastavení i query string) pro lepší deduplikaci.
- Maticová formulace PageRanku je přesně podle zadání v dokumentaci.

Další možné vylepšení
- paralelní asynchronní crawling (aiohttp)
- lepší canonicalizace pomocí knihovny
- přidat unit testy pro `get_links` (mock HTTP)

Kontakt
- Pokud chcete, upravím parametry crawlování nebo přidám testy / vizualizaci výsledků.
