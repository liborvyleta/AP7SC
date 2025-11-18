# PageRank crawler & analyzer

Tento repozitář obsahuje jednoduchý web crawler a implementaci PageRanku podle Google formulace (matice A = beta*M + (1-beta)*(1/N)*E). Projekt byl vytvořen pro školní úkol a slouží k: 1) stažení odkazů z webu do datasetu, 2) výpočtu PageRanku z tohoto datasetu a 3) uložení výsledků do CSV + (volitelně) vykreslení grafu.

Obsah repozitáře
- `main.py` – hlavní skript: crawling, tvorba `dataset.csv`, výpočet PageRank a uložení `pagerank.csv` (a případné vykreslení `pagerank_top20.png`).
- `dataset.csv` – (generovaný) CSV se sběrem párů `source,target`.
- `pagerank.csv` – (generovaný) CSV s URL a jejich PageRank skóry.
- `pagerank_top20.png` – (generovaný, volitelně) sloupcový graf top stránek.
- `requirements.txt` – seznam povinných knihoven (requests, beautifulsoup4, numpy).

Rychlý start (lokálně)
1) (Doporučeno) vytvořte virtuální prostředí a aktivujte ho (volitelné):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Nainstalujte závislosti:

```bash
pip install -r requirements.txt
```

3) Testovací běh (bez crawlingu) — spustí se pouze vestavěný příklad PageRank a ověří, že suma skóre = 1:

```bash
python3 main.py --max_pages 0
```

4) Spustit skutečný crawl (příklad) — stáhne odkazy z `https://ailab.fai.utb.cz/` do hloubky 2 a maximálně 200 stránek:

```bash
python3 main.py --start https://ailab.fai.utb.cz/ --depth 2 --max_pages 200 --iterations 50
```

Po dokončení najdete:
- `dataset.csv` – pár source -> target (hlavička: `source,target`)
- `pagerank.csv` – seřazené URL s jejich PageRank hodnotami (hlavička: `url,pagerank`)
- `pagerank_top20.png` – pokud je instalovaný `matplotlib`, uloží se graf top 20

TIP: zobrazit prvních pár řádků v terminálu

```bash
head -n 20 dataset.csv
head -n 40 pagerank.csv
```

Volitelné (graf):
Pokud chcete uložit graf `pagerank_top20.png`, nainstalujte matplotlib:

```bash
pip install matplotlib
```

a spusťte skript znovu (viz příklad výše).

Poznámky k implementaci PageRanku
- Implementace používá přesnou maticovou formulaci:
  - M: matice přechodů tak, že každý sloupec je rozdělen rovnoměrně mezi unikátní outgoing odkazy z dané stránky.
  - Dangling nodes (stránky bez odchozích odkazů) jsou ošetřeny tak, že jejich sloupce mají hodnotu 1/N.
  - A = beta*M + (1-beta)*(1/N)*E s beta=0.85 (default).
  - Počet iterací je defaultně 50 (parametr `--iterations`).

- Skript obsahuje testovací funkci `test_pagerank_print()` která počítá r(50) pro jednoduchý příklad a ověřuje, že suma PageRank hodnot ≈ 1.

Bezpečnost / etiketa
- Crawler respektuje `robots.txt` a uživatelský agent `PageRankBot/1.0`.
- Při lokálním testování používejte rozumné limity (`--max_pages`, `--max_links_per_page`) aby nedošlo k přetížení cílových serverů.

Doporučené vylepšení (možnosti, které lze přidat)
- Zastavení dle konvergence: místo pevného počtu iterací ukončit iterace, když ||r(t+1)-r(t)||_1 < tol.
- Volitelný `--plot` flag pro vykreslení (pokud je matplotlib k dispozici).
- Export top-N do JSON nebo další metriky (in-degree, out-degree) do CSV.
- Jednotkové testy (pytest) pro PageRank funkci.

Chcete-li, mohu provést některé z těchto vylepšení (např. přidat konvergenci a přidat jednoduchý unit-test). Napište, kterou možnost chcete, a já to rovnou upravím.

Kontakt / autor
- Autor: Libor Výleta

----
Soubor vytvořen automaticky jako README pro projekt PageRank crawler. Pokud chcete jiný styl (angličtina, stručnější nebo rozšířený), dejte vědět a upravím ho.
