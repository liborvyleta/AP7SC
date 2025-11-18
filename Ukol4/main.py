# -------------------------------------------------------------
# 🧠 Implementace PageRank algoritmu podle Google formulace
# Autor: Libor Výleta
# Zadání: Implementovat PageRank a otestovat na testovacích i reálných datech
# -------------------------------------------------------------

import argparse
import time
from collections import defaultdict, deque
from urllib.parse import urljoin, urlparse, urlunparse
from urllib import robotparser

import numpy as np
import requests
from bs4 import BeautifulSoup


# -------------------------------------------------------------
# Utility: canonicalizace URL a robots.txt parser cache
# -------------------------------------------------------------

def canonicalize_url(url, remove_query=True):
    """Canonicalizuje URL pro lepší deduplikaci: odstraní fragment, volitelně query,
    lower-case scheme+netloc, odstraní standardní porty, odstraní koncové '/'."""
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # odstranit standardní porty
    if netloc.endswith(':80') and scheme == 'http':
        netloc = netloc[:-3]
    if netloc.endswith(':443') and scheme == 'https':
        netloc = netloc[:-4]

    path = parsed.path or '/'
    # odstraníme duplicitní lomítka na začátku cesty
    # (nemění význam, jen konzistence)
    while '//' in path:
        path = path.replace('//', '/')

    query = '' if remove_query else parsed.query
    fragment = ''

    # upravíme path tak, aby root zůstal '/'
    path_norm = path.rstrip('/')
    if path_norm == '':
        path_norm = '/'
    canon = urlunparse((scheme, netloc, path_norm, '', query, fragment))
    return canon


_robot_parsers = {}


def get_robot_parser_for(url, session=None):
    """Vrátí RobotFileParser pro danou doménu (cache)."""
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    if base in _robot_parsers:
        return _robot_parsers[base]

    rp = robotparser.RobotFileParser()
    robots_url = base + '/robots.txt'
    try:
        # použijeme requests, aby se respektovala přesměrování a timeout
        s = session or requests.Session()
        r = s.get(robots_url, timeout=5)
        if r.status_code == 200:
            rp.parse(r.text.splitlines())
        else:
            # pokud robots.txt není dostupný, považujeme to za povolené
            rp = None
    except Exception:
        rp = None

    _robot_parsers[base] = rp
    return rp


# -------------------------------------------------------------
# Vylepšená funkce get_links
# -------------------------------------------------------------

def get_links(url, base_netloc,
              session=None,
              include_subdomains=True,
              allow_query=False,
              exclude_exts=None,
              max_links=0,
              timeout=5,
              user_agent='PageRankBot/1.0'):
    """
    Získá odkazy ze stránky `url`, které patří do domény `base_netloc`.
    Vrátí množinu canonicalizovaných URL.

    - include_subdomains: pokud True, povolí subdomény (např. www., sport.)
    - allow_query: pokud False, odstraní query string
    - exclude_exts: tuple přípon, které budou ignorovány
    - max_links: 0 = neomezeně, jinak max počet odkazů vrácených z této stránky
    - timeout: HTTP timeout
    """
    session = session or requests.Session()
    headers = {'User-Agent': user_agent}

    if exclude_exts is None:
        exclude_exts = ('.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.zip', '.rar', '.exe', '.mp4', '.mp3')

    # robots
    rp = get_robot_parser_for(url, session=session)
    if rp is not None:
        try:
            if not rp.can_fetch(user_agent, url):
                # nepovolené podle robots.txt
                # vrátíme prázdné množiny
                return set()
        except Exception:
            # pokud RP selže, pokračujeme (přísnost není kritická)
            pass

    try:
        resp = session.get(url, timeout=timeout, headers=headers)
        resp.raise_for_status()
    except Exception as e:
        # tiskneme stručně chybu a vracíme prázdné
        print(f"  ⚠️  Chyba při stahování {url}: {e}")
        return set()

    soup = BeautifulSoup(resp.text, 'html.parser')
    links = set()

    for a in soup.find_all('a', href=True):
        href_raw = a['href'].strip()
        # ignoruj ne-url schémata
        if href_raw.startswith(('mailto:', 'javascript:', 'tel:', 'sms:')):
            continue

        joined = urljoin(url, href_raw)
        parsed = urlparse(joined)
        if parsed.scheme not in ('http', 'https'):
            continue

        # canonicalizace
        canonical = canonicalize_url(joined, remove_query=not allow_query)
        p = urlparse(canonical)

        # filtrování přípon
        path = p.path.lower()
        if any(path.endswith(ext) for ext in exclude_exts):
            continue

        # doména/subdoména filtr
        if include_subdomains:
            if base_netloc not in p.netloc:
                continue
        else:
            if p.netloc != base_netloc:
                continue

        links.add(canonical)
        if max_links and len(links) >= max_links:
            break

    return links


# -------------------------------------------------------------
# Crawlovací funkce: BFS do zadané hloubky (2 dle zadání)
# -------------------------------------------------------------

def crawl(start_url, depth=2, max_pages=500, max_links_per_page=0,
          include_subdomains=True, allow_query=False, user_agent='PageRankBot/1.0'):
    """
    Provede crawling od start_url do zadané hloubky (inkl. start_url jako level 0).
    Vrátí list tuple (src, dst) bez duplicit.
    - max_pages: maximální počet navštívených stránek (ochrana proti runaway)
    - max_links_per_page: 0 = neomezeně
    """
    session = requests.Session()
    start_canon = canonicalize_url(start_url, remove_query=not allow_query)
    base_netloc = urlparse(start_canon).netloc

    visited = set()
    dataset = set()

    queue = deque([(start_canon, 0)])

    while queue and len(visited) < max_pages:
        url, lvl = queue.popleft()
        if url in visited:
            continue
        if lvl > depth:
            continue

        # zkontrolovat robots, get_links také kontroluje
        links = get_links(url, base_netloc, session=session,
                          include_subdomains=include_subdomains,
                          allow_query=allow_query,
                          exclude_exts=None,
                          max_links=max_links_per_page,
                          timeout=5,
                          user_agent=user_agent)

        visited.add(url)

        for dst in links:
            # vynechat self-linky (url -> url)
            if dst == url:
                continue
            dataset.add((url, dst))
            # pokud ještě nevyužili budget a lvl < depth, přidej do fronty
            if dst not in visited and lvl + 1 <= depth:
                queue.append((dst, lvl + 1))

        # ohleduplné zpomalení
        time.sleep(0.15)

    # vracíme jako list pro kompatibilitu s pagerank
    return list(dataset)


# -------------------------------------------------------------
#  PageRank - přesně podle zadání (matice A explicitně složená)
# -------------------------------------------------------------

def pagerank(links, beta=0.85, iterations=50):
    """
    r(0) = 1/N
    r(t+1) = A . r(t)
    A = beta*M + (1-beta)*(1/N)*E
    """
    pages = sorted(set([src for src, dst in links] + [dst for src, dst in links]))
    N = len(pages)
    if N == 0:
        return {}

    index = {p: i for i, p in enumerate(pages)}

    # Build adjacency from unique outgoing neighbors to avoid duplicates and self-loops
    neighbors = defaultdict(set)
    for src, dst in links:
        if src in index and dst in index and src != dst:
            neighbors[src].add(dst)

    M = np.zeros((N, N))
    # out_degree = number of unique outgoing neighbors
    for src, dsts in neighbors.items():
        j = index[src]
        k = len(dsts)
        if k == 0:
            continue
        prob = 1.0 / k
        for dst in dsts:
            i = index[dst]
            M[i, j] = prob

    # dangling nodes
    for j in range(N):
        if np.sum(M[:, j]) == 0:
            M[:, j] = 1.0 / N

    E = np.ones((N, N))
    A = beta * M + (1 - beta) * (1 / N) * E

    r = np.ones(N) / N
    for _ in range(iterations):
        r = A @ r

    return {pages[i]: float(r[i]) for i in range(N)}


# -------------------------------------------------------------
# Testovací data a hlavní běh
# -------------------------------------------------------------

def test_pagerank_print():
    test_links = [
        (1, 2), (1, 3),
        (2, 4),
        (3, 1), (3, 2), (3, 4),
        (4, 3)
    ]
    print('\n' + '=' * 60)
    print('🔹 TESTOVACÍ DATA – ukázka r(0) → r(50)')
    print('=' * 60)

    pages = sorted(set([src for src, dst in test_links] + [dst for src, dst in test_links]))
    N = len(pages)

    # sestavíme M a A jako v zadání
    index = {p: i for i, p in enumerate(pages)}
    M = np.zeros((N, N))
    out_degree = defaultdict(int)
    for src, dst in test_links:
        out_degree[src] += 1
    for src, dst in test_links:
        M[index[dst], index[src]] = 1.0 / out_degree[src]
    for j in range(N):
        if np.sum(M[:, j]) == 0:
            M[:, j] = 1.0 / N
    beta = 0.85
    E = np.ones((N, N))
    A = beta * M + (1 - beta) * (1 / N) * E

    # r(0)
    r0 = np.ones(N) / N
    print(f'r(0): {np.round(r0, 8)}')

    # provést 50 iterací (r(50)) a ověřit součet
    iterations = 50
    r = r0.copy()
    for t in range(1, iterations + 1):
        r = A @ r

    print(f'r({iterations}): {np.round(r, 8)}')

    print(f'\n📊 Výsledné ranky (po {iterations}. iteraci):')
    for page_num, score in sorted(zip(pages, r), key=lambda x: -x[1]):
        print(f'  Stránka {page_num}: {score:.6f}')

    total = float(np.sum(r))
    print(f'\nSuma PageRanků: {total:.12f}')
    # ověření, že suma je přibližně 1 (malé numerické odchylky povoleny)
    assert abs(total - 1.0) < 1e-9, f'Suma PageRanků není 1 (hodnota: {total})'
    print('✅ Test: suma PageRank hodnot je ≈ 1 (ok)')


def main():
    parser = argparse.ArgumentParser(description='Crawler + PageRank podle zadání')
    parser.add_argument('--start', default='https://ailab.fai.utb.cz/', help='Startovací URL (default: ailab)')
    parser.add_argument('--depth', type=int, default=2, help='Hloubka crawl (default 2)')
    parser.add_argument('--max_pages', type=int, default=300, help='Max počet navštívených stránek')
    parser.add_argument('--max_links_per_page', type=int, default=0, help='Max odkazů z jedné stránky (0 = neomezeně)')
    parser.add_argument('--iterations', type=int, default=50, help='Počet iterací PageRank (default 50)')
    args = parser.parse_args()

    # 1) testovací příklad
    test_pagerank_print()

    # 2) crawling reálného webu
    print('\n' + '=' * 60)
    print('🌐 STAHOVÁNÍ ODKAZŮ Z WEBU (vytvoření datasetu)')
    print('=' * 60)
    print(f'Start URL: {args.start}  |  Hloubka: {args.depth}  |  Max pages: {args.max_pages}')

    dataset = crawl(args.start, depth=args.depth, max_pages=args.max_pages,
                    max_links_per_page=args.max_links_per_page,
                    include_subdomains=True, allow_query=False)

    # odstranění duplicit (už máme jako set v crawl, ale přebíráme list)
    dataset = list(set(dataset))

    # Uložit dataset do CSV pro další analýzu
    try:
        import csv
        with open('dataset.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['source', 'target'])
            for s, d in dataset:
                writer.writerow([s, d])
        print("\n💾 Dataset uložen do: dataset.csv")
    except Exception as e:
        print(f"⚠️ Chyba při ukládání datasetu: {e}")

    print(f'\n✅ VSTUPNÍ DATA — počet odkazů (párů src->dst): {len(dataset)}')
    print('\n📋 Ukázka prvních 20 záznamů:')
    for i, (s, d) in enumerate(dataset[:20], 1):
        print(f'  {i:3d}. {s} -> {d}')

    # 3) PageRank
    print('\n' + '=' * 60)
    print('🏁 VÝPOČET PAGERANKU')
    print('=' * 60)

    ranking = pagerank(dataset, beta=0.85, iterations=args.iterations)

    # Uložit výsledky PageRank do CSV (vždy vytvoříme soubor s hlavičkou, i když je prázdný)
    try:
        import csv
        with open('pagerank.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['url', 'pagerank'])
            if ranking:
                sorted_rank = sorted(ranking.items(), key=lambda x: -x[1])
                for url, score in sorted_rank:
                    writer.writerow([url, score])
        print("\n💾 PageRank výsledky uloženy do: pagerank.csv")
    except Exception as e:
        print(f"⚠️ Chyba při ukládání PageRank výsledků: {e}")

    # Pokud není žádný výsledek, ošetříme výstup a vyhneme se numpy warnings
    if not ranking:
        print('\n⚠️  Nebyly nalezeny žádné stránky pro výpočet PageRanku. Plné výstupy jsou přeskočeny.')
        print('\n📊 Statistiky:')
        print(f'   Celkový počet stránek: 0')
        print(f'   Suma PageRanků: 0.000000')
        print(f'   Průměrný PageRank: 0.000000')
        return

    sorted_rank = sorted(ranking.items(), key=lambda x: -x[1])

    # Vykreslení grafu top 20 (pokud matplotlib je dostupný)
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        top_n = 20
        top = sorted_rank[:top_n]
        if top:
            labels = [u for u, s in top]
            scores = [s for u, s in top]
            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(scores)), scores, color='C0')
            plt.xticks(range(len(labels)), labels, rotation=75, ha='right')
            plt.ylabel('PageRank')
            plt.title(f'Top {min(top_n, len(labels))} PageRank')
            plt.tight_layout()
            out_png = 'pagerank_top20.png'
            plt.savefig(out_png, dpi=150)
            plt.close()
            print(f"💾 Graf top {min(top_n, len(labels))} uložen do: {out_png}")
    except Exception as e:
        # pokud matplotlib není nainstalovaný, instrukce pro uživatele
        if isinstance(e, ModuleNotFoundError):
            print("\n⚠️ modul 'matplotlib' není nainstalovaný. Pro vytvoření grafu nainstalujte ho: pip install matplotlib")
        else:
            print(f"\n⚠️ Chyba při vytváření grafu: {e}")

    print('\n🏆 Top 20 nejdůležitějších stránek:')
    for i, (url, score) in enumerate(sorted_rank[:20], 1):
        print(f'  {i:2d}. {score:.6f} - {url}')

    print('\n📊 Statistiky:')
    print(f'   Celkový počet stránek: {len(ranking)}')
    print(f'   Suma PageRanků: {sum(ranking.values()):.6f}')
    print(f'   Průměrný PageRank: {np.mean(list(ranking.values())):.6f}')


if __name__ == '__main__':
    main()
