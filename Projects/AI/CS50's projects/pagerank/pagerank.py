import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    if len(corpus[page]) == 0:
        return {key: 1 / len(corpus) for key in corpus}
    else:
        tmp = {}
        for key in corpus:
            if key in corpus[page]:
                tmp[key] = (1 - damping_factor) / len(corpus) + damping_factor / len(corpus[page])
            else:
                tmp[key] = (1 - damping_factor) / len(corpus)
        return tmp


def sample_pagerank(corpus, damping_factor, n):
    cur = random.choice(list(corpus.keys()))
    cnt = {}
    for page in corpus:
        cnt[page] = 0
    cnt[cur] = 1
    for i in range(0, n - 1):
        distribution = transition_model(corpus, cur, damping_factor)
        cur = random.choices(list(distribution), weights=distribution.values(), k = 1)[0]
        cnt[cur] = cnt.get(cur, 0) + 1
    return {k: v / n for k, v in cnt.items()}


def iterate_pagerank(corpus, damping_factor):
    actual_rank = {key: 1 / len(corpus) for key in corpus}

    while True: 
        new_rank = {}

        for page in corpus:
            new_rank[page] = new_rank.get(page, 0) + (1 - damping_factor) / len(corpus)
            if corpus[page] == set():
                for page2 in corpus:
                    new_rank[page2] = new_rank.get(page2, 0) + damping_factor * actual_rank[page] / len(corpus)
            else:
                for child in corpus[page]:
                    new_rank[child] = new_rank.get(child, 0) + damping_factor * actual_rank[page] / len(corpus[page])
        
        if all(abs(new_rank[page] - actual_rank[page]) <= 0.001 for page in corpus):
            actual_rank = new_rank
            break
        
        actual_rank = new_rank
    
    sum_rank = 0
    
    for page in new_rank:
        sum_rank += new_rank[page]
    
    mul = 1 / sum_rank

    for page in new_rank:
        new_rank[page] *= mul

    return actual_rank
    
if __name__ == "__main__":
    main()
