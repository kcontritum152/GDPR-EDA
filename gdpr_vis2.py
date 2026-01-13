"""
GDPR Article Embedding and Similarity Analysis - Fixed Ground Truth
Focus: Clean scraping → Quality visualizations
"""

import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import umap
import re
import json
from typing import Dict, List, Tuple, Set
import time
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

class GDPRSimilarityAnalyzer:
    def __init__(self, embedding_model='sentence-transformers'):
        """
        Initialize the analyzer with an embedding model.
        
        Args:
            embedding_model: 'sentence-transformers', 'openai', or 'huggingface'
        """
        self.articles = {}
        self.embeddings = {}
        self.model_type = embedding_model
        
        if embedding_model == 'sentence-transformers':
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        elif embedding_model == 'openai':
            from openai import OpenAI
            self.client = OpenAI()
            self.model = 'text-embedding-3-small'

    def log(self, message, also_print=True):
        """
        Write message to log file and optionally print to console
        """
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
        if also_print:
            print(message)
    
    def scrape_gdpr_article(self, article_num: int) -> Dict:
        """
        Scrape a single GDPR article with IMPROVED ground truth extraction.
        Only extracts genuine inline references, not navigation links.
        
        Returns:
            Dict with article text, title, and related articles
        """
        url = f"https://gdpr-info.eu/art-{article_num}-gdpr/"
        
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract article title
            title_elem = soup.find('h1')
            title = title_elem.text.strip() if title_elem else f"Article {article_num}"
            
            # Extract article text from main content area
            article_text = []
            content = soup.find('div', class_='entry-content') or soup.find('article')
            
            if content:
                # Get text but stop before "Suitable Recitals" section
                for elem in content.find_all(['p', 'li']):
                    text = elem.get_text(strip=True)
                    if text and not any(skip in text for skip in ['Suitable Recitals', 'Table of contents', 'Quick Access']):
                        article_text.append(text)
            
            full_text = ' '.join(article_text)
            
            # IMPROVED: Extract related articles ONLY from within article content
            related_articles = self._extract_inline_references(content, article_num)
            
            # Extract related recitals from the "Suitable Recitals" section
            related_recitals = self._extract_related_recitals(soup)
            
            return {
                'number': article_num,
                'title': title,
                'text': full_text,
                'related_articles': sorted(list(related_articles)),
                'related_recitals': sorted(list(related_recitals)),
                'url': url
            }
            
        except Exception as e:
            print(f"Error scraping article {article_num}: {e}")
            return None
    
    def _extract_inline_references(self, content, article_num: int) -> Set[int]:
        """
        Extract ONLY genuine inline article references from within the article text.
        Filters out navigation/menu links.
        """
        related_articles = set()
        
        if not content:
            return related_articles
        
        # Strategy 1: Find links that are within paragraphs or list items (not headers/nav)
        for link in content.find_all('a', href=re.compile(r'/art-\d+-gdpr/')):
            # Check if link is inside actual content (p, li, blockquote, etc.)
            parent_tags = [p.name for p in link.find_parents()]
            
            # Only accept links inside content elements, not nav/header/footer
            if any(tag in parent_tags for tag in ['p', 'li', 'blockquote', 'td']):
                match = re.search(r'/art-(\d+)-gdpr/', link['href'])
                if match:
                    related_num = int(match.group(1))
                    if related_num != article_num:  # Don't include self-reference
                        related_articles.add(related_num)
        
        # Strategy 2: Also look for text patterns like "Article 6(1)" in the text
        text_content = content.get_text()
        # Match patterns like "Article 6", "Article 6(1)", "Art. 6", "Art 6"
        article_mentions = re.findall(r'(?:Article|Art\.?)\s+(\d+)', text_content)
        for num_str in article_mentions:
            num = int(num_str)
            if num != article_num and 1 <= num <= 99:
                related_articles.add(num)
        
        return related_articles
    
    def _extract_related_recitals(self, soup) -> Set[int]:
        """
        Extract related recitals from the "Suitable Recitals" section.
        """
        related_recitals = set()
        
        # Find the "Suitable Recitals" section
        recitals_header = soup.find('h2', string=re.compile(r'Suitable Recitals|Related Recitals', re.IGNORECASE))
        
        if recitals_header:
            # Get the list that follows this header
            recital_list = recitals_header.find_next(['ul', 'ol'])
            if recital_list:
                for link in recital_list.find_all('a', href=re.compile(r'/recitals/no-\d+/')):
                    match = re.search(r'/recitals/no-(\d+)/', link['href'])
                    if match:
                        related_recitals.add(int(match.group(1)))
        
        return related_recitals
    
    def scrape_all_articles(self, article_range=range(1, 100)):
        """
        Scrape multiple GDPR articles with progress reporting
        """
        print("="*80)
        print("SCRAPING GDPR ARTICLES WITH IMPROVED GROUND TRUTH EXTRACTION")
        print("="*80)
        
        for article_num in article_range:
            article_data = self.scrape_gdpr_article(article_num)
            if article_data and article_data['text']:
                self.articles[article_num] = article_data
                ref_count = len(article_data['related_articles'])
                print(f"✓ Article {article_num:2d}: {article_data['title'][:45]:45s} | {ref_count:2d} refs")
                time.sleep(0.5)  # Be respectful to the server
            else:
                print(f"✗ Article {article_num:2d}: Failed or empty")
        
        # Print summary statistics
        total_refs = sum(len(art['related_articles']) for art in self.articles.values())
        avg_refs = total_refs / len(self.articles) if self.articles else 0
        
        print(f"\n{'='*80}")
        print(f"SCRAPING COMPLETE")
        print(f"{'='*80}")
        print(f"Articles scraped: {len(self.articles)}")
        print(f"Total references: {total_refs}")
        print(f"Average references per article: {avg_refs:.1f}")
        print(f"\nExpected range: 3-15 refs per article (higher = might include nav links)")
    
    def inspect_ground_truth(self, sample_articles=None):
        """
        Inspect the quality of scraped ground truth for specific articles
        """
        if sample_articles is None:
            sample_articles = [6, 17, 15]  # Common articles to check
        
        print(f"\n{'='*80}")
        print("GROUND TRUTH QUALITY CHECK")
        print(f"{'='*80}")
        
        for art_num in sample_articles:
            if art_num in self.articles:
                art = self.articles[art_num]
                print(f"\nArticle {art_num}: {art['title']}")
                print(f"  Related articles ({len(art['related_articles'])}): {art['related_articles']}")
                print(f"  Related recitals ({len(art['related_recitals'])}): {art['related_recitals'][:10]}...")
        
        print(f"\n{'='*80}")
        print("If you see 20+ related articles for most articles, ground truth is still contaminated!")
        print("Good quality: 3-15 related articles per article")
        print(f"{'='*80}\n")
    
    def generate_embeddings(self):
        """
        Generate embeddings for all scraped articles
        """
        print("\n" + "="*80)
        print("GENERATING EMBEDDINGS")
        print("="*80)
        
        for article_num, article_data in self.articles.items():
            text = article_data['text']
            
            if self.model_type == 'sentence-transformers':
                embedding = self.model.encode(text, convert_to_numpy=True)
            elif self.model_type == 'openai':
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model
                )
                embedding = np.array(response.data[0].embedding)
            
            self.embeddings[article_num] = embedding
            print(f"✓ Article {article_num:2d} embedded (dim: {len(embedding)})")
        
        print(f"\nGenerated {len(self.embeddings)} embeddings")
    
    def calculate_similarities(self) -> Dict[int, List[Tuple[int, float]]]:
        """
        Calculate cosine similarity between all article pairs
        
        Returns:
            Dict mapping article_num to list of (related_article_num, similarity_score)
        """
        print("\n" + "="*80)
        print("CALCULATING COSINE SIMILARITIES")
        print("="*80)
        
        article_nums = sorted(self.embeddings.keys())
        embedding_matrix = np.array([self.embeddings[num] for num in article_nums])
        
        # Calculate pairwise cosine similarities
        similarity_matrix = cosine_similarity(embedding_matrix)
        
        similarities = {}
        for i, article_num in enumerate(article_nums):
            # Get similarities for this article (excluding self)
            article_sims = []
            for j, other_num in enumerate(article_nums):
                if i != j:
                    article_sims.append((other_num, similarity_matrix[i][j]))
            
            # Sort by similarity (highest first)
            article_sims.sort(key=lambda x: x[1], reverse=True)
            similarities[article_num] = article_sims
        
        print(f"Calculated similarities for {len(article_nums)} articles")
        return similarities
    
    def evaluate_against_ground_truth(self, similarities: Dict, top_k: int = 10):
        """
        Evaluate similarity results against gdpr-info.eu's related articles
        
        Args:
            similarities: Output from calculate_similarities()
            top_k: How many top similar articles to consider
        """
        print(f"\n{'='*80}")
        print(f"VALIDATION: Comparing top-{top_k} similar vs. ground truth references")
        print(f"{'='*80}\n")
        
        total_related = 0
        total_found = 0
        results = []
        
        for article_num in sorted(similarities.keys()):
            if article_num not in self.articles:
                continue
            
            ground_truth = set(self.articles[article_num]['related_articles'])
            if not ground_truth:
                continue
            
            # Get top-k most similar articles
            top_similar = [art_num for art_num, _ in similarities[article_num][:top_k]]
            found_articles = ground_truth.intersection(set(top_similar))
            
            precision = len(found_articles) / len(top_similar) if top_similar else 0
            recall = len(found_articles) / len(ground_truth) if ground_truth else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            total_related += len(ground_truth)
            total_found += len(found_articles)
            
            result = {
                'article': article_num,
                'title': self.articles[article_num]['title'],
                'ground_truth': sorted(list(ground_truth)),
                'found': sorted(list(found_articles)),
                'missed': sorted(list(ground_truth - found_articles)),
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            results.append(result)
            
            # Only print if article has ground truth
            if ground_truth:
                print(f"Article {article_num}: {self.articles[article_num]['title'][:50]}")
                print(f"  Ground truth: {sorted(list(ground_truth))}")
                print(f"  Found: {sorted(list(found_articles))} {'✓' if found_articles else '✗'}")
                if ground_truth - found_articles:
                    print(f"  Missed: {sorted(list(ground_truth - found_articles))}")
                print(f"  Precision: {precision:.1%} | Recall: {recall:.1%} | F1: {f1:.1%}")
                print()
        
        # Overall statistics
        overall_recall = total_found / total_related if total_related > 0 else 0
        avg_f1 = np.mean([r['f1'] for r in results if r['ground_truth']]) if results else 0
        
        print(f"{'='*80}")
        print(f"OVERALL RESULTS")
        print(f"{'='*80}")
        print(f"Total ground truth relationships: {total_related}")
        print(f"Total found in top-{top_k}: {total_found}")
        print(f"Overall recall: {overall_recall:.1%}")
        print(f"Average F1 score: {avg_f1:.1%}")
        print(f"\nInterpretation:")
        if avg_f1 >= 0.5:
            print("  ✓ GOOD - Model is finding meaningful relationships")
        elif avg_f1 >= 0.3:
            print("  ⚠ FAIR - Model captures some relationships but could be better")
        else:
            print("  ✗ POOR - Model struggles to match expert relationships")
        
        return results
    
    def show_top_similarities(self, article_num: int, similarities: Dict, top_n: int = 10):
        """
        Display top N most similar articles for a given article
        """
        if article_num not in similarities:
            print(f"Article {article_num} not found")
            return
        
        print(f"\n{'='*80}")
        print(f"Top {top_n} most similar articles to Article {article_num}")
        print(f"Title: {self.articles[article_num]['title']}")
        print(f"{'='*80}")
        
        ground_truth = set(self.articles[article_num]['related_articles'])
        
        for i, (other_num, score) in enumerate(similarities[article_num][:top_n], 1):
            is_ground_truth = other_num in ground_truth
            marker = "✓" if is_ground_truth else " "
            print(f"{i:2d}. {marker} Article {other_num:2d} (similarity: {score:.4f})")
            print(f"     {self.articles[other_num]['title']}")
        
        print(f"\n✓ = Confirmed by gdpr-info.eu ground truth")
    
    def visualize_similarity_heatmap(self, figsize=(14, 12), save=True):
        """
        Create a heatmap of cosine similarities between articles
        """
        article_nums = sorted(self.embeddings.keys())
        embedding_matrix = np.array([self.embeddings[num] for num in article_nums])
        similarity_matrix = cosine_similarity(embedding_matrix)
        
        plt.figure(figsize=figsize)
        
        # Create mask for diagonal (similarity with self)
        mask = np.zeros_like(similarity_matrix, dtype=bool)
        np.fill_diagonal(mask, True)
        
        sns.heatmap(similarity_matrix, 
                    mask=mask,
                    xticklabels=article_nums, 
                    yticklabels=article_nums,
                    cmap='RdYlBu_r',
                    center=0.5,
                    vmin=0.3, vmax=0.9,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={'label': 'Cosine Similarity', 'shrink': 0.8})
        
        plt.title('GDPR Article Similarity Heatmap\n(Darker = More Similar)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Article Number', fontsize=12)
        plt.ylabel('Article Number', fontsize=12)
        plt.tight_layout()
        
        if save:
            plt.savefig('gdpr_similarity_heatmap.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: gdpr_similarity_heatmap.png")
        plt.show()
    
    def visualize_network_graph(self, threshold=0.65, figsize=(18, 14), save=True):
        """
        Create a network graph showing highly similar articles
        
        Args:
            threshold: Only show edges with similarity >= threshold (default 0.65)
        """
        article_nums = sorted(self.embeddings.keys())
        embedding_matrix = np.array([self.embeddings[num] for num in article_nums])
        similarity_matrix = cosine_similarity(embedding_matrix)
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for num in article_nums:
            G.add_node(num, title=self.articles[num]['title'])
        
        # Add edges for similar articles
        edge_count = 0
        for i, art_i in enumerate(article_nums):
            for j, art_j in enumerate(article_nums):
                if i < j:  # Avoid duplicate edges
                    sim = similarity_matrix[i][j]
                    if sim >= threshold:
                        G.add_edge(art_i, art_j, weight=sim)
                        edge_count += 1
        
        # Calculate layout
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
        
        # Draw the graph
        plt.figure(figsize=figsize)
        
        # Draw edges
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(G, pos, 
                              width=[w*4 for w in weights],
                              alpha=0.4,
                              edge_color=weights,
                              edge_cmap=plt.cm.RdYlBu_r,
                              edge_vmin=threshold,
                              edge_vmax=1.0)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_color=article_nums,
                              node_size=1000,
                              cmap=plt.cm.viridis,
                              alpha=0.9,
                              edgecolors='black',
                              linewidths=2)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, 
                               font_size=11,
                               font_weight='bold',
                               font_color='white')
        
        plt.title(f'GDPR Article Similarity Network\n(Showing {edge_count} connections with similarity ≥ {threshold})', 
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save:
            plt.savefig('gdpr_similarity_network.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: gdpr_similarity_network.png ({edge_count} edges shown)")
        plt.show()
        
        return G
    
    def visualize_embedding_space_2d(self, method='tsne', perplexity=20, figsize=(16, 12), save=True):
        """
        Visualize embeddings in 2D using t-SNE or UMAP
        
        Args:
            method: 'tsne' or 'umap'
            perplexity: For t-SNE (adjust based on dataset size)
        """
        article_nums = sorted(self.embeddings.keys())
        embedding_matrix = np.array([self.embeddings[num] for num in article_nums])
        
        # Adjust perplexity based on dataset size
        max_perplexity = (len(article_nums) - 1) // 3
        perplexity = min(perplexity, max_perplexity)
        
        # Reduce to 2D
        print(f"\nReducing to 2D using {method.upper()}...")
        if method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
            coords_2d = reducer.fit_transform(embedding_matrix)
            title_method = 't-SNE'
        else:  # umap
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(article_nums)-1))
            coords_2d = reducer.fit_transform(embedding_matrix)
            title_method = 'UMAP'
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                            c=article_nums, cmap='viridis', 
                            s=300, alpha=0.7, edgecolors='black', linewidth=2)
        
        # Add labels
        for i, num in enumerate(article_nums):
            ax.annotate(str(num), 
                       (coords_2d[i, 0], coords_2d[i, 1]),
                       fontsize=10, fontweight='bold',
                       ha='center', va='center',
                       color='white',
                       bbox=dict(boxstyle='circle,pad=0.3', facecolor='none', edgecolor='none'))
        
        plt.colorbar(scatter, label='Article Number', ax=ax)
        ax.set_title(f'GDPR Articles in 2D Embedding Space ({title_method})\n(Proximity indicates semantic similarity)', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel(f'{title_method} Dimension 1', fontsize=12)
        ax.set_ylabel(f'{title_method} Dimension 2', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            plt.savefig(f'gdpr_embedding_{method}.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: gdpr_embedding_{method}.png")
        plt.show()
    
    def visualize_ground_truth_vs_similarity(self, article_num: int, top_k=10, figsize=(14, 8), save=True):
        """
        Compare ground truth relationships vs. embedding-based similarity for one article
        """
        if article_num not in self.articles:
            print(f"Article {article_num} not found")
            return
        
        # Get ground truth
        ground_truth = set(self.articles[article_num]['related_articles'])
        if not ground_truth:
            print(f"No ground truth for Article {article_num}")
            return
        
        # Get similarity rankings
        article_nums = sorted(self.embeddings.keys())
        embedding_matrix = np.array([self.embeddings[num] for num in article_nums])
        article_idx = article_nums.index(article_num)
        
        similarities = cosine_similarity([embedding_matrix[article_idx]], embedding_matrix)[0]
        
        # Create ranking
        rankings = []
        for i, num in enumerate(article_nums):
            if num != article_num:
                rankings.append({
                    'article': num,
                    'similarity': similarities[i],
                    'in_ground_truth': num in ground_truth
                })
        rankings.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Left: Top-k by similarity
        top_articles = [r['article'] for r in rankings[:top_k]]
        top_sims = [r['similarity'] for r in rankings[:top_k]]
        colors = ['green' if rankings[i]['in_ground_truth'] else 'gray' for i in range(top_k)]
        
        ax1.barh(range(top_k), top_sims, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_yticks(range(top_k))
        ax1.set_yticklabels([f"Art {a}" for a in top_articles])
        ax1.set_xlabel('Cosine Similarity')
        ax1.set_title(f'Top {top_k} Most Similar Articles\n(Green = In Ground Truth)')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # Right: Ground truth similarities
        gt_articles = sorted(list(ground_truth))
        gt_sims = []
        gt_ranks = []
        
        for gt_art in gt_articles:
            idx = article_nums.index(gt_art)
            gt_sims.append(similarities[idx])
            # Find rank
            rank = next(i for i, r in enumerate(rankings) if r['article'] == gt_art) + 1
            gt_ranks.append(rank)
        
        ax2.barh(range(len(gt_articles)), gt_sims, color='green', alpha=0.7, edgecolor='black')
        ax2.set_yticks(range(len(gt_articles)))
        ax2.set_yticklabels([f"Art {a} (rank #{r})" for a, r in zip(gt_articles, gt_ranks)])
        ax2.set_xlabel('Cosine Similarity')
        ax2.set_title(f'Ground Truth Articles\n(with ranking among all articles)')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        plt.suptitle(f'Article {article_num}: {self.articles[article_num]["title"]}\nGround Truth vs. Embedding Similarity', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            plt.savefig(f'gdpr_article_{article_num}_comparison.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: gdpr_article_{article_num}_comparison.png")
        plt.show()
    
    def save_results(self, filepath: str):
        """Save articles and embeddings to file"""
        data = {
            'articles': self.articles,
            'embeddings': {k: v.tolist() for k, v in self.embeddings.items()}
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved results to {filepath}")
    
    def load_results(self, filepath: str):
        """Load articles and embeddings from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.articles = data['articles']
        self.embeddings = {int(k): np.array(v) for k, v in data['embeddings'].items()}
        print(f"✓ Loaded {len(self.articles)} articles from {filepath}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Initialize analyzer (using sentence-transformers for local embedding)
    analyzer = GDPRSimilarityAnalyzer(embedding_model='sentence-transformers')
    
    # Scrape key articles (expandable)
    #key_articles = [1, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 21, 22, 
    #                24, 25, 30, 32, 33, 34, 35, 44, 45, 46, 49, 83]
    #analyzer.scrape_all_articles(key_articles)
    analyzer.scrape_all_articles(range(1, 100))
    
    # Generate embeddings
    analyzer.generate_embeddings()
    
    # Calculate similarities
    similarities = analyzer.calculate_similarities()
    
    # Evaluate against ground truth
    results = analyzer.evaluate_against_ground_truth(similarities, top_k=10)
    
    # Show examples
    print("\n" + "="*80)
    print("EXAMPLE: Article 6 (Lawfulness of processing)")
    print("="*80)
    analyzer.show_top_similarities(6, similarities, top_n=10)
    
    # ========== VISUALIZATIONS ==========
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # 1. Similarity Heatmap (static)
    print("\n1. Creating similarity heatmap...")
    #analyzer.visualize_similarity_heatmap()
    
    # 2. Interactive Heatmap (Plotly)
    print("\n2. Creating interactive heatmap...")
    #analyzer.visualize_interactive_heatmap()
    
    # 3. Network Graph (static)
    print("\n3. Creating network graph (similarity >= 0.7)...")
    #analyzer.visualize_network_graph(threshold=0.7)
    
    # 4. Interactive Network Graph
    print("\n4. Creating interactive network graph...")
    #analyzer.visualize_interactive_network(threshold=0.7)
    
    # 5. 2D Embedding Space (t-SNE)
    print("\n5. Creating t-SNE visualization...")
    #analyzer.visualize_embedding_space_2d(method='tsne')
    
    # 6. 2D Embedding Space (UMAP)
    print("\n6. Creating UMAP visualization...")
    #analyzer.visualize_embedding_space_2d(method='umap')
    
    # 7. Interactive Embedding Space
    print("\n7. Creating interactive embedding space...")
    #analyzer.visualize_interactive_embedding_space(method='tsne')
    
    # 8. Focus on specific article
    print("\n8. Creating cluster visualization for Article 6...")
    #analyzer.visualize_article_cluster_comparison(article_num=6, top_k=5)
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - gdpr_similarity_heatmap.png")
    print("  - gdpr_similarity_interactive.html")
    print("  - gdpr_similarity_network.png")
    print("  - gdpr_network_interactive.html")
    print("  - gdpr_embedding_tsne.png")
    print("  - gdpr_embedding_umap.png")
    print("  - gdpr_embedding_tsne_interactive.html")
    print("  - gdpr_article_6_cluster.png")
    
    # Save results
    analyzer.save_results('gdpr_embeddings.json')