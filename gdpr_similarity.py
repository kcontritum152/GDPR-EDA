"""
GDPR Article Embedding and Similarity Analysis
Validates embeddings using gdpr-info.eu's "relevant articles" as ground truth
"""

import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
from typing import Dict, List, Tuple
import time

# For embeddings, you can choose:
# 1. OpenAI API: from openai import OpenAI
# 2. Sentence Transformers (local): from sentence_transformers import SentenceTransformer
# 3. HuggingFace Transformers

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
            # All-MiniLM-L6-v2 is fast and good for semantic similarity
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        elif embedding_model == 'openai':
            from openai import OpenAI
            self.client = OpenAI()  # Requires OPENAI_API_KEY env variable
            self.model = 'text-embedding-3-small'
        # Add more models as needed
    
    def scrape_gdpr_article(self, article_num: int) -> Dict:
        """
        Scrape a single GDPR article from gdpr-info.eu
        
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
            
            # Extract article text (main content)
            article_text = []
            content = soup.find('div', class_='entry-content') or soup.find('article')
            
            if content:
                # Get all paragraphs and list items
                for elem in content.find_all(['p', 'li', 'ol', 'ul']):
                    text = elem.get_text(strip=True)
                    if text and not text.startswith('Suitable Recitals'):
                        article_text.append(text)
            
            full_text = ' '.join(article_text)
            
            # Extract related articles from inline links
            related_articles = set()
            for link in soup.find_all('a', href=re.compile(r'/art-\d+-gdpr/')):
                match = re.search(r'/art-(\d+)-gdpr/', link['href'])
                if match:
                    related_num = int(match.group(1))
                    if related_num != article_num:  # Don't include self-reference
                        related_articles.add(related_num)
            
            # Extract related recitals
            related_recitals = set()
            recitals_section = soup.find('h2', string=re.compile('Suitable Recitals|Related Recitals'))
            if recitals_section:
                recital_list = recitals_section.find_next(['ul', 'ol'])
                if recital_list:
                    for link in recital_list.find_all('a', href=re.compile(r'/recitals/no-\d+/')):
                        match = re.search(r'/recitals/no-(\d+)/', link['href'])
                        if match:
                            related_recitals.add(int(match.group(1)))
            
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
    
    def scrape_all_articles(self, article_range: range = range(1, 100)):
        """
        Scrape multiple GDPR articles
        """
        print("Scraping GDPR articles...")
        for article_num in article_range:
            article_data = self.scrape_gdpr_article(article_num)
            if article_data and article_data['text']:
                self.articles[article_num] = article_data
                print(f"✓ Article {article_num}: {article_data['title'][:50]}...")
                time.sleep(0.5)  # Be respectful to the server
            else:
                print(f"✗ Article {article_num}: Failed or empty")
        
        print(f"\nSuccessfully scraped {len(self.articles)} articles")
    
    def generate_embeddings(self):
        """
        Generate embeddings for all scraped articles
        """
        print("\nGenerating embeddings...")
        
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
            print(f"✓ Article {article_num} embedded")
        
        print(f"Generated {len(self.embeddings)} embeddings")
    
    def calculate_similarities(self) -> Dict[int, List[Tuple[int, float]]]:
        """
        Calculate cosine similarity between all article pairs
        
        Returns:
            Dict mapping article_num to list of (related_article_num, similarity_score)
        """
        print("\nCalculating cosine similarities...")
        
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
        
        return similarities
    
    def evaluate_against_ground_truth(self, similarities: Dict, top_k: int = 10):
        """
        Evaluate similarity results against gdpr-info.eu's related articles
        
        Args:
            similarities: Output from calculate_similarities()
            top_k: How many top similar articles to consider
        """
        print(f"\n{'='*80}")
        print(f"VALIDATION: Comparing top-{top_k} similar articles vs. gdpr-info.eu references")
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
            
            print(f"Article {article_num}: {self.articles[article_num]['title']}")
            print(f"  Ground truth related: {sorted(list(ground_truth))}")
            print(f"  Found in top-{top_k}: {sorted(list(found_articles))}")
            print(f"  Missed: {sorted(list(ground_truth - found_articles))}")
            print(f"  Precision: {precision:.2%} | Recall: {recall:.2%} | F1: {f1:.2%}")
            print()
        
        # Overall statistics
        overall_recall = total_found / total_related if total_related > 0 else 0
        print(f"{'='*80}")
        print(f"OVERALL RESULTS")
        print(f"{'='*80}")
        print(f"Total ground truth relationships: {total_related}")
        print(f"Total found in top-{top_k}: {total_found}")
        print(f"Overall recall: {overall_recall:.2%}")
        print(f"Average F1 score: {np.mean([r['f1'] for r in results]):.2%}")
        
        return results
    
    def show_top_similarities(self, article_num: int, similarities: Dict, top_n: int = 10):
        """
        Display top N most similar articles for a given article
        """
        if article_num not in similarities:
            print(f"Article {article_num} not found")
            return
        
        print(f"\nTop {top_n} most similar articles to Article {article_num}:")
        print(f"Title: {self.articles[article_num]['title']}")
        print(f"{'-'*80}")
        
        for i, (other_num, score) in enumerate(similarities[article_num][:top_n], 1):
            is_ground_truth = other_num in self.articles[article_num]['related_articles']
            marker = "✓" if is_ground_truth else " "
            print(f"{i:2d}. {marker} Article {other_num:2d} (similarity: {score:.4f})")
            print(f"     {self.articles[other_num]['title']}")
        
        print(f"\n✓ = Referenced on gdpr-info.eu")
    
    def save_results(self, filepath: str):
        """Save articles and embeddings to file"""
        data = {
            'articles': self.articles,
            'embeddings': {k: v.tolist() for k, v in self.embeddings.items()}
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved results to {filepath}")
    
    def load_results(self, filepath: str):
        """Load articles and embeddings from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.articles = data['articles']
        self.embeddings = {int(k): np.array(v) for k, v in data['embeddings'].items()}
        print(f"Loaded {len(self.articles)} articles from {filepath}")


# Example usage
if __name__ == "__main__":
    # Initialize analyzer (using sentence-transformers for local embedding)
    analyzer = GDPRSimilarityAnalyzer(embedding_model='sentence-transformers')
    
    # Scrape key articles (you can expand this range)
    key_articles = [1, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 21, 22, 
                    24, 25, 30, 32, 33, 34, 35, 44, 45, 46, 49, 83]
    analyzer.scrape_all_articles(key_articles)
    
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
    
    # Save results
    analyzer.save_results('gdpr_embeddings.json')