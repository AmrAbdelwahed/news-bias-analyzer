from newspaper import Article, ArticleException
from typing import List, Tuple, Dict
from googlesearch import search
import asyncio
import re
from transformers import pipeline
from nltk import ne_chunk, pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from gensim import corpora, models
import aiohttp
import spacy
from collections import Counter, defaultdict
import gradio as gr
import datetime
import pandas as pd
import json
import nltk
import numpy as np
import time

def datetime_handler(obj):
    """Handler for JSON serialization of datetime objects"""
    if isinstance(obj, datetime.datetime):
        return obj.isoformat() if obj else None
    raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

# Download required NLTK data
nltk.download(['punkt','punkt_tab', 'averaged_perceptron_tagger','averaged_perceptron_tagger_eng', 'maxent_ne_chunker', 'words', 'stopwords'])

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

class NewsAnalyzer:
    def __init__(self):
        # Initialize models
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis")
            self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ar-en")
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            self.stop_words = set(stopwords.words('english'))

        except Exception as e:
            print(f"Error loading models: {e}")
            raise Exception("Error loading models. Please check your internet connection and model names.")
        
        # Load stopwords
        self.stop_words = set(stopwords.words('english'))

    def chunk_text(self, text, max_chunk_size=500):
        """Split text into chunks at sentence boundaries"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            if current_size + sentence_size > max_chunk_size:
                if current_chunk:  # Only append if there's content
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        if current_chunk:  # Append the last chunk
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def summarize_text(self, text):
        """Summarize text by processing it in chunks"""
        if not text:
            return ""
            
        chunks = self.chunk_text(text)
        summaries = []
        
        for chunk in chunks:
            try:
                summary = self.summarizer(chunk, 
                                        max_length=130, 
                                        min_length=30, 
                                        batch_size=1)[0]['summary_text']
                summaries.append(summary)
            except Exception as e:
                print(f"Error summarizing chunk: {e}")
                continue
        
        # Combine summaries and summarize again if too long
        final_summary = ' '.join(summaries)
        if len(final_summary.split()) > 500:
            try:
                final_summary = self.summarizer(final_summary, 
                                              max_length=130, 
                                              min_length=30, 
                                              batch_size=1)[0]['summary_text']
            except Exception as e:
                print(f"Error in final summarization: {e}")
        
        return final_summary
        
    def fetch_article(self, url, is_arabic=False):
        """Fetch and process article content"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            content = article.text
            if is_arabic:
                # Translate content in chunks due to length limitations
                chunks = [content[i:i+512] for i in range(0, len(content), 512)]
                translated_chunks = [self.translator(chunk)[0]['translation_text'] for chunk in chunks]
                content = ' '.join(translated_chunks)
            
            return {
                'title': article.title,
                'content': content,
                'publish_date': article.publish_date,
                'authors': article.authors
            }
        except ArticleException as e:
            raise Exception(f"Error fetching article: {str(e)}")
        
    def analyze_sentiment(self, text):
        """Binary sentiment analysis with confidence scores and improved handling"""
        sentences = sent_tokenize(text)
        
        # Skip empty sentences and handle very short ones
        sentences = [sent for sent in sentences if len(sent.split()) > 3]
        
        if not sentences:
            return {
                'overall_scores': {
                    'positive': 0.5,  # Neutral default
                    'negative': 0.5,
                    'confidence': {
                        'positive': 0.0,
                        'negative': 0.0
                    }
                },
                'sentence_sentiments': []
            }

        # Analyze each sentence
        raw_sentiments = []
        for sent in sentences:
            try:
                sentiment = self.sentiment_analyzer(sent)[0]
                raw_sentiments.append(sentiment)
            except Exception as e:
                print(f"Error analyzing sentence: {e}")
                continue

        # Convert to binary sentiments with balanced scoring
        binary_sentiments = []
        for sentiment in raw_sentiments:
            score = sentiment['score']
            if sentiment['label'] == 'NEGATIVE':
                binary_sentiments.append({
                    'label': 'NEGATIVE',
                    'score': score
                })
            else:
                binary_sentiments.append({
                    'label': 'POSITIVE',
                    'score': score
                })
        
        if not binary_sentiments:
            return {
                'overall_scores': {
                    'positive': 0.5,
                    'negative': 0.5,
                    'confidence': {
                        'positive': 0.0,
                        'negative': 0.0
                    }
                },
                'sentence_sentiments': []
            }

        # Calculate proportions with smoothing
        total_sentences = len(binary_sentiments)
        smoothing_factor = 0.1  # Reduce extreme biases
        
        total_positive = sum(1 for s in binary_sentiments if s['label'] == 'POSITIVE')
        total_negative = total_sentences - total_positive
        
        # Apply smoothing to proportions
        positive_ratio = (total_positive + smoothing_factor) / (total_sentences + 2 * smoothing_factor)
        negative_ratio = (total_negative + smoothing_factor) / (total_sentences + 2 * smoothing_factor)
        
        # Calculate confidence scores
        positive_confidence = np.mean([s['score'] for s in binary_sentiments if s['label'] == 'POSITIVE']) if total_positive > 0 else 0
        negative_confidence = np.mean([s['score'] for s in binary_sentiments if s['label'] == 'NEGATIVE']) if total_negative > 0 else 0
        
        return {
            'overall_scores': {
                'positive': float(positive_ratio),
                'negative': float(negative_ratio),
                'confidence': {
                    'positive': float(positive_confidence),
                    'negative': float(negative_confidence)
                }
            },
            'sentence_sentiments': binary_sentiments
        }


    def extract_named_entities(self, text):
        """Extract and classify named entities"""
        doc = nlp(text)
        entities = defaultdict(int)
        entity_types = defaultdict(str)
        
        for ent in doc.ents:
            entities[ent.text] += 1
            entity_types[ent.text] = ent.label_
        
        # Convert to list of dictionaries
        return [
            {'entity': entity, 'type': entity_types[entity], 'count': count}
            for entity, count in sorted(entities.items(), key=lambda x: x[1], reverse=True)
        ]

    def analyze_topics(self, text):
        """Perform topic modeling"""
        # Tokenize and remove stopwords
        tokens = [word.lower() for word in word_tokenize(text) 
                 if word.isalnum() and word.lower() not in self.stop_words]
        
        # Create dictionary and corpus
        dictionary = corpora.Dictionary([tokens])
        corpus = [dictionary.doc2bow(tokens)]
        
        # Train LDA model
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=5,
            random_state=42,
            passes=20
        )
        
        # Get topics
        topics = lda_model.show_topics(formatted=False)
        return [word for topic in topics for word, _ in topic[1]]

    def analyze_language_patterns(self, text):
        """Analyze language patterns and word usage"""
        # Tokenize and get POS tags
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)
        
        # Extract adjectives and emotional words
        adjectives = [word for word, pos in pos_tags if pos.startswith('JJ')]
        
        # Load emotional words (you might want to use a proper lexicon)
        emotional_words = [word for word, pos in pos_tags 
                         if pos.startswith('JJ') or pos.startswith('RB')]
        
        return {
            'emotional_words': list(Counter(emotional_words).most_common(10)),
            'descriptive_terms': list(Counter(adjectives).most_common(10))
        }

    def compare_articles(self, arabic_url, western_url):
        """Compare articles with enhanced bias detection"""
        arabic_article = self.fetch_article(arabic_url, is_arabic=True)
        western_article = self.fetch_article(western_url, is_arabic=False)
        
        # Store original Arabic text for verification
        original_arabic = arabic_article['content']
        
        # Analyze both articles
        analysis = {
            'arabic': {
                'sentiment': self.analyze_sentiment(arabic_article['content']),
                'entities': self.extract_named_entities(arabic_article['content']),
                'topics': self.analyze_topics(arabic_article['content']),
                'language_patterns': self.analyze_language_patterns(arabic_article['content']),
                'summary': self.summarize_text(arabic_article['content']),
                'translation_confidence': self._assess_translation_quality(original_arabic)
            },
            'western': {
                'sentiment': self.analyze_sentiment(western_article['content']),
                'entities': self.extract_named_entities(western_article['content']),
                'topics': self.analyze_topics(western_article['content']),
                'language_patterns': self.analyze_language_patterns(western_article['content']),
                'summary': self.summarize_text(western_article['content'])
            },
            'bias_metrics': {
                'entity_overlap': self._calculate_entity_overlap(
                    self.extract_named_entities(arabic_article['content']),
                    self.extract_named_entities(western_article['content'])
                ),
                'topic_similarity': self._calculate_topic_similarity(
                    self.analyze_topics(arabic_article['content']),
                    self.analyze_topics(western_article['content'])
                )
            }
        }

        return analysis
    
    def _assess_translation_quality(self, text):
        """Assess translation quality metrics"""
        # Implement basic translation quality checks
        return {
            'length_ratio': len(text.split()) / 100,  # Normalized by 100 words
            'named_entity_preservation': len(self.extract_named_entities(text))
        }

    def _calculate_entity_overlap(self, entities1, entities2):
        """Calculate named entity overlap between articles"""
        entities_set1 = set(e['entity'].lower() for e in entities1)
        entities_set2 = set(e['entity'].lower() for e in entities2)
        
        overlap = len(entities_set1.intersection(entities_set2))
        union = len(entities_set1.union(entities_set2))
        
        return {
            'jaccard_similarity': overlap / union if union > 0 else 0,
            'unique_to_arabic': len(entities_set1 - entities_set2),
            'unique_to_western': len(entities_set2 - entities_set1)
        }

    def _calculate_topic_similarity(self, topics1, topics2):
        """Calculate topic similarity between articles"""
        # Convert topics to sets for comparison
        topics_set1 = set(t.lower() for t in topics1)
        topics_set2 = set(t.lower() for t in topics2)
        
        overlap = len(topics_set1.intersection(topics_set2))
        union = len(topics_set1.union(topics_set2))
        
        return {
            'jaccard_similarity': overlap / union if union > 0 else 0,
            'unique_topics_arabic': list(topics_set1 - topics_set2),
            'unique_topics_western': list(topics_set2 - topics_set1)
        }
    

    def historical_analysis(self, urls_list, timeframe='1M'):
        """Analyze historical trends in coverage"""
        results = []
        for arabic_url, western_url in urls_list:
            analysis = self.compare_articles(arabic_url, western_url)
            results.append({
                'date': analysis['metadata']['arabic']['publish_date'],
                'arabic_sentiment': analysis['arabic']['sentiment']['overall_scores']['positive'],
                'western_sentiment': analysis['western']['sentiment']['overall_scores']['positive']
            })
        
        # Convert to pandas DataFrame for time series analysis
        df = pd.DataFrame(results)
        return df.to_dict('records')


class TopicNewsAnalyzer:
    def __init__(self, news_analyzer: NewsAnalyzer):
        self.news_analyzer = news_analyzer
        # Add Arabic news domains - expand this list as needed
        self.arabic_domains = [
            'aljazeera.net'
        ]
        # Add English news domains - expand this list as needed
        self.english_domains = [
            'bbc.com'

        ]
        
    async def fetch_article_content(self, url: str) -> Dict:
        """Fetch and parse article content quickly"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            return {
                'url': url,
                'status': 'success',
                'title': article.title,
                'text': article.text[:1000]  # Only get first 1000 chars for quick matching
            }
        except Exception as e:
            return {'url': url, 'status': 'error', 'error': str(e)}

    async def quick_match_articles(self, topic: str, max_results: int = 1) -> Tuple[str, str]:
        """Quickly find matching articles with timeouts and limits"""
        start_time = time.time()
        
        # Translate topic to Arabic (with timeout)
        try:
            arabic_topic = self.news_analyzer.translator(topic)[0]['translation_text']
        except Exception as e:
            raise Exception(f"Translation error: {str(e)}")

        # Prepare search queries
        arabic_query = f"{arabic_topic} site:({' OR site:'.join(self.arabic_domains)})"
        english_query = f"{topic} site:({' OR site:'.join(self.english_domains)})"

        # Get URLs with timeout
        try:
            # Set lower max_results and use stop parameter
            arabic_urls = list(search(arabic_query, num=max_results, stop=max_results, pause=1.0))
            english_urls = list(search(english_query, num=max_results, stop=max_results, pause=1.0))
        except Exception as e:
            raise Exception(f"Search error: {str(e)}")

        if not arabic_urls or not english_urls:
            raise Exception("No matching articles found")

        # Fetch first article from each set (with timeout)
        try:
            arabic_content = await self.fetch_article_content(arabic_urls[0])
            english_content = await self.fetch_article_content(english_urls[0])

            if arabic_content['status'] == 'error' or english_content['status'] == 'error':
                raise Exception("Error fetching article content")

            return arabic_urls[0], english_urls[0]

        except Exception as e:
            raise Exception(f"Error processing articles: {str(e)}")

        finally:
            print(f"Total processing time: {time.time() - start_time:.2f} seconds")


# Create Gradio interface with tabs
def create_interface():
    analyzer = NewsAnalyzer()
    topic_analyzer = TopicNewsAnalyzer(analyzer)

    
    def analyze_realtime(arabic_url, western_url):
        analysis = analyzer.compare_articles(arabic_url, western_url)
        return json.dumps(analysis, indent=2, default=datetime_handler)
    
    def analyze_historical(urls_file):
        # Assuming urls_file is a CSV with columns: date, arabic_url, western_url
        urls_df = pd.read_csv(urls_file)
        urls_list = zip(urls_df['arabic_url'], urls_df['western_url'])
        analysis = analyzer.historical_analysis(urls_list)
        return json.dumps(analysis, indent=2, default=datetime_handler)

    def analyze_by_topic(topic: str):
        try:
            # Set timeout for entire operation
            async def run_analysis():
                arabic_url, english_url = await topic_analyzer.quick_match_articles(topic)
                analysis = analyzer.compare_articles(arabic_url, english_url)
                analysis['urls'] = {
                    'arabic': arabic_url,
                    'english': english_url
                }
                return analysis

            # Run with timeout
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            analysis = loop.run_until_complete(
                asyncio.wait_for(run_analysis(), timeout=60)  # 60 second timeout
            )
            loop.close()

            return json.dumps(analysis, indent=2, default=datetime_handler)

        except asyncio.TimeoutError:
            return json.dumps({'error': 'Analysis timed out after 60 seconds'}, indent=2)
        except Exception as e:
            return json.dumps({'error': str(e)}, indent=2)
    
    # Create interfaces for each tab
    realtime_interface = gr.Interface(
        fn=analyze_realtime,
        inputs=[
            gr.Textbox(label="Arabic News URL"),
            gr.Textbox(label="Western News URL")
        ],
        outputs=gr.JSON(label="Analysis Results"),
        title="Real-time News Analysis",
        description="Compare individual articles from Arabic and Western sources"
    )
    
    historical_interface = gr.Interface(
        fn=analyze_historical,
        inputs=gr.File(label="URLs CSV File"),
        outputs=gr.JSON(label="Historical Analysis Results"),
        title="Historical Analysis",
        description="Analyze trends over time"
    )

    topic_interface = gr.Interface(
        fn=analyze_by_topic,
        inputs=gr.Textbox(
            label="Enter news topic or headline (in English)",
            placeholder="e.g., Climate change summit 2024"
        ),
        outputs=[
            gr.JSON(label="Analysis Results")
        ],
        title="Topic-based Analysis",
        description="Find and compare Arabic and Western articles on the same topic"
    )
    
    # Combine interfaces
    return gr.TabbedInterface(
        [realtime_interface, historical_interface, topic_interface],
        ["Real-time Analysis", "Historical Analysis", "Topic Analysis"]
    )

# Launch the interface
if __name__ == "__main__":
    interface = create_interface()
    interface.launch()