from transformers import pipeline

class NewsAnalyzer:
    def __init__(self):
        try:
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise Exception("Error loading models. Please check your internet connection and model names.")
    
    def summarize_text(self, text, max_length=240, min_length=30):
        """Summarize the text in chunks if it exceeds model's token limit"""
        # Split the text into chunks of 1024 tokens or fewer
        chunk_size = 1024  # Approximate token limit for the model
        text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        summaries = []
        for chunk in text_chunks:
            summary = self.summarizer(chunk, max_length=max_length, min_length=min_length)[0]['summary_text']
            summaries.append(summary)
        
        # Combine all chunk summaries
        return ' '.join(summaries)

# Example usage
news_analyzer = NewsAnalyzer()
long_text = "Your very long article content here..."
summary = news_analyzer.summarize_text(long_text)
print(summary)
