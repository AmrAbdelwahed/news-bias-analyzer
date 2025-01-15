import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';

const API_URL = 'http://localhost:8000/api';

const AdvancedNewsAnalyzer = () => {
  const [arabicUrl, setArabicUrl] = useState('');
  const [westernUrl, setWesternUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [historicalData, setHistoricalData] = useState([]);

  const analyzeArticles = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          arabic_url: arabicUrl,
          western_url: westernUrl,
        }),
      });

      if (!response.ok) {
        throw new Error('Analysis failed. Please check your URLs and try again.');
      }

      const data = await response.json();
      setAnalysisData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchHistoricalData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Example URLs for historical analysis
      const sampleUrls = [
        {
          arabic_url: 'https://example.com/arabic1',
          western_url: 'https://example.com/western1',
        },
        // Add more sample URLs as needed
      ];

      const response = await fetch(`${API_URL}/historical`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          urls: sampleUrls,
          timeframe: '1M',
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch historical data');
      }

      const data = await response.json();
      setHistoricalData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistoricalData();
  }, []);

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <Card>
        <CardHeader>
          <h1 className="text-3xl font-bold">Advanced News Bias Analyzer</h1>
          <p className="text-gray-600">Compare and analyze coverage between Arabic and Western news sources</p>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="realtime" className="space-y-4">
            <TabsList>
              <TabsTrigger value="realtime">Real-time Analysis</TabsTrigger>
              <TabsTrigger value="historical">Historical Trends</TabsTrigger>
            </TabsList>

            <TabsContent value="realtime" className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <Input 
                  placeholder="Arabic News URL (e.g., Aljazeera)"
                  value={arabicUrl}
                  onChange={(e) => setArabicUrl(e.target.value)}
                  className="w-full"
                />
                <Input 
                  placeholder="Western News URL (e.g., BBC)"
                  value={westernUrl}
                  onChange={(e) => setWesternUrl(e.target.value)}
                  className="w-full"
                />
              </div>
              
              <Button 
                onClick={analyzeArticles}
                disabled={loading || !arabicUrl || !westernUrl}
                className="w-full"
              >
                {loading ? 'Analyzing...' : 'Compare Articles'}
              </Button>

              {error && (
                <Alert variant="destructive">
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              {analysisData && (
                <div className="grid grid-cols-2 gap-4 mt-6">
                  {/* Sentiment Analysis */}
                  <Card className="p-4">
                    <h3 className="font-semibold mb-4">Sentiment Analysis</h3>
                    <div className="space-y-4">
                      <div>
                        <p className="font-medium">Arabic Source:</p>
                        <div className="flex h-4 rounded-full overflow-hidden">
                          <div 
                            style={{width: `${analysisData.arabic.sentiment.overall_scores.positive * 100}%`}}
                            className="bg-green-400"
                          />
                          <div 
                            style={{width: `${analysisData.arabic.sentiment.overall_scores.neutral * 100}%`}}
                            className="bg-gray-300"
                          />
                          <div 
                            style={{width: `${analysisData.arabic.sentiment.overall_scores.negative * 100}%`}}
                            className="bg-red-400"
                          />
                        </div>
                      </div>
                      <div>
                        <p className="font-medium">Western Source:</p>
                        <div className="flex h-4 rounded-full overflow-hidden">
                          <div 
                            style={{width: `${analysisData.western.sentiment.overall_scores.positive * 100}%`}}
                            className="bg-green-400"
                          />
                          <div 
                            style={{width: `${analysisData.western.sentiment.overall_scores.neutral * 100}%`}}
                            className="bg-gray-300"
                          />
                          <div 
                            style={{width: `${analysisData.western.sentiment.overall_scores.negative * 100}%`}}
                            className="bg-red-400"
                          />
                        </div>
                      </div>
                    </div>
                  </Card>

                  {/* Named Entities */}
                  <Card className="p-4">
                    <h3 className="font-semibold mb-4">Most Mentioned Entities</h3>
                    <div className="space-y-4">
                      <div>
                        <p className="font-medium">Arabic Source:</p>
                        <div className="space-y-2">
                          {analysisData.arabic.entities.slice(0, 5).map(entity => (
                            <div key={entity.entity} className="flex justify-between">
                              <span>{entity.entity}</span>
                              <span className="text-gray-600">{entity.count} mentions</span>
                            </div>
                          ))}
                        </div>
                      </div>
                      <div>
                        <p className="font-medium">Western Source:</p>
                        <div className="space-y-2">
                          {analysisData.western.entities.slice(0, 5).map(entity => (
                            <div key={entity.entity} className="flex justify-between">
                              <span>{entity.entity}</span>
                              <span className="text-gray-600">{entity.count} mentions</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </Card>

                  {/* Topics */}
                  <Card className="p-4">
                    <h3 className="font-semibold mb-4">Main Topics</h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="font-medium mb-2">Arabic Focus:</p>
                        <ul className="list-disc pl-4">
                          {analysisData.arabic.topics.slice(0, 5).map(topic => (
                            <li key={topic}>{topic}</li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <p className="font-medium mb-2">Western Focus:</p>
                        <ul className="list-disc pl-4">
                          {analysisData.western.topics.slice(0, 5).map(topic => (
                            <li key={topic}>{topic}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </Card>

                  {/* Language Patterns */}
                  <Card className="p-4">
                    <h3 className="font-semibold mb-4">Language Patterns</h3>
                    <div className="space-y-4">
                      <div>
                        <p className="font-medium">Key Terms (Arabic):</p>
                        <div className="flex flex-wrap gap-2">
                          {analysisData.arabic.language_patterns.descriptive_terms.map(([term, count]) => (
                            <span key={term} className="px-2 py-1 bg-blue-100 rounded-full text-sm">
                              {term} ({count})
                            </span>
                          ))}
                        </div>
                      </div>
                      <div>
                        <p className="font-medium">Key Terms (Western):</p>
                        <div className="flex flex-wrap gap-2">
                          {analysisData.western.language_patterns.descriptive_terms.map(([term, count]) => (
                            <span key={term} className="px-2 py-1 bg-green-100 rounded-full text-sm">
                              {term} ({count})
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </Card>
                </div>
              )}
            </TabsContent>

            <TabsContent value="historical">
              <Card className="p-4">
                <h3 className="font-semibold mb-4">Historical Sentiment Analysis</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={historicalData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="arabic_sentiment" stroke="#8884d8" name="Arabic Sources" />
                      <Line type="monotone" dataKey="western_sentiment" stroke="#82ca9d" name="Western Sources" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};

export default AdvancedNewsAnalyzer;