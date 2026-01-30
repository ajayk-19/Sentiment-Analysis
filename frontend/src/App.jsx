import { useState } from 'react'
import axios from 'axios'
import { Send, BarChart3, PieChart, Activity, MessageSquare, AlertCircle } from 'lucide-react'
import './App.css'

function App() {
    const [review, setReview] = useState('')
    const [result, setResult] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)

    const handleAnalyze = async () => {
        if (!review.trim()) return
        setLoading(true)
        setError(null)
        try {
            const response = await axios.post('http://localhost:5000/predict', { review })
            setResult(response.data)
        } catch (err) {
            setError('Could not connect to the backend server. Make sure app.py is running.')
            console.error(err)
        } finally {
            setLoading(false)
        }
    }

    const getSentimentColor = (sentiment) => {
        switch (sentiment) {
            case 'Positive': return '#10b981'; // Green
            case 'Negative': return '#ef4444'; // Red
            case 'Neutral': return '#f59e0b'; // Amber
            default: return '#6b7280';
        }
    }

    return (
        <div className="app-container">
            <header className="header">
                <div className="logo">
                    <MessageSquare className="logo-icon" />
                    <h1>Swiggy Sentiment <span>Analysis</span></h1>
                </div>
            </header>

            <main className="content">
                <section className="input-section">
                    <div className="card">
                        <h2>Enter Your Review</h2>
                        <p>Tell us about your experience with Swiggy</p>
                        <textarea
                            placeholder="e.g., The delivery was super fast and the food was delicious!"
                            value={review}
                            onChange={(e) => setReview(e.target.value)}
                            rows={5}
                        />
                        <button
                            className="analyze-btn"
                            onClick={handleAnalyze}
                            disabled={loading || !review.trim()}
                        >
                            {loading ? 'Analyzing...' : <>Analyze Sentiment <Send size={18} /></>}
                        </button>
                    </div>
                </section>

                {error && (
                    <div className="error-msg">
                        <AlertCircle size={20} />
                        {error}
                    </div>
                )}

                {result && (
                    <section className="result-section">
                        <div className="result-card" style={{ borderColor: getSentimentColor(result.sentiment) }}>
                            <div className="result-header">
                                <h3>Prediction Result</h3>
                                <span className="badge" style={{ backgroundColor: getSentimentColor(result.sentiment) }}>
                                    {result.sentiment}
                                </span>
                            </div>
                            <p className="review-text">"{result.review}"</p>

                            {result.probabilities && (
                                <div className="probs-container">
                                    <h4>Confidence Scores</h4>
                                    {Object.entries(result.probabilities).map(([label, prob]) => (
                                        <div key={label} className="prob-row">
                                            <span>{label}</span>
                                            <div className="progress-bar">
                                                <div
                                                    className="progress-fill"
                                                    style={{
                                                        width: `${(prob * 100).toFixed(1)}%`,
                                                        backgroundColor: getSentimentColor(label)
                                                    }}
                                                ></div>
                                            </div>
                                            <span>{(prob * 100).toFixed(1)}%</span>
                                        </div>
                                    ))}
                                </div>
                            )}

                            {result.tokens && result.tokens.length > 0 && (
                                <div className="tokens-viz-container">
                                    <h4>Word Influence (Vector Weights)</h4>
                                    <div className="token-bars">
                                        {result.tokens.map((token, idx) => (
                                            <div key={idx} className="token-bar-item">
                                                <div
                                                    className="token-bar-fill"
                                                    style={{
                                                        height: `${Math.min(Math.max(token.weight * 200, 10), 200)}px`,
                                                        background: 'linear-gradient(to top, #fc8019, #ff5c00)'
                                                    }}
                                                ></div>
                                                <span className="token-label">{token.word}</span>
                                            </div>
                                        ))}
                                    </div>
                                    <p className="token-note">*These bars show the TF-IDF vector weight for each word.</p>
                                </div>
                            )}
                        </div>
                    </section>
                )}

                <section className="info-section">
                    <div className="stats-grid">
                        <div className="stat-card">
                            <Activity className="stat-icon" />
                            <h4>SVM Model</h4>
                            <p>Support Vector Machine with Linear Kernel</p>
                        </div>
                        <div className="stat-card">
                            <BarChart3 className="stat-icon" />
                            <h4>Visualizations</h4>
                            <p>Heatmaps, Histograms, and Scatter Plots generated</p>
                        </div>
                    </div>
                </section>
            </main>

            <footer className="footer">
                <p>Built with React + Vite & Flask Backend</p>
            </footer>
        </div>
    )
}

export default App
