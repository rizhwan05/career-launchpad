import React, { useState, useEffect } from 'react';
import './App.css';
import './ChatBox.css';

const ChatBox = () => {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [userProfile] = useState({ name: "Rizhwanulla", email: "rizhwan2005@gmail.com" });

  // Load from localStorage on start
  useEffect(() => {
    const savedMessages = JSON.parse(localStorage.getItem("chatHistory")) || [];
    setMessages(savedMessages);
  }, []);

  // Save to localStorage whenever messages update
  useEffect(() => {
    localStorage.setItem("chatHistory", JSON.stringify(messages));
  }, [messages]);

  const handleQueryChange = (e) => setQuery(e.target.value);

  const handleQuerySubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMessage = { text: query, sender: 'user', timestamp: new Date().toISOString() };
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/query/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      const data = await response.json();
      if (data.answers) {
        data.answers.forEach(answer => {
          const botMessage = { text: answer, sender: 'bot', timestamp: new Date().toISOString() };
          setMessages(prev => [...prev, botMessage]);
        });
      }
    } catch (error) {
      setMessages(prev => [...prev, { text: "Something went wrong. Please try again.", sender: 'bot' }]);
    } finally {
      setLoading(false);
    }

    setQuery('');
  };

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
    document.body.classList.toggle('dark', !darkMode);
  };

  const clearHistory = () => {
    localStorage.removeItem("chatHistory");
    setMessages([]);
  };

  return (
    <div className={`chat-container ${darkMode ? 'dark' : ''}`}>
      <div className="header">
        <h1>🎓 Hey There, I'm CIT GPT</h1>
        <div className="nav-controls">
          <button className="toggle-btn" onClick={toggleDarkMode}>
            {darkMode ? "☀️" : "🌙"}
          </button>
          <button className="toggle-btn" onClick={() => setShowHistory(!showHistory)}>
            🕓
          </button>
          <div className="profile-dropdown">
            <button className="toggle-btn">👤</button>
            <div className="profile-content">
              <p><strong>Name:</strong> {userProfile.name}</p>
              <p><strong>Email:</strong> {userProfile.email}</p>
              <p><strong>Total Chats:</strong> {messages.length}</p>
              <button onClick={clearHistory}>🗑️ Clear History</button>
            </div>
          </div>
        </div>
      </div>

      {showHistory && (
        <div className="history-panel">
          <h3>🕓 Chat History</h3>
          {messages.length === 0 ? (
            <p>No chats yet.</p>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx} className={`history-msg ${msg.sender}`}>
                <span>{msg.sender === 'user' ? '👤' : '🤖'} {msg.text}</span>
              </div>
            ))
          )}
        </div>
      )}

      <div className="message-list">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.sender}`}>
            <div className="avatar">{msg.sender === 'user' ? '👤' : '🤖'}</div>
            <div className="text">{msg.text}</div>
          </div>
        ))}
        {loading && (
          <div className="message bot typing">
            <div className="avatar">🤖</div>
            <div className="text">
              <span className="dot"></span>
              <span className="dot"></span>
              <span className="dot"></span>
            </div>
          </div>
        )}
      </div>

      <form onSubmit={handleQuerySubmit} className="chat-input">
        <input
          type="text"
          value={query}
          onChange={handleQueryChange}
          placeholder="Type your question..."
        />
        <button type="submit" className="send-btn" disabled={loading}>
          ✈️
        </button>
      </form>
    </div>
  );
};

export default ChatBox;
