import React from 'react';
import './HistoryPanel.css';

const HistoryPanel = ({ history }) => {
  return (
    <div className="history-container">
      <h2>🕓 Chat History</h2>
      {history.length === 0 ? (
        <p>No conversations yet.</p>
      ) : (
        history.map((session, idx) => (
          <div key={idx} className="history-item">
            <h4>{session.timestamp}</h4>
            <div className="messages">
              {session.messages.map((msg, i) => (
                <div key={i} className={`message ${msg.sender}`}>
                  <strong>{msg.sender === 'user' ? '👤' : '🤖'}</strong>: {msg.text}
                </div>
              ))}
            </div>
          </div>
        ))
      )}
    </div>
  );
};

export default HistoryPanel;
