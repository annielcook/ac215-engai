import React, { useState, useRef } from 'react';
import './Footer.css'; // Import the CSS file

const Footer = ({ pastPredictions }) => {
    return (
      <div className="footer">
        <h2>Past Predictions:</h2>
        <ul style={{ listStyleType: 'none', padding: 0 }}>
          {pastPredictions.map((prediction, index) => (
            <li key={index}>{prediction}</li>
          ))}
        </ul>
        {/* Other footer content */}
      </div>
    );
  };
  
  export default Footer;