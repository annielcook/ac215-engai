import React, {useState} from 'react';
import ImageUpload from './components/ImageUpload/ImageUpload.js'
import Footer from './components/Footer/Footer.js';
// import {
//   ThemeProvider,
//   CssBaseline
// } from '@material-ui/core';
import './App.css';



const App = (props) => {
  const [pastPredictions, setPastPredictions] = useState([]);

  const handlePredictionChange = (newPrediction) => {
    // Update the list of past predictions, keeping only the latest 5
    const updatedPredictions = [newPrediction, ...pastPredictions.slice(0, 4)];
    setPastPredictions(updatedPredictions);
  };

  console.log("================================== App ======================================");

  // Build App
  let view = (
    <React.Fragment>
      <div className="background-container">
        <div className="overlay"></div>
        <div className="text-container">
          <h1> 🐶 Welcome to DawgAI 🐶</h1>
          <h2>Please select a model and image to continue</h2>
          <ImageUpload onPredictionChange={handlePredictionChange}/>
        </div>
        {/* <div className="footer">
          <h2>Search History</h2>
        </div> */}
        <Footer pastPredictions={pastPredictions}/>
      </div>
    </React.Fragment>
  );

  // Return View
  return view;
}

export default App;