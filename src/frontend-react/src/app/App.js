import React from 'react';
import ImageUpload from './ImageUpload.js'
import {
  ThemeProvider,
  CssBaseline
} from '@material-ui/core';
// import './App.css';


const App = (props) => {

  console.log("================================== App ======================================");

  // Build App
  let view = (
    <React.Fragment>
      <CssBaseline />
        <h1>DawgAI</h1>
        <p1>Upload an image below to get started</p1>
        <ImageUpload></ImageUpload>
    </React.Fragment>
  )

  // Return View
  return view
}

export default App;