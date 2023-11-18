import React from 'react';
import ImageUpload from './components/ImageUpload.js'
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
        {/* <ModelToggle></ModelToggle> */}
        <h1>DawgAI</h1>
        <p1>Please select the model type and upload an image:</p1>
        <ImageUpload></ImageUpload>
    </React.Fragment>
  )

  // Return View
  return view
}

export default App;