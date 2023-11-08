import React from 'react';
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
    </React.Fragment>
  )

  // Return View
  return view
}

export default App;