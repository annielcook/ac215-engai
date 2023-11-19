import './ModelToggle.css'; // Import the CSS file

import React from 'react';


const ModelToggle = ({ selectedModel, onModelChange }) => {
  return (
    <div className="select-container">
    <label htmlFor="custom-select" className="select-label">
      Select Model Type:
    </label>
      <select id="custom-select" className="custom-select" value={selectedModel} onChange={onModelChange}>
        <option value="default" disabled>Select a Model</option>
        <option value="hosted-model">Hosted Model</option>
        <option value="local-model">Local Model</option>
        {/* Add more model options as needed */}
      </select>
    </div>
  );
};

export default ModelToggle;

// import React, { useState } from 'react';

// const ModelToggle = () => {
//   // State to track the selected option
//   const [selectedOption, setSelectedOption] = useState('Hosted Model');

//   // Function to handle dropdown change
//   const handleDropdownChange = (event) => {
//     setSelectedOption(event.target.value);
//   };

//   const fetchDataBasedOnLabel = async (label) => {
//     try {
//       const data = await DataService.getData(label);
//       console.log('Received data:', data);
//     } catch (error) {
//       console.error('Error fetching data:', error);
//     }
//   };

//   return (
//     <div>
//       {/* Dropdown */}
//       <label>
//         Select Model Type:
//         <select value={selectedOption} onChange={handleDropdownChange}>
//           <option value="Hosted Model">Hosted Model</option>
//           <option value="Local Model">Local Model</option>
//         </select>
//       </label>

//       {/* Button to trigger API call
//       <button onClick={handleApiCall}>Make API Call</button> */}
//     </div>
//   );
// };

// export default ModelToggle;