import React from 'react';

const ModelToggle = ({ selectedModel, onModelChange }) => {
  return (
    <label>
      Select Model Type: 
      <select value={selectedModel} onChange={onModelChange}>
        <option value="default" disabled>Select a Model</option>
        <option value="hosted-model">Hosted Model</option>
        <option value="local-model">Local Model</option>
        {/* Add more model options as needed */}
      </select>
    </label>
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