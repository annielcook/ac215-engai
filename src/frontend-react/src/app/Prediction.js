import React, { useState } from 'react';
import DataService from './services/DataService';
import ImageUpload from './ImageUpload';

// const Prediction = () => {
//     const [data, setData] = useState({});

//     ShowPrediction: async function(data) {
//         return await axios.post(API_SERVICE_BASE_URL + '/predict', formData, {
//             headers: {
//                 'Content-Type': 'multipart/form-data'
//             }
//         });
//     }


// }
  
//     useEffect(() => {
//       // Make the Axios request here
//       axios.get('https://example.com/api/data')
//         .then(response => {
//           // Assuming the response has attribute1 and attribute2
//           setData(response.data);
//         })
//         .catch(error => {
//           console.error('Error fetching data:', error);
//         });
//     }, []); // Empty dependency array means the effect runs once when the component mounts
  
//     return (
//       <div>
//         <h2>Data Display</h2>
//         {data && (
//           <div>
//             <p>Attribute 1: {data.attribute1}</p>
//             <p>Attribute 2: {data.attribute2}</p>
//           </div>
//         )}
//       </div>
//     );
//   };
  
// export default Prediction;


//   const DataDisplayComponent = () => {
//     const [data, setData] = useState({});
  
//     useEffect(() => {
//       // Make the Axios request here
//       axios.get('https://example.com/api/data')
//         .then(response => {
//           // Assuming the response has attribute1 and attribute2
//           setData(response.data);
//         })
//         .catch(error => {
//           console.error('Error fetching data:', error);
//         });
//     }, []); // Empty dependency array means the effect runs once when the component mounts
  
//     return (
//       <div>
//         <h2>Data Display</h2>
//         {data && (
//           <div>
//             <p>Attribute 1: {data.attribute1}</p>
//             <p>Attribute 2: {data.attribute2}</p>
//           </div>
//         )}
//       </div>
//     );
//   };
  
//   export default DataDisplayComponent;