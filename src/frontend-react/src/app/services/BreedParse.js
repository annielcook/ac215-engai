import React from 'react';

const capitalizeWords = (str) => {
  // Capitalize the first letter of every word
  return str.replace(/\b\w/g, (match) => match.toUpperCase());
};

const BreedParse = (responseString) => {
  // Replace "-", "_", and "." with a space
  const stringWithSpaces = responseString.replace(/[-_.]/g, ' ');

  // Capitalize the first letter of every word
  const formattedString = capitalizeWords(stringWithSpaces);

  return formattedString;
};

export default BreedParse;