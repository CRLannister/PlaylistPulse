module.exports = {                                                                                                                                                                                                
  content: [                                                                                                                                                                                                      
    "./src/**/*.{js,jsx,ts,tsx}",                                                                                                                                                                                 
  ],                                                                                                                                                                                                              
  theme: {                                                                                                                                                                                                        
    extend: {                                                                                                                                                                                                     
      colors: {                                                                                                                                                                                                   
        'spotify-green': '#1DB954',                                                                                                                                                                               
      },                                                                                                                                                                                                          
      animation: {                                                                                                                                                                                                
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',                                                                                                                                           
      }                                                                                                                                                                                                           
    },                                                                                                                                                                                                            
  },                                                                                                                                                                                                              
  plugins: [                                                                                                                                                                                                      
    require('@tailwindcss/forms'),                                                                                                                                                                                
  ],                                                                                                                                                                                                              
} 
