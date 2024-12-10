// src/components/Header.js                                                                                                                                                                                       
import React from 'react';                                                                                                                                                                                        
import { MusicalNoteIcon } from '@heroicons/react/24/solid';                                                                                                                                                      
                                                                                                                                                                                                                  
export function Header() {                                                                                                                                                                                        
  return (                                                                                                                                                                                                        
    <header className="bg-gray-900 shadow-lg">                                                                                                                                                                    
      <div className="container mx-auto px-4 py-6">                                                                                                                                                               
        <div className="flex items-center justify-between">                                                                                                                                                       
          <div className="flex items-center">                                                                                                                                                                     
            <MusicalNoteIcon className="h-8 w-8 text-spotify-green" />                                                                                                                                            
            <h1 className="ml-2 text-2xl font-bold">PlaylistPulse</h1>                                                                                                                                          
          </div>                                                                                                                                                                                                  
          <p className="text-gray-400">Discover Your Next Favorite Song</p>                                                                                                                                       
        </div>                                                                                                                                                                                                    
      </div>                                                                                                                                                                                                      
    </header>                                                                                                                                                                                                     
  );                                                                                                                                                                                                              
} 
