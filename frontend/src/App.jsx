import React, { useState } from 'react';
import SelectionGrid from './components/SelectionGrid';
import OptimizerView from './components/OptimizerView';
import { startOptimization } from './api';

function App() {
  const [view, setView] = useState('selection'); // 'selection' | 'optimizer'
  const [session, setSession] = useState(null); // { taskId, imagePath, audioPath }

  const handleStart = async (imagePath, audioPath) => {
    // Start initial optimization (Balanced)
    try {
      const res = await startOptimization({
        image_path: imagePath,
        audio_path: audioPath,
        mode: 'single',
        weights: [0.5, 0.5]
      });

      setSession({
        taskId: res.task_id,
        imagePath,
        audioPath
      });
      setView('optimizer');
    } catch (e) {
      console.error("Failed to start", e);
      alert("Optimization failed to start. Check backend.");
    }
  };

  const handleBack = () => {
    setView('selection');
    setSession(null);
  };

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white font-sans selection:bg-purple-500/30">
      <div className="fixed inset-0 bg-[url('/bg-noise.png')] opacity-5 pointer-events-none mix-blend-overlay"></div>
      <div className="relative z-10 w-full h-screen overflow-hidden flex flex-col">
        <nav className="flex items-center justify-center py-6 border-b border-white/5 bg-black/50 backdrop-blur-md relative z-50">
          <div className="flex items-center space-x-3 group cursor-pointer transition-all hover:scale-105">
            <div className="w-2 h-2 rounded-full bg-purple-500 shadow-[0_0_10px_rgba(168,85,247,0.5)] group-hover:shadow-[0_0_20px_rgba(168,85,247,0.8)] transition-all" />
            <div className="font-sans font-bold text-xl tracking-[0.3em] text-transparent bg-clip-text bg-gradient-to-r from-white via-purple-100 to-white/50 group-hover:to-white transition-all duration-500">
              MOSS
            </div>
          </div>
        </nav>

        <main className="flex-1 overflow-auto">
          {view === 'selection' && (
            <SelectionGrid onStart={handleStart} />
          )}
          {view === 'optimizer' && session && (
            <OptimizerView
              initialTaskId={session.taskId}
              imagePath={session.imagePath}
              audioPath={session.audioPath}
              onBack={handleBack}
            />
          )}
        </main>

        <footer className="flex-shrink-0 text-center py-3 text-white/40 text-xs font-light border-t border-white/5">
          Made with <span className="text-purple-400">💜</span> by{' '}
          <a
            href="https://github.com/kuchar-one"
            target="_blank"
            rel="noopener noreferrer"
            className="text-purple-400 hover:text-purple-300 transition-colors underline underline-offset-2"
          >
            Vojtěch Kuchař
          </a>
        </footer>
      </div>
    </div>
  );
}

export default App;
