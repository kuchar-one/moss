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
        <nav className="flex items-center justify-center py-4 border-b border-white/5 bg-black/50 backdrop-blur-sm">
          <div className="font-light tracking-[0.3em] text-white/40 text-sm">MOSS <span className="text-purple-500">///</span> AGENTIC</div>
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
      </div>
    </div>
  );
}

export default App;
