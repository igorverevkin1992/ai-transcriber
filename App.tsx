import React, { useState, useRef, useCallback } from 'react';
import { UploadForm } from './components/UploadForm';
import { ProcessingStatus } from './components/ProcessingStatus';
import { VerificationDashboard } from './components/VerificationDashboard';
import { ToastContainer, ToastMessage } from './components/Toast';
import { ProcessingStatus as StatusType, ProjectData } from './types';
import { api } from './services/api';
import { CheckCircle2 } from 'lucide-react';

const App: React.FC = () => {
  const [status, setStatus] = useState<StatusType>('IDLE');
  const [progressStep, setProgressStep] = useState<string>('');
  const [progressPercent, setProgressPercent] = useState<number>(0);
  const [projectData, setProjectData] = useState<ProjectData | null>(null);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const abortRef = useRef<AbortController | null>(null);

  const addToast = useCallback((type: 'error' | 'success', text: string) => {
    const id = Date.now().toString();
    setToasts(prev => [...prev, { id, type, text }]);
  }, []);

  const dismissToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  const handleUpload = async (link: string) => {
    setStatus('PROCESSING');
    setProgressStep('');
    setProgressPercent(0);
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const projectId = await api.uploadFile(link);

      await api.pollStatus(
        projectId,
        (step, percent) => {
          setProgressStep(step);
          setProgressPercent(percent);
        },
        controller.signal
      );

      const data = await api.getVerificationData(projectId);
      setProjectData(data);
      setStatus('VERIFICATION');
    } catch (error: any) {
      if (error?.name === 'AbortError') {
        addToast('success', 'Обработка отменена');
      } else {
        addToast('error', error?.message || 'Произошла ошибка при обработке файла');
      }
      setStatus('IDLE');
    } finally {
      abortRef.current = null;
    }
  };

  const handleCancel = () => {
    abortRef.current?.abort();
  };

  const handleFinish = () => {
    setStatus('COMPLETED');
    addToast('success', 'Монтажный лист успешно сформирован');
  };

  return (
    <div className="h-screen bg-gray-50 flex flex-col">
      <ToastContainer messages={toasts} onDismiss={dismissToast} />

      <header className="bg-white border-b border-gray-200 h-14 flex items-center px-4 md:px-6 justify-between shrink-0">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 bg-blue-600 rounded flex items-center justify-center text-white font-bold text-xs">
            AI
          </div>
          <span className="font-semibold text-gray-900">ABTGS</span>
          <span className="text-gray-400 mx-2 hidden sm:inline">/</span>
          <span className="text-sm text-gray-500 hidden sm:inline">Генерация монтажных листов</span>
        </div>
        <div className="text-xs text-gray-400 font-mono">v1.1.0</div>
      </header>

      <main className="flex-1 overflow-hidden relative">
        {status === 'IDLE' && (
          <UploadForm onUpload={handleUpload} />
        )}

        {status === 'PROCESSING' && (
          <ProcessingStatus
            currentStep={progressStep}
            progressPercent={progressPercent}
            onCancel={handleCancel}
          />
        )}

        {status === 'VERIFICATION' && projectData && (
          <VerificationDashboard
            data={projectData}
            onFinish={handleFinish}
            onError={(msg) => addToast('error', msg)}
          />
        )}

        {status === 'COMPLETED' && (
          <div className="flex flex-col items-center justify-center h-full animate-fade-in-up px-4">
            <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mb-6">
              <CheckCircle2 className="w-10 h-10 text-green-600" />
            </div>
            <h2 className="text-2xl font-bold text-gray-900">Готово!</h2>
            <p className="text-gray-500 mt-2 mb-8 text-center">Монтажный лист успешно сформирован и скачан.</p>
            <button
              onClick={() => {
                setStatus('IDLE');
                setProjectData(null);
              }}
              className="px-6 py-2 bg-white border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
            >
              Обработать следующий файл
            </button>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
