import React, { useState, useEffect, useRef, useCallback } from 'react';
import { UploadForm } from './components/UploadForm';
import { BatchUploadForm, EngineType, WhisperModel } from './components/BatchUploadForm';
import { BatchProgress } from './components/BatchProgress';
import { BatchVerification } from './components/BatchVerification';
import { ProcessingStatus } from './components/ProcessingStatus';
import { VerificationDashboard } from './components/VerificationDashboard';
import { ToastContainer, ToastMessage } from './components/Toast';
import { ProcessingStatus as StatusType, ProjectData } from './types';
import { api } from './services/api';
import { loadBatchSession, clearBatchSession, BatchSession } from './services/batchSession';
import { downloadBlob } from './services/download';
import { CheckCircle2, Download, RefreshCw } from 'lucide-react';

const App: React.FC = () => {
  const [status, setStatus] = useState<StatusType>('IDLE');
  const [mode, setMode] = useState<'single' | 'batch'>('batch');
  const [progressStep, setProgressStep] = useState<string>('');
  const [progressPercent, setProgressPercent] = useState<number>(0);
  const [projectData, setProjectData] = useState<ProjectData | null>(null);
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [batchEngine, setBatchEngine] = useState<EngineType>('whisper');
  const [batchWhisperModel, setBatchWhisperModel] = useState<WhisperModel>('small');
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const [isDownloadingSaved, setIsDownloadingSaved] = useState(false);
  const [recoveredSession, setRecoveredSession] = useState<BatchSession | null>(null);
  const [batchProjectIds, setBatchProjectIds] = useState<string[]>([]);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const session = loadBatchSession();
    if (session) {
      setRecoveredSession(session);
    }
  }, []);

  const toastIdRef = useRef(0);
  const addToast = useCallback((type: 'error' | 'success', text: string) => {
    const id = `toast-${toastIdRef.current++}`;
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

  const handleStartBatch = (files: File[], engine: EngineType, whisperModel: WhisperModel) => {
    setBatchFiles(files);
    setBatchEngine(engine);
    setBatchWhisperModel(whisperModel);
    setStatus('BATCH_PROCESSING');
  };

  const resetToIdle = () => {
    setStatus('IDLE');
    setProjectData(null);
    setBatchFiles([]);
    setBatchProjectIds([]);
  };

  const handleDownloadSaved = async () => {
    setIsDownloadingSaved(true);
    try {
      const blob = await api.batchDownloadSaved();
      downloadBlob(blob, 'transcripts.zip');
      addToast('success', 'Архив скачан');
    } catch (e: unknown) {
      addToast('error', e instanceof Error ? e.message : 'Ошибка скачивания');
    } finally {
      setIsDownloadingSaved(false);
    }
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
        <div className="flex items-center gap-3">
          <button
            onClick={handleDownloadSaved}
            disabled={isDownloadingSaved}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-gray-600 hover:text-gray-900 bg-gray-100 hover:bg-gray-200 disabled:bg-gray-100 disabled:text-gray-400 rounded-md transition-colors"
            title="Скачать все готовые расшифровки из папки completed_docx/"
          >
            <Download className="w-3.5 h-3.5" />
            {isDownloadingSaved ? 'Скачивание...' : 'Скачать готовые'}
          </button>
          <div className="text-xs text-gray-400 font-mono">v1.2.0</div>
        </div>
      </header>

      <main className="flex-1 overflow-hidden relative">
        {/* Recovery banner */}
        {status === 'IDLE' && recoveredSession && (
          <div className="mx-4 mt-4 bg-amber-50 border border-amber-200 rounded-lg p-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <RefreshCw className="w-5 h-5 text-amber-600" />
              <div>
                <p className="text-sm font-medium text-amber-900">
                  Найдена незавершённая обработка ({recoveredSession.projectIds.length} файлов)
                </p>
                <p className="text-xs text-amber-600 mt-0.5">Можно продолжить отслеживание прогресса</p>
              </div>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => {
                  clearBatchSession();
                  setRecoveredSession(null);
                }}
                className="px-3 py-1.5 text-xs text-gray-600 hover:text-gray-800 bg-white border border-gray-200 rounded-md"
              >
                Отклонить
              </button>
              <button
                onClick={() => {
                  setBatchEngine(recoveredSession.engine as EngineType);
                  setBatchWhisperModel(recoveredSession.whisperModel as WhisperModel);
                  setStatus('BATCH_PROCESSING');
                }}
                className="px-3 py-1.5 text-xs text-white bg-amber-600 hover:bg-amber-700 rounded-md font-medium"
              >
                Продолжить
              </button>
            </div>
          </div>
        )}

        {status === 'IDLE' && mode === 'batch' && (
          <BatchUploadForm
            onStartBatch={handleStartBatch}
            onSwitchToSingle={() => setMode('single')}
          />
        )}

        {status === 'IDLE' && mode === 'single' && (
          <div>
            <UploadForm onUpload={handleUpload} />
            <div className="text-center -mt-2 pb-4">
              <button
                onClick={() => setMode('batch')}
                className="text-sm text-gray-500 hover:text-gray-700"
              >
                Пакетная обработка нескольких файлов
              </button>
            </div>
          </div>
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

        {status === 'BATCH_PROCESSING' && (batchFiles.length > 0 || recoveredSession) && (
          <BatchProgress
            files={batchFiles}
            engine={batchEngine}
            whisperModel={batchWhisperModel}
            onDone={(projectIds) => {
              setRecoveredSession(null);
              setBatchProjectIds(projectIds);
              setStatus('BATCH_VERIFICATION');
            }}
            onError={(msg) => addToast('error', msg)}
            recoveredProjectIds={recoveredSession?.projectIds}
            recoveredFileNames={recoveredSession?.fileNames}
          />
        )}

        {status === 'BATCH_VERIFICATION' && batchProjectIds.length > 0 && (
          <BatchVerification
            projectIds={batchProjectIds}
            onDone={resetToIdle}
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
              onClick={resetToIdle}
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
