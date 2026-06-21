import React, { useState, useEffect, useRef, useCallback } from 'react';
import { BatchUploadForm, EngineType, WhisperModel } from './components/BatchUploadForm';
import { BatchProgress } from './components/BatchProgress';
import { BatchVerification } from './components/BatchVerification';
import { ProcessingStatus } from './components/ProcessingStatus';
import { VerificationDashboard } from './components/VerificationDashboard';
import { StepIndicator } from './components/StepIndicator';
import { ToastContainer, ToastMessage } from './components/Toast';
import { ProcessingStatus as StatusType, ProjectData } from './types';
import { api } from './services/api';
import { loadBatchSession, clearBatchSession, BatchSession } from './services/batchSession';
import { downloadBlob } from './services/download';
import { CheckCircle2, Download, RefreshCw } from 'lucide-react';

const App: React.FC = () => {
  const [status, setStatus] = useState<StatusType>('IDLE');
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
  const [batchPhase, setBatchPhase] = useState<'uploading' | 'processing' | 'done'>('uploading');
  const [lastCompletedInfo, setLastCompletedInfo] = useState<{ filename: string; speakerCount: number; segmentCount: number } | null>(null);
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
    if (projectData) {
      setLastCompletedInfo({
        filename: projectData.original_filename,
        speakerCount: projectData.detected_speakers.length,
        segmentCount: projectData.preview_transcript.length,
      });
    }
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

  const showStepper = status !== 'IDLE';
  const stepperStep: 1 | 2 | 3 = (() => {
    if (status === 'BATCH_PROCESSING' && batchPhase === 'uploading') return 1;
    if (status === 'BATCH_PROCESSING' || status === 'PROCESSING') return 2;
    if (status === 'BATCH_VERIFICATION' || status === 'VERIFICATION') return 3;
    if (status === 'COMPLETED') return 3;
    return 1;
  })();
  const stepperAllComplete = status === 'COMPLETED';

  return (
    <div className="h-screen bg-gray-50 flex flex-col">
      <ToastContainer messages={toasts} onDismiss={dismissToast} />

      <header className="bg-white border-b border-gray-200 h-14 flex items-center px-4 md:px-6 justify-between shrink-0">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 bg-blue-600 rounded flex items-center justify-center text-white font-bold text-xs">
            AI
          </div>
          <span className="font-semibold text-gray-900">ABTGS</span>
        </div>

        {showStepper && (
          <div className="hidden md:flex">
            <StepIndicator currentStep={stepperStep} allComplete={stepperAllComplete} />
          </div>
        )}

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

      {showStepper && (
        <div className="md:hidden bg-white border-b border-gray-200 px-4 py-2">
          <StepIndicator currentStep={stepperStep} allComplete={stepperAllComplete} />
        </div>
      )}

      <main className="flex-1 overflow-hidden relative">
        {status === 'IDLE' && recoveredSession && (
          <div className="mx-4 mt-4 bg-white border border-amber-300 rounded-xl shadow-sm p-5">
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 bg-amber-100 rounded-full flex items-center justify-center flex-shrink-0">
                <RefreshCw className="w-5 h-5 text-amber-600" />
              </div>
              <div className="flex-1">
                <h3 className="text-sm font-bold text-gray-900">Незавершённая обработка</h3>
                <p className="text-sm text-gray-600 mt-1">
                  {recoveredSession.projectIds.length} файлов ожидают завершения. Продолжить отслеживание прогресса?
                </p>
                <div className="flex gap-3 mt-4">
                  <button
                    onClick={async () => {
                      setBatchEngine(recoveredSession.engine as EngineType);
                      setBatchWhisperModel(recoveredSession.whisperModel as WhisperModel);
                      const results = await Promise.allSettled(
                        recoveredSession.projectIds.map(id => api.resumeProject(id))
                      );
                      const failed = results.filter(r => r.status === 'rejected').length;
                      if (failed > 0) {
                        addToast('error', `Не удалось возобновить ${failed} из ${recoveredSession.projectIds.length} файлов`);
                      }
                      setStatus('BATCH_PROCESSING');
                    }}
                    className="px-5 py-2.5 text-sm text-white bg-amber-600 hover:bg-amber-700 rounded-lg font-medium transition-colors"
                  >
                    Продолжить обработку
                  </button>
                  <button
                    onClick={() => {
                      clearBatchSession();
                      setRecoveredSession(null);
                    }}
                    className="px-4 py-2 text-sm text-gray-500 hover:text-red-600 hover:bg-red-50 rounded-md transition-colors"
                  >
                    Отклонить
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {status === 'IDLE' && (
          <BatchUploadForm
            onStartBatch={handleStartBatch}
            onUploadLink={handleUpload}
          />
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
            onPhaseChange={setBatchPhase}
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
            <p className="text-gray-500 mt-2 text-center">Монтажный лист успешно сформирован и скачан.</p>

            {lastCompletedInfo && (
              <div className="mt-4 mb-6 bg-gray-50 rounded-lg px-6 py-3 text-sm text-gray-600 text-center">
                <p className="font-medium text-gray-800">{lastCompletedInfo.filename.replace(/\.[^.]+$/, '.docx')}</p>
                <p className="mt-1">{lastCompletedInfo.speakerCount} спикеров, {lastCompletedInfo.segmentCount} сегментов</p>
              </div>
            )}

            <div className="flex flex-col sm:flex-row gap-3 mt-2">
              <button
                onClick={resetToIdle}
                className="px-5 py-2.5 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors"
              >
                Обработать следующий
              </button>
              <button
                onClick={handleDownloadSaved}
                disabled={isDownloadingSaved}
                className="flex items-center justify-center gap-2 px-5 py-2.5 bg-gray-100 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-200 transition-colors"
              >
                <Download className="w-4 h-4" />
                {isDownloadingSaved ? 'Скачивание...' : 'Скачать все готовые'}
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
