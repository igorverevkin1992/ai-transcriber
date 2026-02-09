import React, { useState, useEffect, useRef, useCallback } from 'react';
import { CheckCircle2, XCircle, Loader2, Download, AlertCircle } from 'lucide-react';
import { api } from '../services/api';
import { BatchFileInfo } from '../types';

interface Props {
  files: File[];
  onDone: () => void;
  onError: (msg: string) => void;
}

type UploadState = 'uploading' | 'processing' | 'done';

interface FileTracker {
  file: File;
  projectId?: string;
  uploadError?: string;
}

const STATUS_ICONS: Record<string, React.ReactNode> = {
  completed: <CheckCircle2 className="w-4 h-4 text-green-500" />,
  error: <XCircle className="w-4 h-4 text-red-500" />,
};

const CONCURRENT_UPLOADS = 3;

export const BatchProgress: React.FC<Props> = ({ files, onDone, onError }) => {
  const [state, setState] = useState<UploadState>('uploading');
  const [trackers, setTrackers] = useState<FileTracker[]>(() =>
    files.map(f => ({ file: f }))
  );
  const [uploadedCount, setUploadedCount] = useState(0);
  const [batchFiles, setBatchFiles] = useState<BatchFileInfo[]>([]);
  const [completedCount, setCompletedCount] = useState(0);
  const [errorCount, setErrorCount] = useState(0);
  const [isDownloading, setIsDownloading] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval>>();
  const abortedRef = useRef(false);

  // Phase 1: Upload files to server
  useEffect(() => {
    let cancelled = false;

    const uploadAll = async () => {
      const queue = [...files];
      const updated = [...trackers];
      let doneCount = 0;

      const uploadOne = async (index: number) => {
        if (cancelled) return;
        const file = queue[index];
        try {
          const projectId = await api.batchUploadFile(file);
          updated[index] = { ...updated[index], projectId };
        } catch (e: any) {
          updated[index] = { ...updated[index], uploadError: e.message };
        }
        doneCount++;
        if (!cancelled) {
          setTrackers([...updated]);
          setUploadedCount(doneCount);
        }
      };

      // Upload in batches of CONCURRENT_UPLOADS
      for (let i = 0; i < queue.length; i += CONCURRENT_UPLOADS) {
        if (cancelled) break;
        const batch = [];
        for (let j = i; j < Math.min(i + CONCURRENT_UPLOADS, queue.length); j++) {
          batch.push(uploadOne(j));
        }
        await Promise.all(batch);
      }

      if (!cancelled) {
        setTrackers([...updated]);
        setState('processing');
      }
    };

    uploadAll();
    return () => { cancelled = true; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Phase 2: Poll for processing status
  useEffect(() => {
    if (state !== 'processing') return;

    const projectIds = trackers
      .filter(t => t.projectId)
      .map(t => t.projectId!);

    if (projectIds.length === 0) {
      setState('done');
      return;
    }

    const poll = async () => {
      try {
        const status = await api.batchStatus(projectIds);
        setBatchFiles(status.files);
        setCompletedCount(status.completed);
        setErrorCount(status.errors);

        if (status.in_progress === 0) {
          setState('done');
          if (pollRef.current) clearInterval(pollRef.current);
        }
      } catch {
        // Ignore polling errors, retry next interval
      }
    };

    poll();
    pollRef.current = setInterval(poll, 3000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [state, trackers]);

  const handleDownload = useCallback(async () => {
    setIsDownloading(true);
    try {
      const projectIds = trackers
        .filter(t => t.projectId)
        .map(t => t.projectId!);

      const blob = await api.batchDownload(projectIds);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'transcripts.zip';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (e: any) {
      onError(e?.message || 'Ошибка скачивания архива');
    } finally {
      setIsDownloading(false);
    }
  }, [trackers, onError]);

  const totalFiles = files.length;
  const uploadPercent = Math.round((uploadedCount / totalFiles) * 100);
  const processPercent = totalFiles > 0 ? Math.round(((completedCount + errorCount) / totalFiles) * 100) : 0;

  // Build a unified file status list
  const fileStatuses: { name: string; status: string; label: string; error?: string }[] = trackers.map(t => {
    if (t.uploadError) {
      return { name: t.file.name, status: 'error', label: 'Ошибка загрузки', error: t.uploadError };
    }
    if (!t.projectId) {
      if (state === 'uploading') {
        return { name: t.file.name, status: 'waiting', label: 'Ожидание...' };
      }
      return { name: t.file.name, status: 'error', label: 'Не загружен' };
    }
    const batchFile = batchFiles.find(bf => bf.id === t.projectId);
    if (batchFile) {
      return { name: t.file.name, status: batchFile.status, label: batchFile.status_label, error: batchFile.error };
    }
    if (state === 'uploading') {
      return { name: t.file.name, status: 'uploaded', label: 'Загружен' };
    }
    return { name: t.file.name, status: 'queued', label: 'В очереди' };
  });

  return (
    <div className="flex flex-col h-full p-4 lg:p-6">
      {/* Header */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-4">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-gray-900">
            {state === 'uploading' && 'Загрузка файлов на сервер...'}
            {state === 'processing' && 'Обработка файлов...'}
            {state === 'done' && 'Обработка завершена'}
          </h2>
          {state === 'done' && (
            <button
              onClick={handleDownload}
              disabled={isDownloading || completedCount === 0}
              className="flex items-center gap-2 px-5 py-2.5 bg-green-600 hover:bg-green-700 disabled:bg-gray-300 text-white rounded-lg font-medium transition-colors"
            >
              <Download className="w-4 h-4" />
              {isDownloading ? 'Скачивание...' : `Скачать ZIP (${completedCount} файлов)`}
            </button>
          )}
        </div>

        {/* Progress bar */}
        {state === 'uploading' && (
          <div>
            <div className="flex justify-between text-sm text-gray-500 mb-1">
              <span>Загрузка: {uploadedCount} / {totalFiles}</span>
              <span>{uploadPercent}%</span>
            </div>
            <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
              <div className="h-full bg-purple-600 rounded-full transition-all duration-300" style={{ width: `${uploadPercent}%` }} />
            </div>
          </div>
        )}

        {(state === 'processing' || state === 'done') && (
          <div>
            <div className="flex justify-between text-sm text-gray-500 mb-1">
              <span>
                Готово: {completedCount} / {totalFiles}
                {errorCount > 0 && <span className="text-red-500 ml-2">(ошибок: {errorCount})</span>}
              </span>
              <span>{processPercent}%</span>
            </div>
            <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-300 ${state === 'done' ? 'bg-green-500' : 'bg-blue-600'}`}
                style={{ width: `${processPercent}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* File list */}
      <div className="flex-1 bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden flex flex-col">
        <div className="px-4 py-3 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
          <span className="text-sm font-medium text-gray-700">Файлы ({totalFiles})</span>
          {state === 'processing' && (
            <span className="flex items-center gap-1 text-xs text-blue-600">
              <Loader2 className="w-3 h-3 animate-spin" />
              Обновление каждые 3 сек
            </span>
          )}
        </div>
        <div className="flex-1 overflow-y-auto divide-y divide-gray-100">
          {fileStatuses.map((fs, i) => (
            <div key={i} className="flex items-center px-4 py-2.5 hover:bg-gray-50">
              <div className="flex-shrink-0 mr-3">
                {fs.status === 'completed' ? STATUS_ICONS.completed :
                 fs.status === 'error' ? STATUS_ICONS.error :
                 fs.status === 'waiting' || fs.status === 'queued' ? (
                   <div className="w-4 h-4 rounded-full border-2 border-gray-300" />
                 ) : (
                   <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
                 )}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm text-gray-800 truncate">{fs.name}</p>
              </div>
              <div className="ml-3 flex-shrink-0">
                <span className={`text-xs font-medium ${
                  fs.status === 'completed' ? 'text-green-600' :
                  fs.status === 'error' ? 'text-red-600' :
                  'text-gray-500'
                }`}>
                  {fs.label}
                </span>
              </div>
              {fs.error && (
                <div className="ml-2 flex-shrink-0" title={fs.error}>
                  <AlertCircle className="w-4 h-4 text-red-400" />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      {state === 'done' && (
        <div className="mt-4 text-center">
          <button
            onClick={onDone}
            className="text-sm text-gray-500 hover:text-gray-700"
          >
            Вернуться на главную
          </button>
        </div>
      )}
    </div>
  );
};
