import React, { useState, useEffect, useRef } from 'react';
import { CheckCircle2, Download, Edit3, ChevronDown, ChevronUp, ArrowLeft, ChevronsUpDown } from 'lucide-react';
import { api } from '../services/api';
import { downloadBlob } from '../services/download';

interface SpeakerData {
  id: string;
  name: string;
  abbr: string;
  duration_sec: number;
}

interface ProjectVerification {
  project_id: string;
  filename: string;
  speakers: SpeakerData[];
  preview_segments: Array<{ timecode: string; speaker: string; text: string }>;
  total_segments: number;
}

interface Props {
  projectIds: string[];
  onDone: () => void;
  onError: (msg: string) => void;
}

export const BatchVerification: React.FC<Props> = ({ projectIds, onDone, onError }) => {
  const [projects, setProjects] = useState<ProjectVerification[]>([]);
  const [loading, setLoading] = useState(true);
  const [isExporting, setIsExporting] = useState(false);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [hasEdits, setHasEdits] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [exportSuccess, setExportSuccess] = useState(false);

  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;

  // Allow Escape to dismiss the confirmation modal
  useEffect(() => {
    if (!showConfirm) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setShowConfirm(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [showConfirm]);

  useEffect(() => {
    const load = async () => {
      try {
        const data = await api.batchVerificationData(projectIds);
        setProjects(data.projects);
      } catch (e: unknown) {
        onErrorRef.current(e instanceof Error ? e.message : 'Ошибка загрузки данных верификации');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [projectIds]);

  const updateSpeaker = (projectId: string, speakerId: string, field: 'name' | 'abbr', value: string) => {
    setHasEdits(true);
    setProjects(prev => prev.map(p => {
      if (p.project_id !== projectId) return p;
      return {
        ...p,
        speakers: p.speakers.map(s =>
          s.id === speakerId ? { ...s, [field]: value } : s
        ),
      };
    }));
  };

  const onExportClick = () => {
    if (hasEdits) {
      setShowConfirm(true);
    } else {
      handleExport();
    }
  };

  const handleExport = async () => {
    setIsExporting(true);
    try {
      const exportData = projects.map(p => ({
        project_id: p.project_id,
        mappings: p.speakers.map(s => ({
          speaker_id: s.id,
          name: s.name,
          abbr: s.abbr,
        })),
      }));

      const blob = await api.batchExportWithMappings(exportData);
      downloadBlob(blob, 'transcripts.zip');
      setExportSuccess(true);
    } catch (e: unknown) {
      onError(e instanceof Error ? e.message : 'Ошибка экспорта');
    } finally {
      setIsExporting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-500">Загрузка данных верификации...</div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full p-4 lg:p-6 gap-4">
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
              <CheckCircle2 className="w-5 h-5 text-green-600" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-gray-900">
                Проверка перед скачиванием
              </h2>
              <p className="text-sm text-gray-500">
                {projects.length} файлов готовы. Проверьте спикеров и аббревиатуры.
              </p>
            </div>
          </div>
          <div className="flex gap-3">
            <button
              onClick={onDone}
              className="flex items-center gap-2 px-4 py-2.5 text-gray-600 hover:text-gray-800 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm font-medium transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
              Назад
            </button>
            <button
              onClick={() => {
                if (expandedIds.size === projects.length) {
                  setExpandedIds(new Set());
                } else {
                  setExpandedIds(new Set(projects.map(p => p.project_id)));
                }
              }}
              className="flex items-center gap-2 px-4 py-2.5 text-gray-600 hover:text-gray-800 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm font-medium transition-colors"
            >
              <ChevronsUpDown className="w-4 h-4" />
              {expandedIds.size === projects.length ? 'Свернуть все' : 'Развернуть все'}
            </button>
            <button
              onClick={onExportClick}
              disabled={isExporting || projects.length === 0}
              className="flex items-center gap-2 px-5 py-2.5 bg-green-600 hover:bg-green-700 disabled:bg-gray-300 text-white rounded-lg font-medium transition-colors"
            >
              <Download className="w-4 h-4" />
              {isExporting ? 'Экспорт...' : `Скачать ZIP (${projects.length})`}
            </button>
          </div>
        </div>
      </div>

      {showConfirm && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50" role="dialog" aria-modal="true" aria-label="Подтверждение экспорта">
          <div className="bg-white rounded-lg p-6 max-w-md mx-4 shadow-xl">
            <h3 className="text-lg font-bold text-gray-900 mb-2">Подтвердите экспорт</h3>
            <p className="text-sm text-gray-600 mb-5">
              Вы внесли правки в данные спикеров. Экспортировать сейчас?
              После экспорта правки нельзя будет вернуть.
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setShowConfirm(false)}
                className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800 bg-gray-100 hover:bg-gray-200 rounded-md"
              >
                Отмена
              </button>
              <button
                onClick={() => { setShowConfirm(false); handleExport(); }}
                className="px-4 py-2 text-sm text-white bg-green-600 hover:bg-green-700 rounded-md font-medium"
              >
                Экспортировать
              </button>
            </div>
          </div>
        </div>
      )}

      {exportSuccess && (
        <div className="bg-green-50 border border-green-200 rounded-lg px-4 py-3 flex items-center gap-3">
          <CheckCircle2 className="w-5 h-5 text-green-600 flex-shrink-0" />
          <span className="text-sm font-medium text-green-800">Архив успешно скачан!</span>
          <button
            onClick={onDone}
            className="ml-auto text-sm text-green-700 hover:text-green-900 font-medium"
          >
            Обработать следующие
          </button>
        </div>
      )}

      <div className="flex-1 overflow-y-auto space-y-3">
        {projects.map(proj => (
          <div key={proj.project_id} className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
            <button
              onClick={() => setExpandedIds(prev => {
                const next = new Set(prev);
                if (next.has(proj.project_id)) next.delete(proj.project_id);
                else next.add(proj.project_id);
                return next;
              })}
              aria-expanded={expandedIds.has(proj.project_id)}
              aria-controls={`panel-${proj.project_id}`}
              className="w-full px-5 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
            >
              <div className="flex items-center gap-3">
                <CheckCircle2 className="w-4 h-4 text-green-500 flex-shrink-0" />
                <span className="text-sm font-medium text-gray-900 truncate">{proj.filename}</span>
                <span className="text-xs text-gray-400">
                  {proj.speakers.length} спикер{proj.speakers.length !== 1 ? 'ов' : ''} / {proj.total_segments} сегментов
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="flex gap-1.5">
                  {proj.speakers.map(s => (
                    <span key={s.id} className="text-xs px-2 py-0.5 bg-blue-50 text-blue-700 rounded-full font-mono">
                      {s.abbr}
                    </span>
                  ))}
                </div>
                {expandedIds.has(proj.project_id)
                  ? <ChevronUp className="w-4 h-4 text-gray-400" />
                  : <ChevronDown className="w-4 h-4 text-gray-400" />
                }
              </div>
            </button>

            {expandedIds.has(proj.project_id) && (
              <div id={`panel-${proj.project_id}`} role="region" aria-label={`Спикеры: ${proj.filename}`} className="border-t border-gray-100 px-5 py-4">
                <div className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-3 flex items-center gap-1.5">
                  <Edit3 className="w-3 h-3" />
                  Спикеры
                </div>
                <div className="space-y-2 mb-4">
                  {proj.speakers.map(s => (
                    <div key={s.id} className="flex items-center gap-3">
                      <div className="w-16 text-xs text-gray-400 font-mono flex-shrink-0">{s.id}</div>
                      <input
                        type="text"
                        value={s.name}
                        onChange={e => updateSpeaker(proj.project_id, s.id, 'name', e.target.value)}
                        aria-label={`Имя спикера ${s.id}`}
                        className="flex-1 text-sm border border-gray-200 rounded-md px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-500"
                      />
                      <input
                        type="text"
                        value={s.abbr}
                        onChange={e => updateSpeaker(proj.project_id, s.id, 'abbr', e.target.value)}
                        aria-label={`Аббревиатура спикера ${s.id}`}
                        className="w-20 text-sm border border-gray-200 rounded-md px-3 py-1.5 font-mono text-center focus:outline-none focus:ring-1 focus:ring-blue-500"
                      />
                      <span className="text-xs text-gray-400 w-16 text-right flex-shrink-0">
                        {Math.round(s.duration_sec)}с
                      </span>
                    </div>
                  ))}
                </div>

                {proj.preview_segments.length > 0 && (
                  <div>
                    <div className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">Превью</div>
                    <div className="bg-gray-50 rounded-lg p-3 space-y-1.5 text-sm text-gray-700">
                      {proj.preview_segments.map((seg, i) => {
                        const speaker = proj.speakers.find(s => s.id === seg.speaker);
                        const abbr = speaker?.abbr || seg.speaker;
                        return (
                          <div key={i} className="flex gap-2">
                            <span className="text-gray-400 font-mono text-xs flex-shrink-0 pt-0.5">{seg.timecode}</span>
                            <span>
                              <strong className="text-gray-600">{abbr}:</strong>{' '}
                              {seg.text}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
