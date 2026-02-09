import React, { useState } from 'react';
import { TranscriptSegment, SpeakerInfo, Candidate } from '../types';
import { FileText, Download, Pencil } from 'lucide-react';

interface Props {
  segments: TranscriptSegment[];
  speakers: SpeakerInfo[];
  candidates: Candidate[];
  filename: string;
  onDownload: () => void;
  isDownloading: boolean;
  onEditSegment: (index: number, newText: string) => void;
}

export const TranscriptPreview: React.FC<Props> = ({
  segments,
  speakers,
  candidates,
  filename,
  onDownload,
  isDownloading,
  onEditSegment,
}) => {
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editText, setEditText] = useState('');

  const getSpeakerAbbr = (tagId: string) => {
    const speaker = speakers.find(s => s.tag_id === tagId);
    if (!speaker) return tagId;
    if (speaker.candidate_id) {
      return candidates.find(c => c.id === speaker.candidate_id)?.abbr || 'UNK';
    }
    return speaker.custom_abbr || `S${tagId}`;
  };

  const startEdit = (idx: number, text: string) => {
    setEditingIndex(idx);
    setEditText(text);
  };

  const commitEdit = () => {
    if (editingIndex !== null) {
      onEditSegment(editingIndex, editText);
      setEditingIndex(null);
    }
  };

  const cancelEdit = () => {
    setEditingIndex(null);
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 h-full flex flex-col">
      <div className="p-3 md:p-4 border-b border-gray-100 bg-gray-50 rounded-t-lg flex justify-between items-center gap-2">
        <h2 className="text-xs md:text-sm font-semibold uppercase tracking-wide text-gray-500 flex items-center gap-2">
          <FileText className="w-4 h-4" />
          <span className="hidden sm:inline">Live Preview</span>
        </h2>
        <button
          onClick={onDownload}
          disabled={isDownloading}
          className="flex items-center gap-2 px-3 md:px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700 disabled:opacity-50 transition-colors shrink-0"
        >
          {isDownloading ? (
            <span>Генерация...</span>
          ) : (
            <>
              <Download className="w-4 h-4" />
              <span className="hidden sm:inline">Скачать</span> .DOCX
            </>
          )}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 md:p-8 bg-gray-50">
        <div className="max-w-3xl mx-auto bg-white shadow-lg min-h-[800px] p-6 md:p-12 font-times text-black">
          <div className="mb-8 text-center uppercase font-bold text-base md:text-lg">
            ИСХОДНИК: {filename}
          </div>

          <div className="mb-8 space-y-1">
            {speakers.map(s => {
              const name = s.candidate_id ? candidates.find(c => c.id === s.candidate_id)?.name : s.custom_name;
              const abbr = getSpeakerAbbr(s.tag_id);
              return (
                <div key={s.tag_id}>
                  {name?.toUpperCase()} ({abbr})
                </div>
              );
            })}
          </div>

          <div className="space-y-3 text-base md:text-lg leading-relaxed">
            {segments.map((seg, idx) => {
              const abbr = getSpeakerAbbr(seg.tag_id);
              const isTechRemark = seg.text.startsWith('(');
              const isEditing = editingIndex === idx;

              return (
                <div key={idx} className="group relative">
                  {isEditing ? (
                    <div className="flex flex-col gap-2">
                      <div className="text-sm text-gray-400">
                        {seg.timecode} <strong>{abbr}:</strong>
                      </div>
                      <textarea
                        className="w-full border border-blue-300 rounded p-2 text-base font-times focus:outline-none focus:ring-2 focus:ring-blue-500"
                        value={editText}
                        onChange={(e) => setEditText(e.target.value)}
                        rows={3}
                        autoFocus
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); commitEdit(); }
                          if (e.key === 'Escape') cancelEdit();
                        }}
                      />
                      <div className="flex gap-2">
                        <button
                          onClick={commitEdit}
                          className="px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700"
                        >
                          Сохранить
                        </button>
                        <button
                          onClick={cancelEdit}
                          className="px-3 py-1 text-xs bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
                        >
                          Отмена
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-start gap-1">
                      <div className="flex-1">
                        {isTechRemark ? (
                          <span>
                            {seg.timecode} <span className="italic">{seg.text}</span>
                          </span>
                        ) : (
                          <span>
                            {seg.timecode} <strong>{abbr}:</strong> {seg.text}
                          </span>
                        )}
                      </div>
                      <button
                        onClick={() => startEdit(idx, seg.text)}
                        className="opacity-0 group-hover:opacity-100 transition-opacity p-1 text-gray-400 hover:text-blue-600 shrink-0"
                        title="Редактировать"
                      >
                        <Pencil className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};
