import React from 'react';
import { TranscriptSegment, SpeakerInfo, Candidate } from '../types';
import { FileText, Download } from 'lucide-react';

interface Props {
  segments: TranscriptSegment[];
  speakers: SpeakerInfo[];
  candidates: Candidate[];
  filename: string;
  onDownload: () => void;
  isDownloading: boolean;
}

export const TranscriptPreview: React.FC<Props> = ({
  segments,
  speakers,
  candidates,
  filename,
  onDownload,
  isDownloading
}) => {

  const getSpeakerAbbr = (tagId: string) => {
    const speaker = speakers.find(s => s.tag_id === tagId);
    if (!speaker) return tagId;
    if (speaker.candidate_id) {
      return candidates.find(c => c.id === speaker.candidate_id)?.abbr || "UNK";
    }
    return speaker.custom_abbr || `S${tagId}`;
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 h-full flex flex-col">
      <div className="p-4 border-b border-gray-100 bg-gray-50 rounded-t-lg flex justify-between items-center">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-gray-500 flex items-center gap-2">
          <FileText className="w-4 h-4" />
          Live Preview (Times New Roman 12pt)
        </h2>
        <button 
          onClick={onDownload}
          disabled={isDownloading}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {isDownloading ? (
            <span>Генерация...</span>
          ) : (
            <>
              <Download className="w-4 h-4" />
              Скачать .DOCX
            </>
          )}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-8 bg-gray-50">
        {/* A4 Paper Simulation */}
        <div className="max-w-3xl mx-auto bg-white shadow-lg min-h-[800px] p-12 font-times text-black">
          {/* Header Simulation */}
          <div className="mb-8 text-center uppercase font-bold text-lg">
            ИСХОДНИК: {filename}
          </div>

          <div className="mb-8 space-y-1">
            {speakers.map(s => {
               const name = s.candidate_id ? candidates.find(c => c.id === s.candidate_id)?.name : s.custom_name;
               const abbr = getSpeakerAbbr(s.tag_id);
               return (
                 <div key={s.tag_id}>
                   {name?.toUpperCase()} – {abbr},
                 </div>
               )
            })}
          </div>

          <div className="space-y-4 text-lg leading-relaxed">
            {segments.map((seg, idx) => {
              const abbr = getSpeakerAbbr(seg.tag_id);
              const isTechRemark = seg.text.startsWith('(');
              
              if (isTechRemark) {
                return (
                  <div key={idx}>
                    {seg.timecode} <span className="italic">{seg.text}</span>
                  </div>
                )
              }

              return (
                <div key={idx}>
                  {seg.timecode} <strong>{abbr}:</strong> {seg.text}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};