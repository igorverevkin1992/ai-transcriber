import React from 'react';
import { SpeakerInfo, Candidate } from '../types';
import { ArrowUpDown, User, Mic2, Settings2 } from 'lucide-react';

interface Props {
  speakers: SpeakerInfo[];
  candidates: Candidate[];
  onSwap: (tagId1: string, tagId2: string) => void;
  onUpdateSpeaker: (tagId: string, field: 'name' | 'abbr', value: string) => void;
  onCandidateSelect: (tagId: string, candidateId: string) => void;
}

export const SpeakerMatrix: React.FC<Props> = ({ 
  speakers, 
  candidates, 
  onSwap,
  onUpdateSpeaker,
  onCandidateSelect
}) => {
  
  const getSpeakerName = (s: SpeakerInfo) => {
    if (s.candidate_id) {
      return candidates.find(c => c.id === s.candidate_id)?.name || "Unknown";
    }
    return s.custom_name || `Speaker ${s.tag_id}`;
  };

  const getSpeakerAbbr = (s: SpeakerInfo) => {
    if (s.candidate_id) {
      return candidates.find(c => c.id === s.candidate_id)?.abbr || "UNK";
    }
    return s.custom_abbr || `S${s.tag_id}`;
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 h-full flex flex-col">
      <div className="p-4 border-b border-gray-100 bg-gray-50 rounded-t-lg">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-gray-500 flex items-center gap-2">
          <Settings2 className="w-4 h-4" />
          Матрица Спикеров
        </h2>
      </div>
      
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {speakers.map((speaker, index) => {
          const isNextAvailable = index < speakers.length - 1;
          const nextSpeaker = isNextAvailable ? speakers[index + 1] : null;

          return (
            <React.Fragment key={speaker.tag_id}>
              {/* Speaker Card */}
              <div className={`p-4 rounded-xl border-2 transition-colors ${
                speaker.percentage > 40 ? 'border-blue-100 bg-blue-50/30' : 'border-gray-100 bg-white'
              }`}>
                <div className="flex justify-between items-start mb-3">
                  <div className="flex items-center gap-2">
                    <div className={`p-2 rounded-lg ${speaker.is_tech ? 'bg-gray-200' : 'bg-blue-100'}`}>
                      {speaker.is_tech ? <Mic2 className="w-4 h-4 text-gray-600" /> : <User className="w-4 h-4 text-blue-600" />}
                    </div>
                    <div>
                      <span className="text-xs font-bold text-gray-400 block">TAG ID: {speaker.tag_id}</span>
                      <div className="text-xs text-gray-500 font-medium">Доля эфира: {speaker.percentage}%</div>
                    </div>
                  </div>
                  {/* Progress Bar */}
                  <div className="w-24 h-2 bg-gray-100 rounded-full mt-2">
                    <div 
                      className={`h-full rounded-full ${speaker.is_tech ? 'bg-gray-400' : 'bg-blue-500'}`}
                      style={{ width: `${Math.max(speaker.percentage, 5)}%` }}
                    ></div>
                  </div>
                </div>

                <div className="space-y-3">
                  {/* Candidate Selection */}
                  <div>
                    <label className="block text-xs font-medium text-gray-600 mb-1">Роль / Имя</label>
                    <select 
                      className="w-full text-sm border-gray-300 rounded-md shadow-sm focus:border-blue-500 focus:ring-blue-500 p-2 border"
                      value={speaker.candidate_id || "custom"}
                      onChange={(e) => onCandidateSelect(speaker.tag_id, e.target.value)}
                    >
                      {candidates.map(c => (
                        <option key={c.id} value={c.id}>{c.name}</option>
                      ))}
                      <option value="custom">Ручной ввод...</option>
                    </select>
                  </div>

                  {/* Manual Edits (Visible if Custom or just to override Abbr) */}
                  <div className="grid grid-cols-3 gap-2">
                    <div className="col-span-2">
                      <label className="block text-xs font-medium text-gray-600 mb-1">Имя (Display Name)</label>
                      <input 
                        type="text" 
                        className="w-full text-sm border-gray-300 rounded-md shadow-sm focus:border-blue-500 focus:ring-blue-500 p-2 border"
                        value={getSpeakerName(speaker)}
                        onChange={(e) => onUpdateSpeaker(speaker.tag_id, 'name', e.target.value)}
                        disabled={!!speaker.candidate_id}
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-600 mb-1">Аббр.</label>
                      <input 
                        type="text" 
                        maxLength={4}
                        className="w-full text-sm border-gray-300 rounded-md shadow-sm focus:border-blue-500 focus:ring-blue-500 p-2 border font-bold text-center"
                        value={getSpeakerAbbr(speaker)}
                        onChange={(e) => onUpdateSpeaker(speaker.tag_id, 'abbr', e.target.value)}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Swap Button Connector */}
              {isNextAvailable && nextSpeaker && (
                <div className="flex justify-center -my-3 relative z-10">
                  <button 
                    onClick={() => onSwap(speaker.tag_id, nextSpeaker.tag_id)}
                    className="bg-white border border-gray-200 p-1.5 rounded-full shadow-sm hover:bg-gray-50 hover:text-blue-600 hover:border-blue-300 transition-all group"
                    title="Поменять местами"
                  >
                    <ArrowUpDown className="w-4 h-4 text-gray-400 group-hover:text-blue-500" />
                  </button>
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};