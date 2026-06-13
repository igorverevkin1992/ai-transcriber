import React from 'react';
import { SpeakerInfo, Candidate, UNKNOWN_SPEAKER_NAME, UNKNOWN_SPEAKER_ABBR } from '../types';
import { ArrowUpDown, User, Mic2, Settings2 } from 'lucide-react';

const SPEAKER_COLORS = ['bg-blue-500', 'bg-purple-500', 'bg-orange-500', 'bg-teal-500', 'bg-pink-500'];

interface Props {
  speakers: SpeakerInfo[];
  candidates: Candidate[];
  onSwap: (tagId1: string, tagId2: string) => void;
  onUpdateSpeaker: (tagId: string, field: 'name' | 'abbr', value: string) => void;
}

export const SpeakerMatrix: React.FC<Props> = ({
  speakers,
  candidates,
  onSwap,
  onUpdateSpeaker,
}) => {
  
  const getSpeakerName = (s: SpeakerInfo) => {
    if (s.candidate_id) {
      return candidates.find(c => c.id === s.candidate_id)?.name || UNKNOWN_SPEAKER_NAME;
    }
    return s.custom_name || `Спикер ${s.tag_id}`;
  };

  const getSpeakerAbbr = (s: SpeakerInfo) => {
    if (s.candidate_id) {
      return candidates.find(c => c.id === s.candidate_id)?.abbr || UNKNOWN_SPEAKER_ABBR;
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
      
      <div className="flex-1 overflow-y-auto p-3 md:p-4 space-y-6">
        {speakers.length === 0 && (
          <div className="text-center py-8 text-gray-400 text-sm">Спикеры не обнаружены</div>
        )}
        {speakers.map((speaker, index) => {
          const isNextAvailable = index < speakers.length - 1;
          const nextSpeaker = isNextAvailable ? speakers[index + 1] : null;
          const colorClass = SPEAKER_COLORS[index % SPEAKER_COLORS.length];

          return (
            <React.Fragment key={speaker.tag_id}>
              <div className={`p-3 md:p-4 rounded-xl border-2 transition-colors ${
                speaker.percentage > 40 ? 'border-blue-100 bg-blue-50/30' : 'border-gray-100 bg-white'
              }`}>
                <div className="flex justify-between items-start mb-3">
                  <div className="flex items-center gap-2">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center ${colorClass}`}>
                      {speaker.is_tech
                        ? <Mic2 className="w-4 h-4 text-white" />
                        : <User className="w-4 h-4 text-white" />}
                    </div>
                    <div>
                      <span className="text-xs font-bold text-gray-500 block">Спикер #{speaker.tag_id}</span>
                      <div className="text-xs text-gray-500 font-medium">Доля эфира: {speaker.percentage}%</div>
                    </div>
                  </div>
                  <div className="w-24 h-2 bg-gray-100 rounded-full mt-2">
                    <div
                      className={`h-full rounded-full ${colorClass.replace('bg-', 'bg-')}`}
                      style={{ width: `${Math.max(speaker.percentage, 5)}%` }}
                    />
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-2">
                  <div className="col-span-2">
                    <label className="block text-xs font-medium text-gray-600 mb-1">Отображаемое имя</label>
                    <input
                      type="text"
                      className="w-full text-sm border-gray-300 rounded-md shadow-sm focus:border-blue-500 focus:ring-blue-500 p-2 border"
                      value={getSpeakerName(speaker)}
                      onChange={(e) => onUpdateSpeaker(speaker.tag_id, 'name', e.target.value)}
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

              {isNextAvailable && nextSpeaker && (
                <div className="flex justify-center -my-3 relative z-10">
                  <button
                    onClick={() => onSwap(speaker.tag_id, nextSpeaker.tag_id)}
                    className="bg-white border border-gray-200 p-1.5 rounded-full shadow-sm hover:bg-gray-50 hover:text-blue-600 hover:border-blue-300 transition-all group"
                    title="Поменять местами"
                    aria-label="Поменять спикеров местами"
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