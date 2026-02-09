import React, { useState } from 'react';
import { ProjectData, SpeakerInfo, TranscriptSegment } from '../types';
import { SpeakerMatrix } from './SpeakerMatrix';
import { TranscriptPreview } from './TranscriptPreview';
import { api } from '../services/api';

interface Props {
  data: ProjectData;
  onFinish: () => void;
  onError: (message: string) => void;
}

export const VerificationDashboard: React.FC<Props> = ({ data, onFinish, onError }) => {
  const [speakers, setSpeakers] = useState<SpeakerInfo[]>(data.detected_speakers);
  const [segments, setSegments] = useState<TranscriptSegment[]>(data.preview_transcript);
  const [isDownloading, setIsDownloading] = useState(false);

  const handleSwap = (tagId1: string, tagId2: string) => {
    const s1Index = speakers.findIndex(s => s.tag_id === tagId1);
    const s2Index = speakers.findIndex(s => s.tag_id === tagId2);

    if (s1Index === -1 || s2Index === -1) return;

    const newSpeakers = [...speakers];
    const s1 = newSpeakers[s1Index];
    const s2 = newSpeakers[s2Index];

    const tempCand = s1.candidate_id;
    const tempName = s1.custom_name;
    const tempAbbr = s1.custom_abbr;

    s1.candidate_id = s2.candidate_id;
    s1.custom_name = s2.custom_name;
    s1.custom_abbr = s2.custom_abbr;

    s2.candidate_id = tempCand;
    s2.custom_name = tempName;
    s2.custom_abbr = tempAbbr;

    setSpeakers(newSpeakers);
  };

  const handleUpdateSpeaker = (tagId: string, field: 'name' | 'abbr', value: string) => {
    setSpeakers(prev => prev.map(s => {
      if (s.tag_id !== tagId) return s;
      return {
        ...s,
        candidate_id: null,
        [field === 'name' ? 'custom_name' : 'custom_abbr']: value,
      };
    }));
  };

  const handleCandidateSelect = (tagId: string, candidateId: string) => {
    if (candidateId === 'custom') {
      setSpeakers(prev => prev.map(s => {
        if (s.tag_id !== tagId) return s;
        return { ...s, candidate_id: null, custom_name: 'Новый спикер', custom_abbr: 'НОВ' };
      }));
    } else {
      setSpeakers(prev => prev.map(s => {
        if (s.tag_id !== tagId) return s;
        const candidate = data.candidates.find(c => c.id === candidateId);
        return {
          ...s,
          candidate_id: candidateId,
          custom_name: candidate?.name,
          custom_abbr: candidate?.abbr,
        };
      }));
    }
  };

  const handleEditSegment = (index: number, newText: string) => {
    setSegments(prev => prev.map((seg, i) => (i === index ? { ...seg, text: newText } : seg)));
  };

  const handleDownload = async () => {
    setIsDownloading(true);
    try {
      const mapping = speakers.reduce((acc, s) => {
        acc[s.tag_id] = {
          name: (s.candidate_id
            ? data.candidates.find(c => c.id === s.candidate_id)?.name
            : s.custom_name) || `Спикер ${s.tag_id}`,
          abbreviation: (s.candidate_id
            ? data.candidates.find(c => c.id === s.candidate_id)?.abbr
            : s.custom_abbr) || '',
        };
        return acc;
      }, {} as any);

      const blob = await api.confirmMapping(data.id, mapping);

      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = data.original_filename.replace(/\.[^.]+$/, '.docx');
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      onFinish();
    } catch (e: any) {
      onError(e?.message || 'Ошибка генерации документа');
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 lg:gap-6 h-full p-4 lg:p-6 overflow-auto lg:overflow-hidden">
      <div className="lg:col-span-4 lg:h-full">
        <SpeakerMatrix
          speakers={speakers}
          candidates={data.candidates}
          onSwap={handleSwap}
          onUpdateSpeaker={handleUpdateSpeaker}
          onCandidateSelect={handleCandidateSelect}
        />
      </div>
      <div className="lg:col-span-8 lg:h-full">
        <TranscriptPreview
          segments={segments}
          speakers={speakers}
          candidates={data.candidates}
          filename={data.original_filename}
          onDownload={handleDownload}
          isDownloading={isDownloading}
          onEditSegment={handleEditSegment}
        />
      </div>
    </div>
  );
};
