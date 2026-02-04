import { ProjectData } from '../types';

const DOVLATOVA_DATA: ProjectData = {
  id: "550e8400-e29b-41d4-a716-446655440000",
  original_filename: "Довлатова Алла, Павел, 05.11.2025_f8.mp3",
  duration_total_ms: 3600000,
  candidates: [
    { id: "cand_1", name: "Довлатова Алла", abbr: "АД" },
    { id: "cand_2", name: "Павел", abbr: "П" },
    { id: "cand_3", name: "Саша", abbr: "С" },
    { id: "cand_4", name: "Мария", abbr: "М" },
  ],
  detected_speakers: [
    { 
      tag_id: "1", 
      candidate_id: "cand_1", 
      total_duration_ms: 2160000, 
      percentage: 60,
      is_tech: false 
    },
    { 
      tag_id: "2", 
      candidate_id: "cand_2", 
      total_duration_ms: 1080000, 
      percentage: 30,
      is_tech: false 
    },
    { 
      tag_id: "3", 
      candidate_id: "cand_4", 
      total_duration_ms: 180000, 
      percentage: 5,
      is_tech: false 
    },
    { 
      tag_id: "4", 
      candidate_id: null, 
      custom_name: "ГЗК",
      custom_abbr: "ГЗК",
      total_duration_ms: 180000, 
      percentage: 5,
      is_tech: true 
    }
  ],
  preview_transcript: [
    { timecode: "15:40:41:00", tag_id: "4", text: "(Технические моменты)" },
    { timecode: "15:40:58:00", tag_id: "1", text: "Сегодня, когда вы будете пить чай или кофе, вот эти финики обязательно всем надо попробовать." },
    { timecode: "15:41:15:00", tag_id: "4", text: "Между собой разговариваем, общаемся. А кофе нет для вас, да?" },
    { timecode: "15:41:18:00", tag_id: "1", text: "А сейчас будет у меня. Вот последняя я только осталась." },
    { timecode: "15:43:07:00", tag_id: "2", text: "Ну, в принципе да." }
  ]
};

const NOSYREV_DATA: ProjectData = {
  id: "project_nosyrev_001",
  original_filename: "30.01.2026_f8.wmv",
  duration_total_ms: 1800000, // 30 min
  candidates: [
    { id: "c_ln", name: "Носырев Леонид", abbr: "ЛН" },
    { id: "c_int", name: "Интервьюер", abbr: "КОРР" },
  ],
  detected_speakers: [
    { 
      tag_id: "1", 
      candidate_id: "c_ln", 
      total_duration_ms: 1350000, // 75%
      percentage: 75,
      is_tech: false 
    },
    { 
      tag_id: "2", 
      candidate_id: "c_int", 
      total_duration_ms: 450000, // 25%
      percentage: 25,
      is_tech: false 
    }
  ],
  preview_transcript: [
    {
      timecode: "00:00:15:12",
      tag_id: "2",
      text: "Леонид Викторович, расскажите, как создавался образ Антошки? Это ведь была одна из первых ваших работ?"
    },
    {
      timecode: "00:00:22:05",
      tag_id: "1",
      text: "Да, это было в альманахе 'Веселая карусель'. Идея пришла, когда я услышал песенку Шаинского. Образ рыжего мальчика... он как-то сам собой нарисовался."
    },
    {
      timecode: "00:00:35:00",
      tag_id: "1",
      text: "Мне хотелось, чтобы он был солнечным, немного вредным, но очень обаятельным. Знаете, такой типичный лентяй, который есть в каждом дворе."
    },
    {
      timecode: "00:00:48:15",
      tag_id: "2",
      text: "А ворона? Она ведь тоже стала культовым персонажем."
    },
    {
      timecode: "00:00:52:10",
      tag_id: "1",
      text: "Ворона появилась чуть позже. Это уже, знаете ли, наблюдательность."
    }
  ]
};

export const getMockData = (inputLink: string): ProjectData => {
  // If the user explicitly asks for Dovlatova or uses a specific test link, give them Dovlatova.
  // Otherwise, default to the Nosyrev case requested by the user for testing.
  if (inputLink.toLowerCase().includes('dovlatova') || inputLink.includes('test_old')) {
    return DOVLATOVA_DATA;
  }
  
  // Defaulting to Nosyrev for any other link to satisfy the current user request
  return NOSYREV_DATA;
};