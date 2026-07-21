export interface ProgressStat {
  label: string;
  value: string;
  trend: string;
}

export interface StudyMetric {
  title: string;
  value: string;
  helper: string;
}

export interface FlashcardCard {
  id: string;
  front: string;
  back: string;
  difficulty: number;
}

export interface FlashcardDeck {
  id: string;
  topic: string;
  created_at: string;
  cards?: FlashcardCard[];
  // Mastery statistics (populated by backend list_decks query)
  total_cards?: number;
  known_cards?: number;
  unknown_cards?: number;
  mastery_percent?: number;
  last_studied_at?: string;
}
