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
}
