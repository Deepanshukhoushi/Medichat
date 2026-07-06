export interface ChatCitation {
  title: string;
  source: string;
  chapter: string;
}

export interface ChatMessage {
  id: string;
  role: 'assistant' | 'user';
  content: string;
  timestamp: string;
  citations?: ChatCitation[];
  liked?: boolean | null;
}

export interface ConversationSummary {
  id: string;
  title: string;
  topic: string;
  updatedAt: string;
  badge?: string;
}
