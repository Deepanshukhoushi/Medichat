import { HttpClient, HttpContext, HttpEvent, HttpHeaders, HttpParams } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import type { FlashcardDeck } from '../../shared/models/dashboard.model';
import { RuntimeConfigService } from './runtime-config.service';
import { SKIP_GLOBAL_ERROR } from '../interceptors/skip-global-error.token';

export interface BackendHealthResponse {
  status: 'healthy' | 'degraded';
  service: string;
  checks: Record<string, string>;
}

export interface AuthResponse {
  message: string;
  error?: string;
  type?: string;
}

export interface TopicMemoryResponse {
  current_topic: string | null;
  related_topics: string[];
}

export interface SessionSummaryResponse {
  summary: string | null;
}

export interface UploadDocumentResponse {
  chunks_indexed: number;
  filename: string;
  message: string;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at?: string;
}

export interface QuizSession {
  id: string;
  topic: string;
  score: number | null;
  completed_at: string | null;
  created_at: string;
}

export type { FlashcardDeck } from '../../shared/models/dashboard.model';

export interface UserProfile {
  user_id: string;
  display_name: string;
  medical_year: number | null;
  specialty: string;
  university: string;
}

export interface StudyStats {
  total_sessions: number;
  total_messages: number;
  unique_topics: number;
  streak_days: number;
}

export interface DashboardStats {
  stats: StudyStats;
  recent_conversations: Conversation[];
}

@Injectable({ providedIn: 'root' })
export class BackendApiService {
  private readonly http = inject(HttpClient);
  private readonly runtimeConfig = inject(RuntimeConfigService);

  get health$(): Observable<BackendHealthResponse> {
    return this.http.get<BackendHealthResponse>(this.url('/health'), { withCredentials: true });
  }

  getHealth(): Observable<BackendHealthResponse> {
    return this.health$;
  }

  streamChatMessage(prompt: string, conversationId: string | null = null, abortSignal?: AbortSignal, isRegenerate: boolean = false): Observable<{token?: string, error?: string, conversation_id?: string, message_id?: string, sources?: {title: string, source: string, chapter: string}[]}> {
    return new Observable<{token?: string, error?: string, conversation_id?: string, message_id?: string, sources?: {title: string, source: string, chapter: string}[]}>(observer => {
      let body = new HttpParams().set('msg', prompt);
      if (conversationId) {
        body = body.set('conversation_id', conversationId);
      }
      if (isRegenerate) {
        body = body.set('is_regenerate', 'true');
      }

      // Read CSRF token from cookie
      const csrfToken = this.readCookie('csrf_token');

      fetch(this.url('/api/chat/stream'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          'X-CSRF-Token': csrfToken || '',
        },
        body: body.toString(),
        credentials: 'include',
        signal: abortSignal
      }).then(async response => {
        if (!response.ok) {
          observer.error(`Server error: ${response.status} ${response.statusText}`);
          return;
        }

        if (!response.body) {
          observer.error('ReadableStream not supported in this browser.');
          return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const messages = buffer.split('\n\n');
          buffer = messages.pop() || '';

          for (const msg of messages) {
            if (msg.startsWith('data: ')) {
              const data = msg.slice(6);
              if (data === '[DONE]') {
                observer.complete();
                return;
              }
              try {
                const parsed = JSON.parse(data);
                // unescape newlines
                if (parsed.token) {
                  parsed.token = parsed.token.replace(/\\n/g, '\n').replace(/\\r/g, '\r');
                }
                observer.next(parsed);
              } catch (e) {
                // Silently swallow parse errors. Streaming chunks can be cut mid-JSON
                // over the wire; we just wait for the rest of the buffer.
              }
            }
          }
        }
        observer.complete();
      }).catch(err => {
        if (err.name === 'AbortError') {
          observer.complete();
        } else {
          observer.error(err);
        }
      });
    });
  }

  getTopicMemory(): Observable<TopicMemoryResponse> {
    return this.http.get<TopicMemoryResponse>(this.url('/api/memory/topic'), { withCredentials: true });
  }

  getSessionSummary(): Observable<SessionSummaryResponse> {
    return this.http.get<SessionSummaryResponse>(this.url('/api/memory/summary'), { withCredentials: true });
  }

  signup(email: string, password: string, displayName?: string): Observable<AuthResponse> {
    return this.http.post<AuthResponse>(
      this.url('/signup'),
      { email, password, display_name: displayName ?? null },
      { withCredentials: true }
    );
  }

  login(email: string, password: string, rememberMe: boolean = false): Observable<AuthResponse> {
    return this.http.post<AuthResponse>(this.url('/login'), { email, password, remember_me: rememberMe }, { withCredentials: true });
  }

  googleLoginUrl(): string {
    return this.url('/api/auth/google/login');
  }

  logout(): Observable<AuthResponse> {
    return this.http.post<AuthResponse>(this.url('/logout'), {}, { withCredentials: true });
  }

  sendPasswordResetEmail(email: string): Observable<AuthResponse> {
    return this.http.post<AuthResponse>(this.url('/reset-password-request'), { email }, { withCredentials: true });
  }

  updatePassword(password: string, accessToken: string): Observable<AuthResponse> {
    return this.http.post<AuthResponse>(this.url('/reset-password'), { password, access_token: accessToken }, { withCredentials: true });
  }

  // --- Conversations ---
  getConversations(limit = 30, offset = 0): Observable<Conversation[]> {
    let params = new HttpParams().set('limit', limit.toString());
    if (offset > 0) {
      params = params.set('offset', offset.toString());
    }
    return this.http.get<Conversation[]>(this.url('/api/conversations'), {
      withCredentials: true,
      params
    });
  }

  deleteConversation(id: string): Observable<{ message: string }> {
    return this.http.delete<{ message: string }>(this.url(`/api/conversations/${id}`), { withCredentials: true });
  }

  getConversationMessages(id: string): Observable<{ id?: string, role: string, content: string, liked?: boolean }[]> {
    return this.http.get<{ id?: string, role: string, content: string, liked?: boolean }[]>(this.url(`/api/conversations/${id}/messages`), { withCredentials: true });
  }

  deleteMessage(conversationId: string, messageId: string): Observable<{ status: string }> {
    return this.http.delete<{ status: string }>(this.url(`/api/conversations/${conversationId}/messages/${messageId}`), { withCredentials: true });
  }

  rateMessage(conversationId: string, messageId: string, liked: boolean): Observable<{ status: string }> {
    return this.http.post<{ status: string }>(this.url(`/api/conversations/${conversationId}/messages/${messageId}/feedback`), { liked }, { withCredentials: true });
  }

  // --- Flashcards ---
  getFlashcardDecks(): Observable<FlashcardDeck[]> {
    return this.http.get<FlashcardDeck[]>(this.url('/api/flashcards'), {
      withCredentials: true,
      context: new HttpContext().set(SKIP_GLOBAL_ERROR, true)
    });
  }

  getFlashcardDeck(id: string): Observable<FlashcardDeck> {
    return this.http.get<FlashcardDeck>(this.url(`/api/flashcards/${id}`), { withCredentials: true });
  }

  generateFlashcardDeck(topic: string, count: number = 5): Observable<{ message: string; deck_id: string }> {
    return this.http.post<{ message: string; deck_id: string }>(this.url('/api/flashcards/generate'), { topic, count }, {
      withCredentials: true,
      context: new HttpContext().set(SKIP_GLOBAL_ERROR, true)
    });
  }

  rateFlashcard(deckId: string, cardId: string, rating: 'known' | 'unknown'): Observable<{ success: boolean }> {
    return this.http.post<{ success: boolean }>(this.url(`/api/flashcards/${deckId}/cards/${cardId}/rating`), { rating }, { withCredentials: true });
  }

  // --- Quizzes ---
  getQuizSessions(): Observable<QuizSession[]> {
    return this.http.get<QuizSession[]>(this.url('/api/quiz/sessions'), { withCredentials: true });
  }

  getQuizSession(id: string): Observable<QuizSession & { questions: any[] }> {
    return this.http.get<QuizSession & { questions: any[] }>(this.url(`/api/quiz/${id}`), { withCredentials: true });
  }

  generateQuiz(topic: string, count: number = 5): Observable<{ message: string; session_id: string }> {
    return this.http.post<{ message: string; session_id: string }>(this.url('/api/quiz/generate'), { topic, count }, { withCredentials: true });
  }

  submitQuizScore(sessionId: string, answers: number[]): Observable<{ message: string; score: number }> {
    return this.http.post<{ message: string; score: number }>(this.url(`/api/quiz/${sessionId}/score`), { answers }, { withCredentials: true });
  }

  // --- Study Tools ---
  explainTopic(topic: string): Observable<{ result: string }> {
    return this.http.post<{ result: string }>(this.url('/api/tools/explain'), { topic }, { withCredentials: true });
  }

  summarizeText(text: string): Observable<{ result: string }> {
    return this.http.post<{ result: string }>(this.url('/api/tools/summarize'), { text }, { withCredentials: true });
  }

  generateMnemonics(topic: string): Observable<{ result: string }> {
    return this.http.post<{ result: string }>(this.url('/api/tools/mnemonics'), { topic }, { withCredentials: true });
  }

  // --- Profile & Analytics ---
  getProfile(): Observable<UserProfile> {
    return this.http.get<UserProfile>(this.url('/api/profile/me'), { withCredentials: true });
  }

  updateProfile(profileData: Partial<UserProfile>): Observable<{ message: string; profile: UserProfile }> {
    return this.http.put<{ message: string; profile: UserProfile }>(this.url('/api/profile/me'), profileData, { withCredentials: true });
  }

  getStudyStats(): Observable<StudyStats> {
    return this.http.get<StudyStats>(this.url('/api/analytics/study-stats'), {
      withCredentials: true,
      context: new HttpContext().set(SKIP_GLOBAL_ERROR, true)
    });
  }

  getDashboardStats(): Observable<DashboardStats> {
    return this.http.get<DashboardStats>(this.url('/api/dashboard/stats'), {
      withCredentials: true,
      context: new HttpContext().set(SKIP_GLOBAL_ERROR, true)
    });
  }

  // --- Document Upload ---
  uploadDocument(file: File): Observable<{ chunks_indexed: number; filename: string; message: string }> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<{ chunks_indexed: number; filename: string; message: string }>(
      this.url('/api/documents/upload'),
      formData,
      { withCredentials: true }
    );
  }

  uploadDocumentWithProgress(file: File): Observable<HttpEvent<UploadDocumentResponse>> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<UploadDocumentResponse>(this.url('/api/documents/upload'), formData, {
      withCredentials: true,
      observe: 'events',
      reportProgress: true
    });
  }

  private readCookie(name: string): string | null {
    if (typeof document === 'undefined') {
      return null;
    }
    const match = document.cookie.split('; ').find((entry) => entry.startsWith(`${name}=`));
    return match ? decodeURIComponent(match.split('=').slice(1).join('=')) : null;
  }

  private url(path: string): string {
    return `${this.runtimeConfig.apiBaseUrl}${path}`;
  }
}
