import { Injectable, inject, signal } from '@angular/core';
import { finalize, map, Observable, tap } from 'rxjs';

import { BackendApiService, Conversation, TopicMemoryResponse } from './backend-api.service';
import { ChatCitation, ChatMessage, ConversationSummary } from '../../shared/models/chat.model';
import { extractErrorMessage } from '../utils/extract-error-message';

@Injectable({ providedIn: 'root' })
export class ChatService {
  private readonly backendApi = inject(BackendApiService);
  private readonly messagesSignal = signal<ChatMessage[]>([]);
  private readonly conversationsSignal = signal<ConversationSummary[]>([]);
  private readonly isStreamingSignal = signal(false);
  private readonly backendReadySignal = signal(false);
  private readonly isHistoryLoadingSignal = signal(false);
  private readonly historyLoadErrorSignal = signal<string | null>(null);
  private readonly topicMemorySignal = signal<TopicMemoryResponse>({ current_topic: null, related_topics: [] });
  private readonly sessionSummarySignal = signal<string | null>(null);
  private readonly activeConversationIdSignal = signal<string | null>(null);
  /** True when /health reports supabase: 'disabled' — database is not connected. */
  private readonly persistenceDisabledSignal = signal(false);
  private abortController: AbortController | null = null;
  // Tracks the optimistic placeholder conversation ID inserted into the sidebar
  // before the backend confirms the real ID. Reset once the real ID arrives.
  private pendingConversationId: string | null = null;

  readonly messages = this.messagesSignal.asReadonly();
  readonly conversationHistory = this.conversationsSignal.asReadonly();
  readonly isStreaming = this.isStreamingSignal.asReadonly();
  readonly isHistoryLoading = this.isHistoryLoadingSignal.asReadonly();
  readonly historyLoadError = this.historyLoadErrorSignal.asReadonly();
  readonly backendReady = this.backendReadySignal.asReadonly();
  readonly topicMemory = this.topicMemorySignal.asReadonly();
  readonly sessionSummary = this.sessionSummarySignal.asReadonly();
  readonly activeConversationId = this.activeConversationIdSignal.asReadonly();
  /** Exposed so layouts can render a persistence-disabled warning banner. */
  readonly persistenceDisabled = this.persistenceDisabledSignal.asReadonly();

  bootstrap(): void {
    this.backendApi.getHealth().subscribe({
      next: (response) => {
        this.backendReadySignal.set(response.status === 'healthy' || response.status === 'degraded');
        // Surface persistence-disabled state so the UI can warn users.
        const supabaseCheck = (response as any).checks?.supabase;
        this.persistenceDisabledSignal.set(supabaseCheck === 'disabled');
      },
      error: () => {
        this.backendReadySignal.set(false);
      }
    });
  }

  createConversation(prompt: string): void {
    // Guard: if a stream is already in flight, abort it cleanly before starting
    // a new one. Without this, a double-fire (IME enter, fast retry, future UI
    // entry points) would orphan the first AbortController, making it impossible
    // to cancel and causing finalize() races on shared signals.
    if (this.isStreamingSignal()) {
      this.stopGeneration();
    }

    const trimmedPrompt = prompt.trim();
    if (!trimmedPrompt) {
      return;
    }

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: trimmedPrompt,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    const pendingMessage = this.pendingMessage();
    this.messagesSignal.update((messages) => [...messages, userMessage, pendingMessage]);

    // Insert an optimistic sidebar entry; record its ID so refreshConversationHistory
    // can swap it for the real backend ID without relying on string-prefix guessing.
    this.pendingConversationId = crypto.randomUUID();
    this.conversationsSignal.update((conversations) => [
      {
        id: this.pendingConversationId!,
        title: this.trimPrompt(trimmedPrompt),
        topic: 'Current session',
        updatedAt: 'Just now'
      },
      ...conversations.slice(0, 4)
    ]);

    this.isStreamingSignal.set(true);
    this.abortController = new AbortController();
    let currentAnswer = '';
    let currentCitations: ChatCitation[] = [];

    this.backendApi
      .streamChatMessage(trimmedPrompt, this.activeConversationId(), this.abortController.signal)
      .pipe(
        finalize(() => {
          this.isStreamingSignal.set(false);
          this.abortController = null;
          this.refreshMemory();
          this.refreshConversationHistory().subscribe({ error: () => undefined });
        })
      )
      .subscribe({
        next: (chunk) => {
          if (chunk.conversation_id && !this.activeConversationId()) {
            this.activeConversationIdSignal.set(chunk.conversation_id);
          }
          if (chunk.sources) {
            currentCitations = chunk.sources;
          }
          if (chunk.token) {
            currentAnswer += chunk.token;
            this.replacePendingMessage(pendingMessage.id, currentAnswer, trimmedPrompt, currentCitations);
          }
          if (chunk.message_id) {
            this.replacePendingMessage(pendingMessage.id, currentAnswer, trimmedPrompt, currentCitations, chunk.message_id);
            pendingMessage.id = chunk.message_id;
          }
          if (chunk.error) {
            this.replacePendingMessage(
              pendingMessage.id,
              chunk.error,
              trimmedPrompt
            );
          }
        },
        error: () =>
          this.replacePendingMessage(
            pendingMessage.id,
            'I could not reach the backend right now. Please try again after the API is available.',
            trimmedPrompt
          )
      });
  }

  stopGeneration(): void {
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
    this.isStreamingSignal.set(false);
  }

  rateMessage(messageId: string, liked: boolean): void {
    const cid = this.activeConversationId();
    if (cid && !messageId.startsWith('msg-')) {
      this.backendApi.rateMessage(cid, messageId, liked).subscribe({ error: () => undefined });
    }
    this.messagesSignal.update((messages) =>
      messages.map((message) => (message.id === messageId ? { ...message, liked } : message))
    );
  }

  removeMessage(messageId: string): void {
    const cid = this.activeConversationId();
    if (cid && !messageId.startsWith('msg-')) {
      this.backendApi.deleteMessage(cid, messageId).subscribe({ error: () => undefined });
    }
    this.messagesSignal.update((messages) => messages.filter((message) => message.id !== messageId));
  }

  regenerateLastResponse(): void {
    // Guard: same orphaned-AbortController protection as createConversation.
    if (this.isStreamingSignal()) {
      this.stopGeneration();
    }

    const messages = this.messagesSignal();
    const lastUserMessage = [...messages].reverse().find((message) => message.role === 'user');
    const lastAssistantMessage = [...messages].reverse().find((message) => message.role === 'assistant');
    if (!lastUserMessage || !lastAssistantMessage) {
      return;
    }

    this.isStreamingSignal.set(true);
    this.abortController = new AbortController();
    const trimmedPrompt = lastUserMessage.content;
    let currentAnswer = '';
    let currentCitations: ChatCitation[] = [];

    this.replacePendingMessage(
      lastAssistantMessage.id,
      '',
      trimmedPrompt
    );

    this.backendApi
      .streamChatMessage(trimmedPrompt, this.activeConversationId(), this.abortController.signal, true)
      .pipe(
        finalize(() => {
          this.isStreamingSignal.set(false);
          this.abortController = null;
          this.refreshMemory();
        })
      )
      .subscribe({
        next: (chunk) => {
          if (chunk.sources) {
            currentCitations = chunk.sources;
          }
          if (chunk.token) {
            currentAnswer += chunk.token;
            this.replacePendingMessage(lastAssistantMessage.id, currentAnswer, trimmedPrompt, currentCitations);
          }
          if (chunk.message_id) {
            this.replacePendingMessage(lastAssistantMessage.id, currentAnswer, trimmedPrompt, currentCitations, chunk.message_id);
            lastAssistantMessage.id = chunk.message_id;
          }
          if (chunk.error) {
            this.replacePendingMessage(lastAssistantMessage.id, chunk.error, trimmedPrompt);
          }
        },
        error: () =>
          this.replacePendingMessage(
            lastAssistantMessage.id,
            'I could not reach the backend right now. Please try again after the API is available.',
            trimmedPrompt
          )
      });
  }

  resetSession(): void {
    this.messagesSignal.set([]);
    this.activeConversationIdSignal.set(null);
    this.historyLoadErrorSignal.set(null);
    this.topicMemorySignal.set({ current_topic: null, related_topics: [] });
    this.sessionSummarySignal.set(null);
  }

  loadHistory(conversationId: string): void {
    this.isHistoryLoadingSignal.set(true);
    this.historyLoadErrorSignal.set(null);
    this.activeConversationIdSignal.set(conversationId);

    this.backendApi.getConversationMessages(conversationId).pipe(
      finalize(() => this.isHistoryLoadingSignal.set(false))
    ).subscribe({
      next: (data) => {
        const messages: ChatMessage[] = data.map((msg, i) => ({
          id: msg.id || `msg-${conversationId}-${i}`,
          role: msg.role as 'user' | 'assistant',
          content: msg.content,
          timestamp: '',
          liked: msg.liked
        }));
        this.messagesSignal.set(messages);
        this.refreshMemory();
        this.refreshConversationHistory().subscribe({ error: () => undefined });
      },
      error: (err) => {
        this.messagesSignal.set([]);
        this.historyLoadErrorSignal.set(
          extractErrorMessage(err, 'Failed to load the conversation history. Please try again.')
        );
      }
    });
  }

  refreshConversationHistory(): Observable<ConversationSummary[]> {
    return this.backendApi.getConversations().pipe(
      map((conversations) => conversations.map((conversation) => this.toConversationSummary(conversation))),
      tap((conversationHistory) => {
        const currentId = this.activeConversationId();
        let finalHistory = [...conversationHistory];

        if (currentId) {
          const existsInBackend = finalHistory.find(c => c.id === currentId);
          if (!existsInBackend && this.pendingConversationId) {
            // Swap the optimistic placeholder for the confirmed backend entry,
            // matched by the explicit pendingConversationId we set in createConversation.
            const placeholder = this.conversationsSignal().find(c => c.id === this.pendingConversationId);
            if (placeholder) {
              finalHistory.unshift({ ...placeholder, id: currentId });
            }
            this.pendingConversationId = null;
          }
        }
        this.conversationsSignal.set(finalHistory);
      })
    );
  }

  readonly isLoadingMoreHistorySignal = signal(false);
  readonly hasMoreHistorySignal = signal(true);

  loadMoreHistory(): void {
    if (this.isLoadingMoreHistorySignal() || !this.hasMoreHistorySignal()) {
      return;
    }
    
    this.isLoadingMoreHistorySignal.set(true);
    const currentHistory = this.conversationsSignal();
    const currentOffset = currentHistory.length;
    
    this.backendApi.getConversations(30, currentOffset).subscribe({
      next: (conversations) => {
        if (conversations.length < 30) {
          this.hasMoreHistorySignal.set(false);
        }
        
        const summaries = conversations.map(c => this.toConversationSummary(c));
        this.conversationsSignal.set([...currentHistory, ...summaries]);
        this.isLoadingMoreHistorySignal.set(false);
      },
      error: (err) => {
        console.error('Failed to load more history', err);
        this.isLoadingMoreHistorySignal.set(false);
      }
    });
  }

  private refreshMemory(): void {
    this.backendApi.getTopicMemory().subscribe({
      next: (data) => this.topicMemorySignal.set(data),
      error: () => { /* silent - memory panel stays empty on error */ }
    });
    this.backendApi.getSessionSummary().subscribe({
      next: (data) => this.sessionSummarySignal.set(data.summary),
      error: () => { /* silent */ }
    });
  }

  private pendingMessage(): ChatMessage {
    return {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: '',
      timestamp: 'Thinking...'
    };
  }

  private replacePendingMessage(id: string, content: string, originalPrompt: string, citations?: ChatCitation[], newId?: string): void {
    this.messagesSignal.update((messages) =>
      messages.map((message) => {
        if (message.id === id) {
          return {
            ...message,
            content,
            citations: citations || message.citations || [],
            id: newId || message.id,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
          };
        }
        return message;
      })
    );
  }


  private toConversationSummary(conversation: Conversation): ConversationSummary {
    return {
      id: conversation.id,
      title: conversation.title,
      topic: 'Recent conversation',
      updatedAt: conversation.updated_at ?? conversation.created_at
    };
  }

  private trimPrompt(prompt: string): string {
    return prompt.length > 42 ? `${prompt.slice(0, 42)}...` : prompt;
  }
}
