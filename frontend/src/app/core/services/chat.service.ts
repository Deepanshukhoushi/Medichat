import { Injectable, inject, signal } from '@angular/core';
import { finalize, map, Observable, tap } from 'rxjs';

import { BackendApiService, Conversation, TopicMemoryResponse } from './backend-api.service';
import { ChatCitation, ChatMessage, ConversationSummary } from '../../shared/models/chat.model';

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
  private abortController: AbortController | null = null;

  readonly messages = this.messagesSignal.asReadonly();
  readonly conversationHistory = this.conversationsSignal.asReadonly();
  readonly isStreaming = this.isStreamingSignal.asReadonly();
  readonly isHistoryLoading = this.isHistoryLoadingSignal.asReadonly();
  readonly historyLoadError = this.historyLoadErrorSignal.asReadonly();
  readonly backendReady = this.backendReadySignal.asReadonly();
  readonly topicMemory = this.topicMemorySignal.asReadonly();
  readonly sessionSummary = this.sessionSummarySignal.asReadonly();
  readonly activeConversationId = this.activeConversationIdSignal.asReadonly();

  bootstrap(): void {
    this.backendApi.getHealth().subscribe({
      next: (response) => {
        this.backendReadySignal.set(response.status === 'healthy' || response.status === 'degraded');
      },
      error: () => {
        this.backendReadySignal.set(false);
      }
    });
  }

  createConversation(prompt: string): void {
    const trimmedPrompt = prompt.trim();
    if (!trimmedPrompt) {
      return;
    }

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: trimmedPrompt,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    const pendingMessage = this.pendingMessage();
    this.messagesSignal.update((messages) => [...messages, userMessage, pendingMessage]);
    this.conversationsSignal.update((conversations) => [
      {
        id: `session-${Date.now()}`,
        title: this.trimPrompt(trimmedPrompt),
        topic: 'Current session',
        updatedAt: 'Just now'
      },
      ...conversations.slice(0, 4)
    ]);

    this.isStreamingSignal.set(true);
    this.abortController = new AbortController();
    let currentAnswer = '';

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
          if (chunk.token) {
            currentAnswer += chunk.token;
            this.replacePendingMessage(pendingMessage.id, currentAnswer, trimmedPrompt);
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
    this.messagesSignal.update((messages) =>
      messages.map((message) => (message.id === messageId ? { ...message, liked } : message))
    );
  }

  removeMessage(messageId: string): void {
    this.messagesSignal.update((messages) => messages.filter((message) => message.id !== messageId));
  }

  regenerateLastResponse(): void {
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
          if (chunk.token) {
            currentAnswer += chunk.token;
            this.replacePendingMessage(lastAssistantMessage.id, currentAnswer, trimmedPrompt);
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
          id: `msg-${conversationId}-${i}`,
          role: msg.role as 'user' | 'assistant',
          content: msg.content,
          timestamp: ''
        }));
        this.messagesSignal.set(messages);
        this.refreshMemory();
        this.refreshConversationHistory().subscribe({ error: () => undefined });
      },
      error: (err) => {
        this.messagesSignal.set([]);
        this.historyLoadErrorSignal.set(
          err?.error?.error || 'Failed to load the conversation history. Please try again.'
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
          const exists = finalHistory.find(c => c.id === currentId);
          if (!exists) {
            const currentInSignal = this.conversationsSignal().find(c => c.id === currentId || c.id.startsWith('session-'));
            if (currentInSignal) {
              finalHistory.unshift({
                ...currentInSignal,
                id: currentId
              });
            }
          }
        }
        this.conversationsSignal.set(finalHistory);
      })
    );
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
      id: `assistant-${Date.now()}`,
      role: 'assistant',
      content: '',
      timestamp: 'Thinking...'
    };
  }

  private replacePendingMessage(messageId: string, content: string, prompt: string): void {
    const assistantMessage: ChatMessage = {
      id: messageId,
      role: 'assistant',
      content,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      citations: this.buildCitations(content, prompt)
    };

    this.messagesSignal.update((messages) =>
      messages.map((message) => (message.id === messageId ? assistantMessage : message))
    );
  }

  private buildCitations(_content: string, _prompt: string): ChatCitation[] {
    return [];
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
