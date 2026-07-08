import { ChangeDetectionStrategy, Component, ElementRef, HostListener, ViewChild, computed, effect, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { LucideDynamicIcon } from '@lucide/angular';

import { ChatService } from '../../../core/services/chat.service';
import { BackendApiService, DashboardStats } from '../../../core/services/backend-api.service';
import { ProfileService } from '../../../core/services/profile.service';
import { appIcons } from '../../../shared/icons/lucide-icons';
import { ChatComposerComponent } from '../components/chat-composer/chat-composer.component';
import { ChatMessageComponent } from '../components/chat-message/chat-message.component';
import { MemoryPanelComponent } from '../components/memory-panel/memory-panel.component';

@Component({
  selector: 'mc-chat-page',
  standalone: true,
  imports: [FormsModule, LucideDynamicIcon, ChatComposerComponent, ChatMessageComponent, MemoryPanelComponent],
  templateUrl: './chat-page.component.html',
  styleUrl: './chat-page.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class ChatPageComponent {
  @ViewChild('scrollContainer') private scrollContainer!: ElementRef<HTMLDivElement>;
  @ViewChild(ChatComposerComponent) private composer!: ChatComposerComponent;
  
  protected readonly icons = appIcons;
  protected readonly chatService = inject(ChatService);
  protected readonly backendApi = inject(BackendApiService);
  private readonly profileService = inject(ProfileService);
  private readonly router = inject(Router);
  
  protected readonly isNearBottom = signal(true);
  protected readonly composerFocused = signal(false);
  protected readonly showShortcuts = signal(false);
  /** Read from shared ProfileService cache — no extra network call. */
  protected readonly userProfile = this.profileService.profile;
  protected readonly dashboardStats = signal<DashboardStats | null>(null);

  protected readonly greeting = computed(() => {
    const hour = new Date().getHours();
    let timeGreeting = 'Good Evening';
    if (hour < 12) timeGreeting = 'Good Morning';
    else if (hour < 18) timeGreeting = 'Good Afternoon';

    const name = this.userProfile()?.display_name || 'Medical Student';
    return `${timeGreeting}, ${name}`;
  });

  protected readonly starterPrompts = computed(() => {
    const topicMemory = this.chatService.topicMemory();
    const topic = topicMemory.current_topic?.trim();
    const related = topicMemory.related_topics.filter(Boolean);

    if (topic) {
      const relatedTopic = related[0] ?? topic;
      return [
        { label: `Explain ${topic}`, prompt: `Explain ${topic} in simple terms.` },
        { label: `MCQs on ${topic}`, prompt: `Give me 5 MCQs on ${topic}.` },
        { label: `Compare with ${relatedTopic}`, prompt: `Compare ${topic} with ${relatedTopic}.` },
        { label: `Viva on ${topic}`, prompt: `Ask me viva questions on ${topic}.` }
      ];
    }

    return [
      { label: 'MCQs on cardiac muscle', prompt: 'Generate 10 MCQs on cardiac muscle tissue.' },
      { label: 'Cranial nerve flashcards', prompt: 'Create 5 flashcards on the cranial nerves.' },
      { label: 'Renal physiology summary', prompt: 'Summarize the latest chapter on renal physiology.' },
      { label: 'Liver anatomy viva', prompt: 'Ask me 3 viva questions on liver anatomy.' }
    ];
  });

  constructor() {

    effect(() => {
      const messages = this.chatService.messages();
      
      // Auto-clear composer when starting a new chat (messages reset to 0)
      if (messages.length === 0 && this.composer) {
        this.composer.setPrompt('');
      }

      this.chatService.isStreaming();
      setTimeout(() => {
        if (this.isNearBottom() && !this.chatService.isStreaming()) {
          this.scrollToBottom();
        }
        this.updateScrollState();
      }, 50);
    });
  }

  protected onScroll(): void {
    this.updateScrollState();
  }

  protected scrollToBottom(): void {
    try {
      const el = this.scrollContainer.nativeElement;
      el.scrollTop = el.scrollHeight;
      this.isNearBottom.set(true);
    } catch (err) {
      // Ignore errors if element isn't ready
    }
  }

  private updateScrollState(): void {
    try {
      const el = this.scrollContainer.nativeElement;
      const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
      this.isNearBottom.set(distanceFromBottom < 120);
    } catch {
      this.isNearBottom.set(true);
    }
  }

  protected startConversation(prompt: string): void {
    this.chatService.createConversation(prompt);
    setTimeout(() => this.scrollToBottom(), 50);
  }

  protected insertPrompt(prompt: string): void {
    if (this.composer) {
      this.composer.setPrompt(prompt);
    }
  }

  protected regenerate(): void {
    this.chatService.regenerateLastResponse();
    setTimeout(() => this.scrollToBottom(), 50);
  }

  protected react(messageId: string, liked: boolean): void {
    this.chatService.rateMessage(messageId, liked);
  }

  protected removeMessage(messageId: string): void {
    this.chatService.removeMessage(messageId);
  }

  protected editMessage(messageId: string): void {
    const msg = this.chatService.messages().find((m) => m.id === messageId);
    if (msg && msg.role === 'user') {
      this.insertPrompt(msg.content);
      this.chatService.removeMessage(messageId);
    }
  }

  protected resumeLastChat(): void {
    const stats = this.dashboardStats();
    const lastConversation = stats?.recent_conversations?.[0];
    if (lastConversation?.id) {
      this.chatService.loadHistory(lastConversation.id);
    }
  }

  protected stopGeneration(): void {
    this.chatService.stopGeneration();
  }

  @HostListener('document:keydown.escape')
  onDocumentEscape(): void {
    if (this.showShortcuts()) {
      this.closeShortcuts();
    }
  }

  @HostListener('document:keydown', ['$event'])
  onDocumentKeydown(event: KeyboardEvent): void {
    if (event.key === '/' && !this.composerFocused() && !this.showShortcuts()) {
      // Don't focus if user is already typing in an input somewhere else
      const target = event.target as HTMLElement;
      if (target.tagName !== 'INPUT' && target.tagName !== 'TEXTAREA' && !target.isContentEditable) {
        event.preventDefault();
        // Since we have two composers (hero vs footer), we focus the visible one
        const composer = document.querySelector('textarea') as HTMLTextAreaElement;
        if (composer) {
          composer.focus();
        }
      }
    }
  }

  protected retryHistoryLoad(): void {
    const conversationId = this.chatService.activeConversationId();
    if (!conversationId) {
      return;
    }
    this.chatService.loadHistory(conversationId);
  }

  protected openShortcuts(): void {
    this.showShortcuts.set(true);
  }

  protected closeShortcuts(): void {
    this.showShortcuts.set(false);
  }

  protected onComposerFocusChange(isFocused: boolean): void {
    this.composerFocused.set(isFocused);
  }
}
