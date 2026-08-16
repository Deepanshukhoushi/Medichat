import { ChangeDetectionStrategy, Component, input, output, signal } from '@angular/core';
import { Clipboard } from '@angular/cdk/clipboard';
import { LucideDynamicIcon } from '@lucide/angular';
import { MarkdownComponent } from 'ngx-markdown';
import { ToastrService } from 'ngx-toastr';


import { ChatMessage } from '../../../../shared/models/chat.model';
import { appIcons } from '../../../../shared/icons/lucide-icons';

@Component({
  selector: 'mc-chat-message',
  standalone: true,
  imports: [MarkdownComponent, LucideDynamicIcon],
  templateUrl: './chat-message.component.html',
  styleUrl: './chat-message.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class ChatMessageComponent {
  readonly message = input.required<ChatMessage>();
  readonly isLast = input<boolean>(false);
  readonly isStreaming = input<boolean>(false);
  
  readonly deleted = output<string>();
  readonly reacted = output<boolean>();
  readonly regenerated = output<void>();
  readonly editRequested = output<string>();
  
  protected readonly icons = appIcons;

  /** Controls the "More" dropdown — click-toggled, touch-safe. */
  protected readonly showMoreMenu = signal(false);

  constructor(
    private readonly clipboard: Clipboard,
    private readonly toastr: ToastrService
  ) {}

  protected copy(): void {
    this.clipboard.copy(this.message().content);
    this.toastr.success('Answer copied to clipboard');
  }

  protected deleteMessage(): void {
    this.deleted.emit(this.message().id);
  }

  protected react(helpful: boolean): void {
    this.reacted.emit(helpful);
  }

  protected regenerate(): void {
    this.closeMoreMenu();
    this.regenerated.emit();
  }

  protected editMessage(): void {
    this.editRequested.emit(this.message().id);
  }

  // ── More-menu toggle ──────────────────────────────────────────────────────

  protected toggleMoreMenu(): void {
    this.showMoreMenu.update(v => !v);
  }

  protected closeMoreMenu(): void {
    this.showMoreMenu.set(false);
  }

  // ── Coming-soon stubs (Flashcards, Quiz, Notes, Bookmark, Share) ──────────

  protected openFlashcards(): void {
    this.comingSoon('Flashcards');
  }

  protected createQuiz(): void {
    this.comingSoon('Quiz');
  }

  protected createNotes(): void {
    this.comingSoon('Notes');
  }

  protected bookmarkMessage(): void {
    this.closeMoreMenu();
    this.comingSoon('Bookmark');
  }

  protected shareMessage(): void {
    this.closeMoreMenu();
    this.comingSoon('Share');
  }

  private comingSoon(feature: string): void {
    this.toastr.info(`${feature} is coming soon!`, '', { timeOut: 2500 });
  }
}
