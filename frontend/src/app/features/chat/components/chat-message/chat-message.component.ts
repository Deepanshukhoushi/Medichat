import { ChangeDetectionStrategy, Component, input, output } from '@angular/core';
import { Clipboard } from '@angular/cdk/clipboard';
import { LucideDynamicIcon } from '@lucide/angular';
import { MarkdownComponent } from 'ngx-markdown';
import { ToastrService } from 'ngx-toastr';

import { ChatMessage } from '../../../../shared/models/chat.model';
import { appIcons } from '../../../../shared/icons/lucide-icons';
import { MedicalOrbComponent } from '../../../../shared/components/medical-orb/medical-orb.component';

@Component({
  selector: 'mc-chat-message',
  standalone: true,
  imports: [MarkdownComponent, MedicalOrbComponent, LucideDynamicIcon],
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
  
  protected readonly icons = appIcons;

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
    this.regenerated.emit();
  }
}
