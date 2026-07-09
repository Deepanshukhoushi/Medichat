import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterLink } from '@angular/router';
import { LucideDynamicIcon } from '@lucide/angular';
import packageJson from '../../../../../package.json';

import { ThemeService } from '../../../core/services/theme.service';
import { ThemePreference } from '../../../shared/models/theme.model';
import { GlassCardComponent } from '../../../shared/components/glass-card/glass-card.component';
import { SectionHeadingComponent } from '../../../shared/components/section-heading/section-heading.component';
import { appIcons } from '../../../shared/icons/lucide-icons';
import { ChatService } from '../../../core/services/chat.service';
import { BackendApiService } from '../../../core/services/backend-api.service';

@Component({
  selector: 'mc-settings-page',
  standalone: true,
  imports: [CommonModule, RouterLink, LucideDynamicIcon, GlassCardComponent, SectionHeadingComponent],
  templateUrl: './settings-page.component.html',
  styleUrl: './settings-page.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'block h-full min-h-0' }
})
export class SettingsPageComponent {
  readonly themeService = inject(ThemeService);
  readonly icons = appIcons;
  readonly version = packageJson.version;

  private readonly chatService = inject(ChatService);
  private readonly backendApi = inject(BackendApiService);
  private readonly router = inject(Router);

  readonly clearHistoryDone = signal(false);
  readonly clearHistoryConfirm = signal(false);
  readonly isDeletingConversations = signal(false);

  setTheme(theme: ThemePreference): void {
    this.themeService.setPreference(theme);
  }

  clearCurrentSession(): void {
    this.chatService.resetSession();
    this.clearHistoryDone.set(true);
    setTimeout(() => this.clearHistoryDone.set(false), 3000);
  }

  promptClearAllConversations(): void {
    this.clearHistoryConfirm.set(true);
  }

  cancelClearAll(): void {
    this.clearHistoryConfirm.set(false);
  }

  confirmClearAllConversations(): void {
    this.clearHistoryConfirm.set(false);
    this.isDeletingConversations.set(true);

    // Delete each conversation in the sidebar list
    const conversations = this.chatService.conversationHistory();
    if (conversations.length === 0) {
      this.chatService.resetSession();
      this.isDeletingConversations.set(false);
      return;
    }

    let remaining = conversations.length;
    for (const conv of conversations) {
      this.backendApi.deleteConversation(conv.id).subscribe({
        next: () => {
          remaining--;
          if (remaining === 0) {
            this.chatService.resetSession();
            this.chatService.refreshConversationHistory().subscribe({ error: () => undefined });
            this.isDeletingConversations.set(false);
          }
        },
        error: () => {
          remaining--;
          if (remaining === 0) {
            this.chatService.resetSession();
            this.isDeletingConversations.set(false);
          }
        }
      });
    }
  }
}
