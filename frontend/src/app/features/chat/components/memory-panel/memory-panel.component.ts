import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { LucideDynamicIcon } from '@lucide/angular';

import { ChatService } from '../../../../core/services/chat.service';
import { appIcons } from '../../../../shared/icons/lucide-icons';

@Component({
  selector: 'mc-memory-panel',
  standalone: true,
  imports: [LucideDynamicIcon],
  templateUrl: './memory-panel.component.html',
  styleUrl: './memory-panel.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class MemoryPanelComponent {
  protected readonly chatService = inject(ChatService);
  protected readonly icons = appIcons;
  protected readonly summaryExpanded = signal(false);

  protected toggleSummary(): void {
    this.summaryExpanded.update((v) => !v);
  }
}
