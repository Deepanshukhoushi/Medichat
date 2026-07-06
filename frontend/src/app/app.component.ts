import { ChangeDetectionStrategy, Component, effect, inject } from '@angular/core';
import { RouterOutlet } from '@angular/router';

import { ChatService } from './core/services/chat.service';
import { ThemeService } from './core/services/theme.service';
import { ToastrService } from 'ngx-toastr';

@Component({
  selector: 'mc-root',
  standalone: true,
  imports: [RouterOutlet],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class AppComponent {
  private readonly themeService = inject(ThemeService);
  private readonly chatService = inject(ChatService);
  private readonly toastr = inject(ToastrService);

  constructor() {
    this.chatService.bootstrap();
    effect(() => {
      this.themeService.activeTheme();
    });

    if (typeof window !== 'undefined') {
      window.addEventListener('offline', () => {
        this.toastr.warning('You are currently offline. Some features may not be available.', 'Offline Mode', {
          timeOut: 0,
          extendedTimeOut: 0,
          closeButton: true
        });
      });

      window.addEventListener('online', () => {
        this.toastr.clear();
        this.toastr.success('You are back online!', 'Connection Restored');
      });
    }
  }
}
