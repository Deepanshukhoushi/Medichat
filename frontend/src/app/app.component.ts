import { ChangeDetectionStrategy, Component, effect, inject } from '@angular/core';
import { NavigationEnd, Router, RouterOutlet } from '@angular/router';
import { filter, take } from 'rxjs';

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
  private readonly router = inject(Router);

  constructor() {
    // Only wake up the backend when the user actually navigates into the
    // authenticated app shell (/app/*). The landing page and auth pages
    // never need the backend at bootstrap time, so we avoid triggering
    // Render cold-start 504s on public pages entirely.
    this.router.events.pipe(
      filter((e): e is NavigationEnd => e instanceof NavigationEnd),
      take(1),
    ).subscribe((e) => {
      if (e.urlAfterRedirects.startsWith('/app')) {
        this.chatService.bootstrap();
      }
    });

    effect(() => {
      this.themeService.activeTheme();
    });

    if (typeof window !== 'undefined') {
      window.addEventListener('offline', () => {
        this.toastr.warning("You're offline. Some AI features are temporarily unavailable.", 'Offline', {
          timeOut: 0,
          extendedTimeOut: 0,
          closeButton: true
        });
      });

      window.addEventListener('online', () => {
        this.toastr.clear();
        this.toastr.success('Your connection has been restored.', 'Online');
      });
    }
  }
}
