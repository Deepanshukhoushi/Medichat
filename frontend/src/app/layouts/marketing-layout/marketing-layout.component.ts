import { ChangeDetectionStrategy, Component, inject, OnInit, signal } from '@angular/core';
import { RouterLink, RouterOutlet } from '@angular/router';

import { AppLogoComponent } from '../../shared/components/app-logo/app-logo.component';
import { ThemeToggleComponent } from '../../shared/components/theme-toggle/theme-toggle.component';
import { BackendApiService } from '../../core/services/backend-api.service';

@Component({
  selector: 'mc-marketing-layout',
  standalone: true,
  imports: [RouterLink, RouterOutlet, AppLogoComponent, ThemeToggleComponent],
  templateUrl: './marketing-layout.component.html',
  styleUrl: './marketing-layout.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class MarketingLayoutComponent implements OnInit {
  private readonly api = inject(BackendApiService);
  protected readonly isLoggedIn = signal(false);

  ngOnInit() {
    this.api.getProfile().subscribe({
      next: (profile) => {
        if (profile) {
          this.isLoggedIn.set(true);
        }
      },
      error: () => {
        this.isLoggedIn.set(false);
      }
    });
  }
}
