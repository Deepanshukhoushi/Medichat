import { ChangeDetectionStrategy, Component, computed, inject, OnInit } from '@angular/core';
import { RouterLink, RouterOutlet } from '@angular/router';
import { LucideDynamicIcon } from '@lucide/angular';

import { AppLogoComponent } from '../../shared/components/app-logo/app-logo.component';
import { ThemeToggleComponent } from '../../shared/components/theme-toggle/theme-toggle.component';
import { ProfileService } from '../../core/services/profile.service';
import { appIcons } from '../../shared/icons/lucide-icons';

@Component({
  selector: 'mc-marketing-layout',
  standalone: true,
  imports: [RouterLink, RouterOutlet, AppLogoComponent, ThemeToggleComponent, LucideDynamicIcon],
  templateUrl: './marketing-layout.component.html',
  styleUrl: './marketing-layout.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class MarketingLayoutComponent implements OnInit {
  private readonly profileService = inject(ProfileService);
  protected readonly icons = appIcons;

  /** Derived from the shared profile cache — no extra network call. */
  protected readonly isLoggedIn = computed(() => {
    const profile = this.profileService.profile();
    return profile != null && !profile.user_id.startsWith('guest_');
  });

  ngOnInit() {
    // Warm up the cache so the signal is populated; the guard may already have done this.
    this.profileService.profile$.subscribe({ error: () => undefined });
  }
}
