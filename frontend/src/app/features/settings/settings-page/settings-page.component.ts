import { ChangeDetectionStrategy, Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { LucideDynamicIcon } from '@lucide/angular';

import { ThemeService } from '../../../core/services/theme.service';
import { ThemePreference } from '../../../shared/models/theme.model';
import { GlassCardComponent } from '../../../shared/components/glass-card/glass-card.component';
import { SectionHeadingComponent } from '../../../shared/components/section-heading/section-heading.component';
import { appIcons } from '../../../shared/icons/lucide-icons';

@Component({
  selector: 'mc-settings-page',
  standalone: true,
  imports: [CommonModule, RouterLink, LucideDynamicIcon, GlassCardComponent, SectionHeadingComponent],
  templateUrl: './settings-page.component.html',
  styleUrl: './settings-page.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class SettingsPageComponent {
  readonly themeService = inject(ThemeService);
  readonly icons = appIcons;

  setTheme(theme: ThemePreference): void {
    this.themeService.setPreference(theme);
  }
}
