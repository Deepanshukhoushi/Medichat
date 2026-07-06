import { ChangeDetectionStrategy, Component, inject } from '@angular/core';
import { LucideDynamicIcon } from '@lucide/angular';

import { ThemeService } from '../../../core/services/theme.service';
import { appIcons } from '../../icons/lucide-icons';

@Component({
  selector: 'mc-theme-toggle',
  standalone: true,
  imports: [LucideDynamicIcon],
  templateUrl: './theme-toggle.component.html',
  styleUrl: './theme-toggle.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class ThemeToggleComponent {
  protected readonly icons = appIcons;
  protected readonly themeService = inject(ThemeService);
}
