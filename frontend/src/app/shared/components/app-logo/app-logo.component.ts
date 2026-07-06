import { ChangeDetectionStrategy, Component } from '@angular/core';
import { RouterLink } from '@angular/router';
import { LucideDynamicIcon } from '@lucide/angular';

import { appIcons } from '../../icons/lucide-icons';

@Component({
  selector: 'mc-app-logo',
  standalone: true,
  imports: [RouterLink, LucideDynamicIcon],
  templateUrl: './app-logo.component.html',
  styleUrl: './app-logo.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class AppLogoComponent {
  protected readonly icons = appIcons;
}
