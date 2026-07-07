import { ChangeDetectionStrategy, Component, Input } from '@angular/core';
import { RouterLink } from '@angular/router';
import { LucideDynamicIcon } from '@lucide/angular';
import { CommonModule } from '@angular/common';

import { appIcons } from '../../icons/lucide-icons';

@Component({
  selector: 'mc-app-logo',
  standalone: true,
  imports: [RouterLink, LucideDynamicIcon, CommonModule],
  templateUrl: './app-logo.component.html',
  styleUrl: './app-logo.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class AppLogoComponent {
  @Input() variant: 'auto' | 'light' = 'auto';
  
  protected readonly icons = appIcons;
}
