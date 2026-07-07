import { ChangeDetectionStrategy, Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { LucideDynamicIcon } from '@lucide/angular';

import { AppLogoComponent } from '../../shared/components/app-logo/app-logo.component';
import { appIcons } from '../../shared/icons/lucide-icons';

@Component({
  selector: 'mc-auth-layout',
  standalone: true,
  imports: [RouterOutlet, LucideDynamicIcon, AppLogoComponent],
  templateUrl: './auth-layout.component.html',
  styleUrl: './auth-layout.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class AuthLayoutComponent {
  protected readonly icons = appIcons;
}
