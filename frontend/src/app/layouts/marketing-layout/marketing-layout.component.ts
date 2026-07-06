import { ChangeDetectionStrategy, Component } from '@angular/core';
import { RouterLink, RouterOutlet } from '@angular/router';

import { AppLogoComponent } from '../../shared/components/app-logo/app-logo.component';
import { ThemeToggleComponent } from '../../shared/components/theme-toggle/theme-toggle.component';

@Component({
  selector: 'mc-marketing-layout',
  standalone: true,
  imports: [RouterLink, RouterOutlet, AppLogoComponent, ThemeToggleComponent],
  templateUrl: './marketing-layout.component.html',
  styleUrl: './marketing-layout.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class MarketingLayoutComponent {}
