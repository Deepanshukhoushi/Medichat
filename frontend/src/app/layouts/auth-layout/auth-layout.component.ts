import { ChangeDetectionStrategy, Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';

import { AppLogoComponent } from '../../shared/components/app-logo/app-logo.component';
import { MedicalOrbComponent } from '../../shared/components/medical-orb/medical-orb.component';

@Component({
  selector: 'mc-auth-layout',
  standalone: true,
  imports: [RouterOutlet],
  templateUrl: './auth-layout.component.html',
  styleUrl: './auth-layout.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class AuthLayoutComponent {}
