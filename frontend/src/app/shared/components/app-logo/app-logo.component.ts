import { ChangeDetectionStrategy, Component, Input } from '@angular/core';
import { RouterLink } from '@angular/router';
import { NgClass } from '@angular/common';

@Component({
  selector: 'mc-app-logo',
  standalone: true,
  imports: [RouterLink, NgClass],
  templateUrl: './app-logo.component.html',
  styleUrl: './app-logo.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class AppLogoComponent {
  @Input() variant: 'auto' | 'light' | 'icon' = 'auto';
  @Input() size: number | string = 28;
}
