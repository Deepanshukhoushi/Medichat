import { ChangeDetectionStrategy, Component, input } from '@angular/core';
import { NgClass } from '@angular/common';

@Component({
  selector: 'mc-glass-card',
  standalone: true,
  imports: [NgClass],
  templateUrl: './glass-card.component.html',
  styleUrl: './glass-card.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class GlassCardComponent {
  readonly padding = input('p-6 sm:p-7');
  readonly extraClass = input('');
}
