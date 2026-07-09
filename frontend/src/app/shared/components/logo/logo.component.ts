import { ChangeDetectionStrategy, Component, Input } from '@angular/core';

@Component({
  selector: 'app-logo',
  standalone: true,
  template: `
    <svg [attr.width]="size" [attr.height]="size" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg">
      <g stroke="currentColor" class="text-accent opacity-70" stroke-width="1.3" fill="none">
        <line x1="30" y1="30" x2="30" y2="10"/>
        <line x1="30" y1="30" x2="30" y2="50"/>
        <line x1="30" y1="30" x2="10" y2="30"/>
        <line x1="30" y1="30" x2="50" y2="30"/>
        <line x1="30" y1="10" x2="50" y2="30"/>
        <line x1="10" y1="30" x2="30" y2="50"/>
      </g>
      <circle cx="30" cy="30" r="4.5" class="fill-accent"/>
      <circle cx="30" cy="10" r="3" class="fill-accent"/>
      <circle cx="30" cy="50" r="3" class="fill-accent"/>
      <circle cx="10" cy="30" r="3" class="fill-accent"/>
      <circle cx="50" cy="30" r="3" class="fill-accent"/>
    </svg>
  `,
  styles: [],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class LogoComponent {
  @Input() size: number | string = 32;
}
